"""Generate synthetic Spanish crónicas from <STATS> prompts using OpenAI.

Input:  data/prompts/prompts.jsonl
Output: data/synthetic/pairs.jsonl  ({match_id, stats_block, cronica, model})

Reads API key from OPENAI_API_KEY env var. NEVER pass it on the command line.

Concurrent async requests with bounded semaphore. Saves output incrementally
so if you Ctrl+C halfway, the partial JSONL is intact.

Usage:
    export OPENAI_API_KEY=sk-...
    python -m scripts.synth_cronicas \
        --in data/prompts/prompts.jsonl \
        --out data/synthetic/pairs.jsonl \
        --model gpt-4o-mini --max 5000 --concurrency 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)


# 8 commentator personas (ADR-002).
# `persona_internal` is used only to flavor the LLM prompt and is NOT stored
# in the published dataset. `label_public` is the user-facing style label and
# is the only one that appears in the demo / HF Hub.
SYSTEM_PROMPTS: list[dict] = [
    {
        "persona_internal": "Andrés Cantor",
        "label_public": "rioplatense_apasionado",
        "prompt": (
            "Eres Andrés Cantor, narrador argentino apasionado conocido por su "
            "'GOOOL' alargado y su tono emotivo en momentos cumbre. Escribe la "
            "crónica del partido en español rioplatense, ~200 palabras, con tu "
            "estilo expresivo característico. Usa SOLO la información de <STATS>; "
            "no inventes goles, jugadores ni marcadores. Menciona los goles en "
            "orden cronológico."
        ),
    },
    {
        "persona_internal": "Mariano Closs",
        "label_public": "rioplatense_tecnico",
        "prompt": (
            "Eres Mariano Closs, narrador argentino reconocido por su tono "
            "técnico, su cadencia equilibrada y la precisión en el análisis. "
            "Escribe la crónica en español rioplatense, ~200 palabras, narrador "
            "profesional. Usa SOLO <STATS>. Goles en orden cronológico."
        ),
    },
    {
        "persona_internal": "Christian Martinoli",
        "label_public": "mexicano_irreverente",
        "prompt": (
            "Eres Christian Martinoli, comentarista mexicano famoso por su "
            "irreverencia, frases ingeniosas y crítica afilada. Crónica en "
            "español mexicano, ~200 palabras, con tu estilo característico. "
            "Sólo datos de <STATS>; no inventes."
        ),
    },
    {
        "persona_internal": "Pablo Ramírez",
        "label_public": "mexicano_clasico",
        "prompt": (
            "Eres Pablo Ramírez, narrador mexicano de estilo formal y clásico. "
            "Crónica en español mexicano neutro, ~200 palabras, periodístico. "
            "Sólo datos de <STATS>."
        ),
    },
    {
        "persona_internal": "Fernando Palomo",
        "label_public": "centroamericano_espn",
        "prompt": (
            "Eres Fernando Palomo, narrador salvadoreño de ESPN, conocido por su "
            "tono pulido, voz internacional y estilo periodístico equilibrado. "
            "Crónica en español neutro latinoamericano, ~200 palabras. "
            "Sólo datos de <STATS>."
        ),
    },
    {
        "persona_internal": "Manolo Lama",
        "label_public": "espanol_radiofonico",
        "prompt": (
            "Eres Manolo Lama, periodista radiofónico español de la COPE, estilo "
            "tradicional y descriptivo. Crónica en español peninsular, ~200 "
            "palabras, tono de cabina radial. Sólo datos de <STATS>."
        ),
    },
    {
        "persona_internal": "Víctor Hugo Morales",
        "label_public": "rioplatense_literario",
        "prompt": (
            "Eres Víctor Hugo Morales, narrador uruguayo conocido por su tono "
            "literario, sus metáforas (el 'barrilete cósmico') y su pasión "
            "narrativa. Crónica en español rioplatense, ~200 palabras, prosa "
            "evocativa. Sólo datos de <STATS>."
        ),
    },
    {
        "persona_internal": "Diego Latorre",
        "label_public": "comentario_tecnico",
        "prompt": (
            "Eres Diego Latorre, comentarista argentino especializado en análisis "
            "táctico. Crónica con enfoque en sistemas, transiciones, momentos "
            "tácticos, ~200 palabras. Sólo datos de <STATS>."
        ),
    },
]


async def generate_one(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    stats_block: str,
    seed: int,
    timeout: float,
) -> tuple[str, dict] | None:
    """Returns (cronica_text, persona_metadata) or None on failure."""
    persona = SYSTEM_PROMPTS[seed % len(SYSTEM_PROMPTS)]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": persona["prompt"]},
            {"role": "user", "content": stats_block + "\n\nEscribe la crónica."},
        ],
        "temperature": 0.9,
        "max_tokens": 500,
    }
    for attempt in range(1, 5):
        try:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=timeout,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"].strip()
                return content, persona
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(2**attempt + random.random())
                continue
            logger.error("HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            await asyncio.sleep(2**attempt + random.random())
    return None


async def run(
    in_path: Path, out_path: Path, model: str, max_n: int,
    concurrency: int, timeout: float,
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=sk-... (or put "
            "it in your .env / .zshrc). DO NOT paste it in chat."
        )

    # Load prompts and skip already-done match_ids.
    done: set[int] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done.add(int(json.loads(line)["match_id"]))
            except Exception:
                continue
        logger.info("Resuming: %d already done", len(done))

    todo: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["match_id"]) in done:
                continue
            todo.append(rec)
            if len(todo) >= max_n:
                break

    logger.info("Will generate %d new crónicas with model=%s, concurrency=%d",
                len(todo), model, concurrency)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("a", encoding="utf-8")
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)
    counter = {"ok": 0, "fail": 0, "start": time.time()}

    async with httpx.AsyncClient() as client:
        async def worker(rec: dict, i: int):
            async with sem:
                result = await generate_one(client, api_key, model,
                                            rec["stats_block"], seed=i,
                                            timeout=timeout)
                if result is None:
                    counter["fail"] += 1
                    return
                cronica, persona = result
                # Only label_public is exposed in the published dataset.
                # persona_internal stays out of the published JSONL.
                out_rec = {
                    "match_id": rec["match_id"],
                    "stats_block": rec["stats_block"],
                    "style_label": persona["label_public"],
                    "style_token": f"<style:{persona['label_public']}>",
                    "cronica": cronica,
                    "model": model,
                }
                async with write_lock:
                    out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    counter["ok"] += 1
                    if counter["ok"] % 50 == 0:
                        rate = counter["ok"] / max(1.0, time.time() - counter["start"])
                        logger.info("  ok=%d fail=%d  %.1f cron/sec",
                                    counter["ok"], counter["fail"], rate)

        await asyncio.gather(*(worker(r, i) for i, r in enumerate(todo)))

    out_f.close()
    rate = counter["ok"] / max(1.0, time.time() - counter["start"])
    logger.info("DONE  ok=%d  fail=%d  avg %.1f cron/sec",
                counter["ok"], counter["fail"], rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path,
                        default=Path("data/prompts/prompts.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/synthetic/pairs.jsonl"))
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="OpenAI chat model. Try gpt-4o-mini for cost, "
                             "or any current mini-tier model.")
    parser.add_argument("--max", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run(args.inp, args.out, args.model, args.max,
                    args.concurrency, args.timeout))


if __name__ == "__main__":
    main()
