"""Evaluate the trained model on held-out matches and write results/samples.md.

Held-out matches are picked from prompts.jsonl with match_ids NOT seen in
pairs.jsonl. For each, we generate one crónica per chosen style and write
side-by-side OpenAI vs JAX outputs to results/samples.md.

Usage:
    python -m scripts.eval_heldout --ckpt checkpoints/run01/ckpt_002000.pkl \
        --n 20 --out results/samples.md
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from tokenizers import Tokenizer

from cronica.model import Config
from cronica.sample import generate_cronica
from cronica.train import load_ckpt

logger = logging.getLogger(__name__)

STYLES = [
    "rioplatense_apasionado",
    "rioplatense_literario",
    "mexicano_irreverente",
    "comentario_tecnico",
]


def find_heldout(prompts_path: Path, pairs_path: Path, n: int, seed: int = 7) -> list[dict]:
    """Return n prompt records whose match_id is NOT in pairs.jsonl."""
    train_ids: set[int] = set()
    with pairs_path.open() as f:
        for line in f:
            r = json.loads(line)
            train_ids.add(int(r["match_id"]))
    candidates: list[dict] = []
    with prompts_path.open() as f:
        for line in f:
            r = json.loads(line)
            if int(r["match_id"]) not in train_ids:
                candidates.append(r)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--prompts", type=Path,
                        default=Path("data/prompts/prompts.jsonl"))
    parser.add_argument("--pairs", type=Path,
                        default=Path("data/synthetic/pairs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/samples.md"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--styles", nargs="+", default=STYLES)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params, cfg, step = load_ckpt(args.ckpt)
    tok = Tokenizer.from_file(str(args.tokenizer))

    heldout = find_heldout(args.prompts, args.pairs, args.n, seed=args.seed)
    logger.info("Selected %d held-out matches (from %s)", len(heldout), args.prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("# cronica-jax — held-out samples\n\n")
        f.write(f"Checkpoint: `{args.ckpt}` (step {step})\n\n")
        f.write(f"Generated {len(heldout)} held-out crónicas, one per match per "
                f"chosen style. **None of these matches were in the training "
                f"set.**\n\n")
        f.write(f"Sampling: temperature={args.temperature}, top_k={args.top_k}, "
                f"top_p={args.top_p}, max_new_tokens={args.max_new_tokens}.\n\n")
        f.write("---\n\n")

        for i, rec in enumerate(heldout, 1):
            f.write(f"## Match {i} — id={rec['match_id']}\n\n")
            f.write("**Stats**\n\n```\n" + rec["stats_block"] + "\n```\n\n")
            for style in args.styles:
                logger.info("Match %d/%d  style=%s", i, len(heldout), style)
                text = generate_cronica(
                    params, cfg, tok,
                    rec["stats_block"], style,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    seed=args.seed + i,
                )
                f.write(f"### Style: `{style}`\n\n{text}\n\n")
            f.write("---\n\n")
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
