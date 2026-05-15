"""Generate synthetic crónicas via OpenAI Batch API.

Batch API is 50% cheaper than realtime and has no per-minute rate limits.
Trade-off: results return asynchronously, typically in minutes-to-hours
(within 24h SLA).

Workflow:
    1. submit: build a JSONL request file and create a batch job.
    2. status: poll batch job state.
    3. fetch:  when job completes, download results and assemble pairs.jsonl.

Reads OPENAI_API_KEY from env. Never accepts it on the command line.

Usage:
    # Step 1: submit (returns batch_id, save it)
    python -m scripts.synth_batch submit \
        --in data/prompts/prompts.jsonl --max 5000 \
        --model gpt-4o-mini --out data/synthetic/batch_req.jsonl

    # Step 2: check status
    python -m scripts.synth_batch status --batch-id batch_xxx

    # Step 3: fetch results (when status==completed)
    python -m scripts.synth_batch fetch --batch-id batch_xxx \
        --out data/synthetic/pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

from scripts.synth_cronicas import SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

OPENAI_BASE = "https://api.openai.com/v1"
BATCH_STATE_FILE = Path("data/synthetic/batch_state.json")


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set. Add to ~/.zshrc or .env.")
    return key


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}


# ---------- submit ----------


def build_batch_jsonl(in_path: Path, out_path: Path, model: str, max_n: int) -> int:
    """Write a Batch-API JSONL request file. Returns number of requests."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if n >= max_n:
                break
            rec = json.loads(line)
            persona = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
            req = {
                "custom_id": f"match-{rec['match_id']}-style-{persona['label_public']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": persona["prompt"]},
                        {"role": "user", "content": rec["stats_block"] + "\n\nEscribe la crónica."},
                    ],
                    "temperature": 0.9,
                    "max_tokens": 500,
                },
            }
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")
            n += 1
    return n


def submit_batch(jsonl_path: Path, model: str) -> dict:
    """Upload the JSONL and create a batch job."""
    # 1. upload file
    with jsonl_path.open("rb") as f:
        files = {
            "file": (jsonl_path.name, f, "application/jsonl"),
            "purpose": (None, "batch"),
        }
        r = httpx.post(
            f"{OPENAI_BASE}/files",
            headers=_headers(),
            files=files,
            timeout=120.0,
        )
    r.raise_for_status()
    file_id = r.json()["id"]
    logger.info("Uploaded file: %s", file_id)

    # 2. create batch
    r = httpx.post(
        f"{OPENAI_BASE}/batches",
        headers={**_headers(), "Content-Type": "application/json"},
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {"project": "cronica-jax", "model": model},
        },
        timeout=60.0,
    )
    r.raise_for_status()
    job = r.json()
    logger.info("Submitted batch: %s (status=%s)", job["id"], job["status"])

    # Persist state for convenience
    BATCH_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_STATE_FILE.write_text(json.dumps(job, indent=2))
    return job


# ---------- status ----------


def get_status(batch_id: str) -> dict:
    r = httpx.get(f"{OPENAI_BASE}/batches/{batch_id}", headers=_headers(), timeout=30.0)
    r.raise_for_status()
    return r.json()


# ---------- fetch ----------


def fetch_results(batch_id: str, out_path: Path) -> int:
    """Download the output file and assemble pairs.jsonl."""
    job = get_status(batch_id)
    if job["status"] != "completed":
        raise SystemExit(f"Batch not done yet: status={job['status']}")
    out_file_id = job.get("output_file_id")
    if not out_file_id:
        raise SystemExit("No output_file_id on completed job; check OpenAI dashboard.")
    r = httpx.get(f"{OPENAI_BASE}/files/{out_file_id}/content",
                  headers=_headers(), timeout=300.0)
    r.raise_for_status()

    # Build label_public lookup by custom_id suffix.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    n_fail = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for line in r.text.splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row.get("custom_id", "")
            try:
                # custom_id format: "match-<mid>-style-<label_public>"
                _, mid, _, label = cid.split("-", 3)
                match_id = int(mid)
            except Exception:
                logger.warning("Unparseable custom_id: %s", cid)
                n_fail += 1
                continue
            resp = row.get("response", {}).get("body")
            if not resp or "choices" not in resp:
                n_fail += 1
                continue
            cronica = resp["choices"][0]["message"]["content"].strip()
            out_rec = {
                "match_id": match_id,
                "style_label": label,
                "style_token": f"<style:{label}>",
                "cronica": cronica,
                "model": job.get("metadata", {}).get("model", "unknown"),
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Fetched %d cronicas (failed %d) -> %s", n, n_fail, out_path)
    return n


# ---------- driver ----------


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--in", dest="inp", type=Path, required=True)
    p_sub.add_argument("--out", type=Path, default=Path("data/synthetic/batch_req.jsonl"))
    p_sub.add_argument("--model", default="gpt-4o-mini")
    p_sub.add_argument("--max", type=int, default=5000)

    p_st = sub.add_parser("status")
    p_st.add_argument("--batch-id", required=False, default=None)
    p_st.add_argument("--watch", action="store_true",
                      help="Poll every 30s until completed or failed.")

    p_ft = sub.add_parser("fetch")
    p_ft.add_argument("--batch-id", required=True)
    p_ft.add_argument("--out", type=Path, default=Path("data/synthetic/pairs.jsonl"))

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "submit":
        n = build_batch_jsonl(args.inp, args.out, args.model, args.max)
        logger.info("Wrote %d requests to %s", n, args.out)
        job = submit_batch(args.out, args.model)
        print(f"\nBATCH_ID = {job['id']}\nstatus  = {job['status']}\nCheck:    python -m scripts.synth_batch status --batch-id {job['id']} --watch")
    elif args.cmd == "status":
        bid = args.batch_id
        if not bid and BATCH_STATE_FILE.exists():
            bid = json.loads(BATCH_STATE_FILE.read_text())["id"]
        if not bid:
            raise SystemExit("Pass --batch-id or run submit first.")
        while True:
            job = get_status(bid)
            counts = job.get("request_counts", {})
            print(f"{time.strftime('%H:%M:%S')}  status={job['status']:12s} "
                  f"total={counts.get('total','?')} "
                  f"completed={counts.get('completed','?')} "
                  f"failed={counts.get('failed','?')}")
            if job["status"] in ("completed", "failed", "expired", "cancelled"):
                break
            if not args.watch:
                break
            time.sleep(30)
    elif args.cmd == "fetch":
        fetch_results(args.batch_id, args.out)


if __name__ == "__main__":
    main()
