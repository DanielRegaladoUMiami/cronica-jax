"""Push the synthetic crónicas dataset to Hugging Face Hub.

Validates BEFORE upload that:
  - pairs.jsonl is non-empty and parseable
  - all 8 styles are present and roughly balanced
  - all records have the 4 required fields
  - parquet roundtrip preserves row count and string content
  - HF Hub auth is functional

Only after all validations pass does it actually push.

Usage:
    python -m scripts.push_dataset \
        --pairs data/synthetic/pairs.jsonl \
        --repo DanielRegaladoCardoso/cronicas-d2t \
        --private  # remove flag for public
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"match_id", "stats_block", "cronica", "style_label", "style_token", "model"}
EXPECTED_STYLES = {
    "rioplatense_apasionado", "rioplatense_tecnico", "rioplatense_literario",
    "mexicano_irreverente", "mexicano_clasico",
    "centroamericano_espn", "espanol_radiofonico", "comentario_tecnico",
}


def validate_pairs(path: Path) -> list[dict]:
    """Hard fail if anything is off. Return parsed list on success."""
    assert path.exists(), f"FAIL: {path} does not exist"
    pairs: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"FAIL: line {i} not valid JSON: {e}")
            missing = REQUIRED_FIELDS - set(rec)
            assert not missing, f"FAIL: line {i} missing fields {missing}"
            for k in REQUIRED_FIELDS:
                v = rec.get(k)
                if isinstance(v, str):
                    assert len(v) > 0, f"FAIL: line {i} field {k!r} is empty"
            assert rec["style_label"] in EXPECTED_STYLES, (
                f"FAIL: line {i} unknown style {rec['style_label']!r}"
            )
            pairs.append(rec)

    assert len(pairs) >= 100, f"FAIL: only {len(pairs)} pairs — too few"

    # style distribution
    counts = Counter(r["style_label"] for r in pairs)
    missing_styles = EXPECTED_STYLES - set(counts)
    assert not missing_styles, f"FAIL: missing styles {missing_styles}"
    min_n, max_n = min(counts.values()), max(counts.values())
    ratio = max_n / min_n if min_n > 0 else float("inf")
    assert ratio < 3.0, f"FAIL: style imbalance ratio {ratio:.2f} (min={min_n} max={max_n})"

    # length sanity
    short = sum(1 for r in pairs if len(r["cronica"].split()) < 50)
    assert short < 0.05 * len(pairs), (
        f"FAIL: {short}/{len(pairs)} crónicas under 50 words (>5%)"
    )

    logger.info("Validation PASSED: %d pairs, %d styles balanced, "
                "all required fields present", len(pairs), len(counts))
    return pairs


def to_parquet(pairs: list[dict], parquet_path: Path) -> None:
    df = pd.DataFrame(pairs)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    # verify roundtrip
    df2 = pd.read_parquet(parquet_path)
    assert len(df) == len(df2), "FAIL: parquet roundtrip lost rows"
    assert df["match_id"].tolist() == df2["match_id"].tolist(), \
        "FAIL: parquet roundtrip changed match_id"
    assert df["cronica"].iloc[0] == df2["cronica"].iloc[0], \
        "FAIL: parquet roundtrip changed cronica text"
    logger.info("Parquet roundtrip verified: %d rows, %.1f MB",
                len(df), parquet_path.stat().st_size / 1e6)


DATA_CARD = """---
license: apache-2.0
language:
  - es
size_categories:
  - 1K<n<10K
task_categories:
  - text2text-generation
  - text-generation
tags:
  - football
  - soccer
  - sports
  - data-to-text
  - spanish
  - synthetic
  - latin-america
---

# cronicas-d2t — Synthetic Spanish Football Match Reports

Training data for [cronica-jax](https://github.com/DanielRegaladoUMiami/cronica-jax),
a from-scratch JAX mini-LLM that learns the **data-to-text** task: given a
structured `<STATS>` block describing a football match, produce a Spanish
crónica in one of 8 regional commentator styles.

## Provenance

- **Structured match data:** `davidcariboo/player-scores` on Kaggle
  (Transfermarkt, [CC0](https://www.kaggle.com/datasets/davidcariboo/player-scores)).
  82,073 matches with at least one goal were filtered down to 38,462 matches
  in a curated allow-list of 22 leagues that both `gpt-4o-mini` and a Spanish-
  speaking audience recognize.
- **Crónicas (target text):** generated synthetically with OpenAI
  `gpt-4o-mini` via the Batch API, using 8 prompt variants inspired by
  real broadcasters. The internal persona names are NOT included in this
  dataset — only the descriptive public label appears, to avoid likeness
  issues. See
  [docs/decisions/002-commentator-personas.md](https://github.com/DanielRegaladoUMiami/cronica-jax/blob/main/docs/decisions/002-commentator-personas.md).

## Schema

| field        | type   | description                                  |
|--------------|--------|----------------------------------------------|
| match_id     | int    | Transfermarkt game_id                         |
| stats_block  | string | The full `<STATS>` prompt block in Spanish   |
| style_label  | string | One of 8 commentator-style descriptive labels |
| style_token  | string | The `<style:LABEL>` token for the JAX model  |
| cronica      | string | The Spanish-language match report             |
| model        | string | Generating OpenAI model (e.g. `gpt-4o-mini`) |

## Style labels

| label                       | region / flavor                       |
|----------------------------|---------------------------------------|
| `rioplatense_apasionado`   | Argentina, emotive, "gol" lengthened  |
| `rioplatense_tecnico`      | Argentina, analytical                 |
| `rioplatense_literario`    | Uruguay, evocative prose              |
| `mexicano_irreverente`     | Mexico, sarcastic                     |
| `mexicano_clasico`         | Mexico, formal                        |
| `centroamericano_espn`     | El Salvador / ESPN Latam, polished    |
| `espanol_radiofonico`      | Spain, radio broadcast style          |
| `comentario_tecnico`       | tactical analysis                     |

## Quality notes

- **Grounding** (verified in `scripts/diagnose_data.py`):
  - 98.0% mention the home team
  - 97.5% mention the away team
  - 95.7% of scorers mentioned by last name
  - 70.7% print the final score literally (the rest use phrases like
    "venció por la mínima" or "tres goles a uno" — natural Spanish).
- **Length:** 215 avg words/crónica (min 175, max 309).
- **Duplicates:** zero exact 120-char-prefix duplicates.
- **Hallucinations:** ~3.6 unfamiliar capitalized names per crónica on
  average (managers, stadiums, derby names added as context by
  `gpt-4o-mini`). These are not in the `<STATS>` and a small model
  trained on this data may invent them at inference time.

## License

Apache 2.0. Source structured data is CC0 (Transfermarkt via Kaggle).
The synthetic crónicas are model-generated and redistributed under
Apache 2.0.

## Citation

```bibtex
@misc{regalado2026cronicasd2t,
  author = {Daniel Regalado},
  title  = {cronicas-d2t: A synthetic data-to-text corpus for Spanish
            football match reports},
  year   = {2026},
  url    = {https://huggingface.co/datasets/DanielRegaladoCardoso/cronicas-d2t}
}
```
"""


def push(parquet_path: Path, card_path: Path, repo: str, private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()

    # Sanity-check auth
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: huggingface_hub auth not configured. "
                         f"Run `hf auth login` first. Error: {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="dataset",
                    exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/pairs.parquet",
        repo_id=repo, repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo, repo_type="dataset",
    )
    logger.info("Pushed to https://huggingface.co/datasets/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("data/synthetic/pairs.jsonl"))
    parser.add_argument("--repo", default="DanielRegaladoCardoso/cronicas-d2t")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and build artifacts but do not push to HF.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = validate_pairs(args.pairs)
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        parquet_path = tdp / "pairs.parquet"
        card_path = tdp / "README.md"
        to_parquet(pairs, parquet_path)
        card_path.write_text(DATA_CARD, encoding="utf-8")

        if args.dry_run:
            logger.info("DRY RUN: validations passed, parquet built (%d rows). "
                        "Skipping HF upload.", len(pairs))
            return

        push(parquet_path, card_path, args.repo, args.private)


if __name__ == "__main__":
    main()
