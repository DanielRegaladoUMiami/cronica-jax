"""Build train/val/test parquet files from the cleaned JSONL and push to HF Hub.

Splits:
  - test: 1000 rows held out, stratified by (liga, fuente).
  - val:  500 rows held out, same stratification.
  - train: remainder.

Pushes to: DanielRegaladoCardoso/cronicas-futbol-latam

Usage:
    python -m scripts.build_dataset --in data/clean/cronicas.jsonl --push
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

from scripts.common import CLEAN_DIR, read_jsonl

logger = logging.getLogger(__name__)

HF_REPO = "DanielRegaladoCardoso/cronicas-futbol-latam"


def stratified_split(records: list[dict], n_test: int, n_val: int, seed: int = 42):
    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["liga"], r["fuente"])].append(r)
    for v in buckets.values():
        rng.shuffle(v)

    total = len(records)
    test_frac = n_test / total
    val_frac = n_val / total

    test, val, train = [], [], []
    for v in buckets.values():
        n = len(v)
        n_t = max(1, int(round(n * test_frac))) if total > n_test else 0
        n_v = max(1, int(round(n * val_frac))) if total > n_val else 0
        test.extend(v[:n_t])
        val.extend(v[n_t:n_t + n_v])
        train.extend(v[n_t + n_v:])
    return train, val, test


def to_parquet(records: list[dict], path: Path) -> None:
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info("Wrote %d rows to %s (%.1f MB)", len(df), path, path.stat().st_size / 1e6)


def push_to_hub(out_dir: Path, repo: str, private: bool = False) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=private)
    for fname in ("train.parquet", "val.parquet", "test.parquet", "README.md"):
        fpath = out_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=("data/" + fname) if fname.endswith(".parquet") else fname,
                repo_id=repo,
                repo_type="dataset",
            )
    logger.info("Pushed dataset to https://huggingface.co/datasets/%s", repo)


DATA_CARD_TEMPLATE = """---
license: cc-by-sa-4.0
language:
  - es
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
tags:
  - football
  - soccer
  - sports
  - latin-america
  - spanish
  - journalism
---

# cronicas-futbol-latam

Curated corpus of Spanish-language football match reports (crónicas) from
under-covered Latin American leagues — Primera Nacional Argentina, Liga de
Expansión MX, Primera B Colombia, Segunda Uruguay, Primera B Chile, etc.

Built as training data for [cronica-jax](https://github.com/DanielRegaladoUMiami/cronica-jax),
a 25M-parameter mini-LLM implemented from scratch in pure JAX.

## Splits

| split | rows |
|-------|------|
| train | {n_train} |
| val   | {n_val} |
| test  | {n_test} |

Stratified by `(liga, fuente)`.

## Schema

| field         | type   | description |
|---------------|--------|-------------|
| liga          | str    | League slug (e.g. `primera_nacional_arg`). |
| temporada     | str    | Season (e.g. `2024`, `2024-2025`); may be `""`. |
| fecha         | str    | ISO date if known. |
| local         | str    | Home team. |
| visitante     | str    | Away team. |
| resultado     | str    | Scoreline (`2-1`). |
| fuente        | str    | Source: `wikinoticias`, `wikipedia`, `ole`, `promiedos`, `record`. |
| url           | str    | Source URL (attribution). |
| titulo        | str    | Article title. |
| cronica       | str    | The narrative body. |
| licencia      | str    | `CC-BY-SA-4.0` or `fair-use-research`. |
| n_palabras    | int    | Word count of `cronica`. |
| content_hash  | str    | SHA-256 (16 hex) for dedup. |

## Sources & licenses

- **Wikinoticias ES** and **Wikipedia ES** content is **CC-BY-SA-4.0**.
  Attribution: Wikimedia contributors; redistributed here under the same license.
- **Olé**, **Promiedos**, **Récord** content is included under a
  **fair-use-research** label for non-commercial language-modeling research.
  Each row carries the original URL for attribution. If you are a rightsholder
  and want content removed, open an issue on
  [the repo](https://github.com/DanielRegaladoUMiami/cronica-jax/issues) — items
  will be removed within 72h.

## Cleaning pipeline

1. Per-source scraping with `robots.txt` respected and ~1–2s/req rate limiting.
2. Quality filters: length ∈ [150, 2000] words, Spanish detected via diacritic
   + stopword heuristic, must contain scoreline or football verb.
3. Exact dedup by content hash.
4. Near-dedup via 5-gram Jaccard (threshold 0.85).

## Limitations

- Most rows lack structured `local`/`visitante`/`fecha` fields (extracting them
  reliably from prose is left as future work).
- Source mix is uneven; Argentina is over-represented.
- Web scraping was limited to a weekend, so coverage is partial.

## Citation

If you use this dataset, please cite:

```
@misc{regalado2026cronicasfutbol,
  author = {Daniel Regalado},
  title  = {cronicas-futbol-latam: Spanish-language match reports
            for underserved LATAM football leagues},
  year   = {2026},
  url    = {https://huggingface.co/datasets/DanielRegaladoCardoso/cronicas-futbol-latam}
}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=CLEAN_DIR / "cronicas.jsonl")
    parser.add_argument("--out-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = list(read_jsonl(args.inp))
    logger.info("Loaded %d cleaned records", len(records))

    train, val, test = stratified_split(records, args.n_test, args.n_val)
    logger.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    to_parquet(train, args.out_dir / "train.parquet")
    to_parquet(val, args.out_dir / "val.parquet")
    to_parquet(test, args.out_dir / "test.parquet")

    card = DATA_CARD_TEMPLATE.format(
        n_train=len(train), n_val=len(val), n_test=len(test),
    )
    (args.out_dir / "README.md").write_text(card, encoding="utf-8")

    if args.push:
        push_to_hub(args.out_dir, args.repo, private=args.private)


if __name__ == "__main__":
    main()
