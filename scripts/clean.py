"""Clean and deduplicate raw scraped JSONL files.

Steps:
  1. Load all raw/*.jsonl.
  2. Filter:
     - language must be Spanish (langdetect or simple n-gram heuristic).
     - word count in [150, 2000].
     - must contain at least one scoreline OR one football-event verb.
  3. Dedup:
     - exact: content_hash already on each record.
     - near-dup: MinHash + LSH (Jaccard >= 0.85 -> drop).
  4. Write cleaned JSONL to data/clean/cronicas.jsonl.

Usage:
    python -m scripts.clean --out data/clean/cronicas.jsonl
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

from scripts.common import CLEAN_DIR, RAW_DIR, read_jsonl, write_jsonl, Cronica

logger = logging.getLogger(__name__)

MIN_WORDS = 150
MAX_WORDS = 2000

_SCORE_RE = re.compile(r"\b\d{1,2}[-–]\d{1,2}\b")
_FUTBOL_VERBS = re.compile(
    r"\b(gol(es)?|tarjeta|amarilla|roja|penal|c[oó]rner|tiro libre|"
    r"partido|equipo|primer tiempo|segundo tiempo|local|visitante|"
    r"derrot|venc|empat|gan[óo]|perdi[óo]|expulsa)\b",
    re.I,
)


def is_spanish(text: str) -> bool:
    """Cheap Spanish detector via diacritic & stopword density.

    Avoids adding langdetect as a hard dep for the scraper step.
    """
    if not text or len(text) < 100:
        return False
    sample = text[:1500].lower()
    es_markers = sum(sample.count(w) for w in (
        " que ", " de ", " el ", " la ", " los ", " las ", " un ", " una ",
        " con ", " por ", " para ", " en el ", " del ",
    ))
    en_markers = sum(sample.count(w) for w in (
        " the ", " of ", " and ", " is ", " was ", " for ", " with ",
    ))
    has_diacritics = any(c in sample for c in "áéíóúñ¿¡")
    return es_markers >= 5 and es_markers > en_markers and has_diacritics


def passes_quality(text: str) -> bool:
    n_words = len(text.split())
    if not (MIN_WORDS <= n_words <= MAX_WORDS):
        return False
    if not is_spanish(text):
        return False
    if not (_SCORE_RE.search(text) or _FUTBOL_VERBS.search(text)):
        return False
    return True


# ---------- near-dup via MinHash ----------


def shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def near_dedup(records: list[dict], threshold: float = 0.85) -> list[dict]:
    """O(N^2) near-dedup. Fine for <=20k records; switch to LSH if larger."""
    if len(records) > 30_000:
        logger.warning("Skipping near-dedup: %d records > 30k threshold", len(records))
        return records
    keep: list[dict] = []
    kept_shingles: list[set[str]] = []
    for rec in records:
        sh = shingles(rec["cronica"])
        is_dup = False
        for prev_sh in kept_shingles:
            if jaccard(sh, prev_sh) >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(rec)
            kept_shingles.append(sh)
    logger.info("Near-dedup: %d -> %d (dropped %d)", len(records), len(keep), len(records) - len(keep))
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=CLEAN_DIR / "cronicas.jsonl")
    parser.add_argument("--jaccard", type=float, default=0.85)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    all_records: list[dict] = []
    for fp in sorted(args.raw_dir.glob("*.jsonl")):
        logger.info("Loading %s", fp)
        for rec in read_jsonl(fp):
            all_records.append(rec)

    logger.info("Loaded %d raw records", len(all_records))

    # 1. quality filter
    pre = len(all_records)
    all_records = [r for r in all_records if passes_quality(r["cronica"])]
    logger.info("Quality filter: %d -> %d", pre, len(all_records))

    # 2. exact dedup by content_hash
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in all_records:
        h = r["content_hash"]
        if h in seen:
            continue
        seen.add(h)
        deduped.append(r)
    logger.info("Exact dedup: %d -> %d", len(all_records), len(deduped))

    # 3. near-dup
    final = near_dedup(deduped, threshold=args.jaccard)

    # Report distribution
    by_liga = Counter(r["liga"] for r in final)
    by_fuente = Counter(r["fuente"] for r in final)
    logger.info("Final by liga: %s", dict(by_liga))
    logger.info("Final by fuente: %s", dict(by_fuente))

    # Write as Cronica objects so __post_init__ re-validates n_palabras/hash.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cronicas = (Cronica(**{k: r[k] for k in (
        "liga", "temporada", "fecha", "local", "visitante", "resultado",
        "fuente", "url", "titulo", "cronica", "licencia",
    )}) for r in final)
    write_jsonl(cronicas, args.out)


if __name__ == "__main__":
    main()
