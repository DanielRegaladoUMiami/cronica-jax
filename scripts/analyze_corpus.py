"""Diagnostic analysis of the cleaned corpus.

Validates whether the data is usable for training a small LM:
  - size: tokens, words, docs
  - quality: language, length distribution, football specificity
  - diversity: type-token ratio, top n-grams
  - duplication: near-dup that may have escaped cleaning
  - coverage: by source, league, year
  - readiness for tokenizer + model
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev


def main() -> None:
    path = Path("data/clean/cronicas.jsonl")
    recs = [json.loads(l) for l in path.open()]

    print(f"{'='*70}")
    print("  CORPUS DIAGNOSTIC — data/clean/cronicas.jsonl")
    print(f"{'='*70}\n")

    # ---------- size ----------
    n_docs = len(recs)
    word_counts = [r["n_palabras"] for r in recs]
    total_words = sum(word_counts)
    chars = sum(len(r["cronica"]) for r in recs)
    print("SIZE")
    print(f"  Docs: {n_docs}")
    print(f"  Total words:        {total_words:,}")
    print(f"  Total chars:        {chars:,}")
    print(f"  Avg / median words: {mean(word_counts):.0f} / {median(word_counts):.0f}")
    print(f"  Stdev words:        {stdev(word_counts):.0f}")
    print(f"  Min / Max words:    {min(word_counts)} / {max(word_counts)}")
    print()

    # ---------- coverage ----------
    by_fuente = Counter(r["fuente"] for r in recs)
    by_liga = Counter(r["liga"] for r in recs)
    by_licencia = Counter(r["licencia"] for r in recs)
    print("COVERAGE")
    print(f"  Source:   {dict(by_fuente)}")
    print(f"  League:   {dict(by_liga)}")
    print(f"  License:  {dict(by_licencia)}")
    print()

    # ---------- football specificity ----------
    score_re = re.compile(r"\b\d{1,2}[-–]\d{1,2}\b")
    futbol_verbs = re.compile(r"\b(gol|gan[óo]|derrot|venc|empat|partido|equipo|club|liga|copa|torneo|jugador|entrenador|técnico|director técnico|árbitro|estadio|cancha|delantero|defensor|arquero|portero)\b", re.I)
    has_score = sum(1 for r in recs if score_re.search(r["cronica"]))
    has_verb = sum(1 for r in recs if futbol_verbs.search(r["cronica"]))
    rich = sum(1 for r in recs if score_re.search(r["cronica"]) and len(futbol_verbs.findall(r["cronica"])) >= 3)
    print("FOOTBALL SPECIFICITY")
    print(f"  Docs with score:         {has_score}/{n_docs} ({100*has_score/n_docs:.0f}%)")
    print(f"  Docs with futbol verb:   {has_verb}/{n_docs} ({100*has_verb/n_docs:.0f}%)")
    print(f"  'Rich' docs (score + 3+ verbs):  {rich}/{n_docs} ({100*rich/n_docs:.0f}%)")
    print()

    # ---------- diversity ----------
    full = " ".join(r["cronica"] for r in recs).lower()
    tokens = re.findall(r"\w+", full)
    unique = set(tokens)
    ttr = len(unique) / len(tokens)
    print("LEXICAL DIVERSITY")
    print(f"  Total tokens (whitespace): {len(tokens):,}")
    print(f"  Unique types:              {len(unique):,}")
    print(f"  Type-token ratio:          {ttr:.3f}")
    print(f"  Estimated BPE tokens (~1.3x words): {int(total_words * 1.3):,}")
    print()

    # ---------- top words ----------
    stop = {"de", "la", "el", "que", "en", "y", "a", "los", "del", "las", "se", "un", "por",
            "con", "no", "una", "su", "para", "es", "al", "lo", "como", "más", "o", "pero",
            "sus", "le", "ya", "este", "ha", "esta", "fue", "ser", "son", "han", "muy", "sin"}
    top = Counter(t for t in tokens if t not in stop and len(t) > 2).most_common(25)
    print("TOP 25 CONTENT WORDS")
    for w, c in top:
        print(f"  {w:20s} {c:6d}")
    print()

    # ---------- duplication audit ----------
    # check if any 50-char prefix appears in >5 docs
    prefixes = Counter(r["cronica"][:60].lower() for r in recs)
    suspicious = {p: c for p, c in prefixes.items() if c > 2}
    print("DUPLICATION AUDIT")
    print(f"  Repeated 60-char prefixes (count > 2): {len(suspicious)}")
    for p, c in list(suspicious.items())[:5]:
        print(f"    [{c}x] {p[:80]}")
    print()

    # ---------- year coverage (extract years from text) ----------
    year_re = re.compile(r"\b(20\d{2})\b")
    years: Counter = Counter()
    for r in recs:
        for m in year_re.findall(r["cronica"][:500]):
            years[m] += 1
    print("YEAR MENTIONS (top 15 from first 500 chars of each doc)")
    for y, c in years.most_common(15):
        print(f"  {y}: {c}")
    print()

    # ---------- verdict ----------
    print(f"{'='*70}")
    print("  VERDICT")
    print(f"{'='*70}")
    est_tokens = int(total_words * 1.3)
    chinchilla_optimal = est_tokens / 20  # rough rule: 20 tokens per param
    print(f"  Estimated BPE tokens: ~{est_tokens:,}")
    print(f"  Chinchilla-optimal param count for this corpus: ~{chinchilla_optimal:.0f} params")
    print(f"    -> A {chinchilla_optimal/1e3:.0f}K param model would be 'data-balanced'.")
    print(f"  Our 5M target is ~{5_000_000/chinchilla_optimal:.0f}x over Chinchilla-optimal.")
    print(f"  -> Heavy memorization expected. Useful for proof-of-concept only.")
    print()
    print(f"  Sample / param ratio for various model sizes:")
    for p in [500_000, 1_000_000, 5_000_000, 25_000_000]:
        ratio = est_tokens / p
        verdict = "BALANCED" if ratio > 10 else "OVERFIT" if ratio < 1 else "MEMORIZE"
        print(f"    {p/1e6:5.1f}M params -> {ratio:5.2f} tokens/param  [{verdict}]")


if __name__ == "__main__":
    main()
