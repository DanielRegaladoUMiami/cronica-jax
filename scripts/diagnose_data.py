"""Deep-dive diagnostic over pairs.jsonl + tokenizer.json.

Checks both data quality and tokenization correctness before training.

  DATA QUALITY
    - length distributions
    - per-style stats and lexical fingerprint
    - duplicate detection
    - hallucination check (scorers in cronica must appear in stats)
    - team-name grounding (home/away teams must appear in cronica)
    - score grounding (final score must appear in cronica)
    - language sanity

  TOKEN CREATION
    - all special tokens are recoverable single ids
    - <unk> rate over the entire corpus
    - average tokens/word ratio
    - encode -> decode roundtrip preserves structure
    - target-mask construction yields plausible spans

Usage:
    python -m scripts.diagnose_data
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from tokenizers import Tokenizer

PAIRS = Path("data/synthetic/pairs.jsonl")
TOK = Path("tokenizer.json")
STYLES = [
    "rioplatense_apasionado", "rioplatense_tecnico", "rioplatense_literario",
    "mexicano_irreverente", "mexicano_clasico",
    "centroamericano_espn", "espanol_radiofonico", "comentario_tecnico",
]


def load_pairs() -> list[dict]:
    return [json.loads(l) for l in PAIRS.open()]


# ---------- 1. data quality ----------


_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b")
_MIN_RE = re.compile(r"^\s+-\s+(\d{1,3})'\s+(.+?)\s+\((.+?)\)$", re.M)


def extract_stats_from_block(block: str) -> dict:
    out = {"local": None, "visitante": None, "resultado": None, "goles": []}
    for line in block.splitlines():
        if line.startswith("local: "):
            out["local"] = line[len("local: "):].strip()
        elif line.startswith("visitante: "):
            out["visitante"] = line[len("visitante: "):].strip()
        elif line.startswith("resultado: "):
            out["resultado"] = line[len("resultado: "):].strip()
    for m in _MIN_RE.finditer(block):
        out["goles"].append((m.group(1), m.group(2), m.group(3)))
    return out


def check_grounding(rec: dict) -> dict:
    """Return per-record checks; True = grounded."""
    info = extract_stats_from_block(rec["stats_block"])
    cron = rec["cronica"]
    cron_norm = cron.lower()

    # team name appears in cronica?
    local_ok = info["local"] and any(
        tok in cron_norm for tok in info["local"].lower().split() if len(tok) > 3
    )
    visit_ok = info["visitante"] and any(
        tok in cron_norm for tok in info["visitante"].lower().split() if len(tok) > 3
    )

    # final score appears (in some normalized form)?
    score_ok = info["resultado"] and (
        info["resultado"] in cron
        or info["resultado"].replace("-", "–") in cron
        or info["resultado"].replace("-", " a ") in cron
    )

    # scorers: each player in stats appears at least once in cronica
    scorers_match = 0
    scorers_total = 0
    for _, player, _ in info["goles"]:
        scorers_total += 1
        # match by last name (most reliable token)
        last = player.split()[-1].lower()
        if last and last in cron_norm:
            scorers_match += 1

    # alucinations: detect player-name mentions in cronica that aren't in stats?
    # (heuristic: capitalized two-word names; very imperfect, just signal)
    cron_caps = set(re.findall(r"\b[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+\b", cron))
    stats_names = set()
    for _, player, _ in info["goles"]:
        stats_names.add(player)
    halluc_candidates = cron_caps - stats_names

    return {
        "local_ok": bool(local_ok),
        "visitante_ok": bool(visit_ok),
        "score_ok": bool(score_ok),
        "scorers_match": scorers_match,
        "scorers_total": scorers_total,
        "halluc_candidates": len(halluc_candidates),
    }


def data_quality_report(pairs: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  DATA QUALITY")
    print("=" * 72)

    cron_lens = [len(p["cronica"].split()) for p in pairs]
    print(f"  Crónica words: avg={mean(cron_lens):.0f} median={median(cron_lens):.0f} "
          f"min={min(cron_lens)} max={max(cron_lens)}")

    # exact duplicates
    seen = Counter(p["cronica"][:120] for p in pairs)
    dups = {k: v for k, v in seen.items() if v > 1}
    print(f"  Exact 120-char-prefix dups: {len(dups)} (count > 1)")

    # per-style stats
    print("\n  Per-style sample lengths and lexical fingerprint:")
    for style in STYLES:
        recs = [p for p in pairs if p["style_label"] == style]
        if not recs:
            continue
        wc = [len(r["cronica"].split()) for r in recs]
        # find distinctive words for this style (heuristic)
        from collections import Counter as C
        all_words = " ".join(r["cronica"] for r in recs).lower()
        words = re.findall(r"\b[a-záéíóúñü]+\b", all_words)
        cnt = C(w for w in words if len(w) > 3 and w not in {
            "para","con","como","más","pero","sus","muy","esta","tras","fue",
            "este","entre","desde","sobre","durante","hasta","cuando",
            "partido","equipo","minuto","minutos","jugador","gol","goles",
            "primer","primera","segundo","segunda","final","liga","torneo",
        })
        top3 = ", ".join(w for w, _ in cnt.most_common(5))
        print(f"    {style:32s}  n={len(recs):4d}  avg_words={mean(wc):3.0f}  top: {top3}")

    # Grounding checks across whole corpus
    print("\n  Grounding (does the crónica reflect the stats?):")
    rs = [check_grounding(p) for p in pairs]
    n = len(rs)
    local_ok = sum(r["local_ok"] for r in rs)
    visit_ok = sum(r["visitante_ok"] for r in rs)
    score_ok = sum(r["score_ok"] for r in rs)
    scorers_match = sum(r["scorers_match"] for r in rs)
    scorers_total = sum(r["scorers_total"] for r in rs)
    print(f"    Home team named:                 {local_ok}/{n}  ({100*local_ok/n:.1f}%)")
    print(f"    Away team named:                 {visit_ok}/{n}  ({100*visit_ok/n:.1f}%)")
    print(f"    Final score appears:             {score_ok}/{n}  ({100*score_ok/n:.1f}%)")
    print(f"    Scorers mentioned (last-name):   {scorers_match}/{scorers_total}  "
          f"({100*scorers_match/scorers_total:.1f}%)")
    avg_halluc = mean(r["halluc_candidates"] for r in rs)
    print(f"    Avg unfamiliar capitalized names per crónica: {avg_halluc:.1f} "
          f"(heuristic — high values = potential hallucinations)")


# ---------- 2. tokenization quality ----------


def tokenization_report(pairs: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  TOKENIZATION")
    print("=" * 72)

    tok = Tokenizer.from_file(str(TOK))
    vocab = tok.get_vocab()
    print(f"  Vocab size: {len(vocab)}")

    # Special tokens reachable as single IDs?
    print("\n  Special tokens reachable:")
    special = [
        "<pad>", "<bos>", "<eos>", "<unk>",
        "<stats>", "</stats>", "<cronica>", "</cronica>",
        *[f"<style:{s}>" for s in STYLES],
    ]
    for t in special:
        ids = tok.encode(t).ids
        single = (len(ids) == 1)
        flag = "✓" if single else "✗"
        print(f"    {flag} {t:36s} -> ids={ids} {'single' if single else 'SPLIT'}")

    # <unk> rate over corpus
    unk_id = vocab["<unk>"]
    n_unk = 0
    n_tok = 0
    for p in pairs:
        ids = tok.encode(p["cronica"]).ids
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == unk_id)
    print(f"\n  Total cronica tokens: {n_tok:,}")
    print(f"  <unk> count:          {n_unk:,} ({100*n_unk/max(n_tok,1):.3f}%)")

    # Tokens-per-word
    words = sum(len(p["cronica"].split()) for p in pairs)
    print(f"  Tokens / word:        {n_tok/words:.2f}")

    # Length distribution of FULL training example
    full_lens = []
    for p in pairs:
        full = f"<stats>{p['stats_block']}</stats>\n{p['style_token']}\n<cronica>{p['cronica']}</cronica>"
        full_lens.append(len(tok.encode(full).ids))
    full_lens.sort()
    print(f"\n  Full training example token lengths:")
    print(f"    avg={mean(full_lens):.0f}  median={median(full_lens):.0f}  "
          f"p90={full_lens[int(0.90*len(full_lens))]}  "
          f"p99={full_lens[int(0.99*len(full_lens))]}  max={max(full_lens)}")

    # Round-trip
    print("\n  Encode -> decode roundtrip on a representative example:")
    p = pairs[0]
    full = f"<stats>{p['stats_block']}</stats>\n{p['style_token']}\n<cronica>{p['cronica']}</cronica>"
    enc = tok.encode(full)
    dec = tok.decode(enc.ids, skip_special_tokens=False)
    same_words = len(re.findall(r"\w+", full)) == len(re.findall(r"\w+", dec))
    print(f"    Original word count: {len(re.findall(r'\w+', full))}")
    print(f"    Decoded  word count: {len(re.findall(r'\w+', dec))}")
    print(f"    Match: {'✓' if same_words else '✗ (potential info loss)'}")

    # Check that loss-mask boundary is well-defined
    cron_open_id = vocab["<cronica>"]
    cron_close_id = vocab["</cronica>"]
    boundary_ok = 0
    for p in pairs:
        full = f"<stats>{p['stats_block']}</stats>\n{p['style_token']}\n<cronica>{p['cronica']}</cronica>"
        ids = tok.encode(full).ids
        if cron_open_id in ids and cron_close_id in ids:
            if ids.index(cron_close_id) > ids.index(cron_open_id):
                boundary_ok += 1
    print(f"\n  Loss-mask boundary detectable: {boundary_ok}/{len(pairs)} "
          f"({100*boundary_ok/len(pairs):.1f}%)")


def main() -> None:
    pairs = load_pairs()
    print(f"\nLoaded {len(pairs)} pairs from {PAIRS}")
    data_quality_report(pairs)
    tokenization_report(pairs)
    print("\n" + "=" * 72)
    print("  END OF DIAGNOSTIC")
    print("=" * 72)


if __name__ == "__main__":
    main()
