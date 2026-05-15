"""Compute grounding and style-fidelity metrics over results/samples.md.

Parses the markdown produced by eval_heldout.py and reports:

  GROUNDING
    - % crónicas that mention the home team (by any token len>3)
    - % crónicas that mention the away team
    - % crónicas that print the literal final score
    - % of scorers (by last name) actually referenced in each crónica

  STYLE FIDELITY
    - % crónicas with the expected stylistic marker present
      (e.g. 'gooool' for rioplatense_apasionado, 'barrilete' for literario,
       'transiciones'/'sistema' for tecnico, etc.)

Writes a machine-readable summary to results/metrics.json and a
human-readable summary to stdout.

Usage:
    python -m scripts.compute_metrics --samples results/samples.md
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


# Distinctive vocabulary per style. Presence of any of these tokens in the
# crónica counts as "style activated". This is a heuristic, not a label —
# it tracks whether the model learned the *flavor* of each persona.
STYLE_MARKERS: dict[str, list[str]] = {
    "rioplatense_apasionado":  ["gooool", "goool", "¡goool", "¡qué partido", "qué espectáculo"],
    "rioplatense_tecnico":     ["la jerarquía", "marcador", "ventaja", "encuentro"],
    "rioplatense_literario":   ["barrilete", "alma", "magia", "sinfonía", "poema"],
    "mexicano_irreverente":    ["telenovela", "salud por", "como si", "qué tal", "ni"],
    "mexicano_clasico":        ["en la tarde", "afición", "dirección", "encuentro"],
    "centroamericano_espn":    ["dominical", "los diablos", "espn", "mítico"],
    "espanol_radiofonico":     ["cope", "buenas noches", "oyentes", "señor"],
    "comentario_tecnico":      ["transiciones", "sistema", "estructura", "presión alta", "bloque"],
}


# ---------- markdown parsing ----------


SECTION_RE = re.compile(r"^## Match \d+ — id=(\d+)$", re.M)
STATS_BLOCK_RE = re.compile(r"```\n(<STATS>.*?</STATS>)\n```", re.S)
STYLE_HEADING_RE = re.compile(r"^### Style: `([^`]+)`$", re.M)


def parse_samples(md_text: str) -> list[dict]:
    """Return list of {match_id, stats_block, style, cronica}."""
    out = []

    # Split by match section
    parts = re.split(r"^## Match \d+ — id=", md_text, flags=re.M)
    # First chunk is preamble; skip.
    for chunk in parts[1:]:
        # match_id is the digits before \n
        m_id = re.match(r"(\d+)", chunk)
        if not m_id:
            continue
        match_id = int(m_id.group(1))
        stats_m = STATS_BLOCK_RE.search(chunk)
        if not stats_m:
            continue
        stats_block = stats_m.group(1)

        # Split this match's chunk by style headings
        style_chunks = re.split(r"^### Style: `([^`]+)`$", chunk, flags=re.M)
        # Pattern: [pre, style1, body1, style2, body2, ...]
        i = 1
        while i + 1 < len(style_chunks):
            style = style_chunks[i].strip()
            body = style_chunks[i + 1]
            # Stop body at horizontal rule or next match
            body = re.split(r"^---$", body, flags=re.M)[0].strip()
            if body:
                out.append({
                    "match_id": match_id,
                    "stats_block": stats_block,
                    "style": style,
                    "cronica": body,
                })
            i += 2
    return out


# ---------- grounding ----------


def parse_stats(block: str) -> dict:
    info = {"local": None, "visitante": None, "resultado": None, "scorers": []}
    for line in block.splitlines():
        if line.startswith("local: "):
            info["local"] = line[len("local: "):].strip()
        elif line.startswith("visitante: "):
            info["visitante"] = line[len("visitante: "):].strip()
        elif line.startswith("resultado: "):
            info["resultado"] = line[len("resultado: "):].strip()
        elif re.match(r"^\s+-\s+\d+'", line):
            m = re.match(r"^\s+-\s+\d+'\s+(.+?)\s+\(", line)
            if m:
                info["scorers"].append(m.group(1))
    return info


def grounding_metrics(samples: list[dict]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0}
    home_ok = away_ok = score_ok = 0
    scorers_total = scorers_matched = 0
    for s in samples:
        info = parse_stats(s["stats_block"])
        cron = s["cronica"].lower()
        if info["local"]:
            toks = [t.lower() for t in info["local"].split() if len(t) > 3]
            if any(t in cron for t in toks):
                home_ok += 1
        if info["visitante"]:
            toks = [t.lower() for t in info["visitante"].split() if len(t) > 3]
            if any(t in cron for t in toks):
                away_ok += 1
        if info["resultado"] and (info["resultado"] in s["cronica"]
                                  or info["resultado"].replace("-", "–") in s["cronica"]):
            score_ok += 1
        for player in info["scorers"]:
            scorers_total += 1
            last = player.split()[-1].lower()
            if last and last in cron:
                scorers_matched += 1
    return {
        "n": n,
        "home_team_pct":  100.0 * home_ok / n,
        "away_team_pct":  100.0 * away_ok / n,
        "score_pct":      100.0 * score_ok / n,
        "scorers_recall_pct": 100.0 * scorers_matched / max(scorers_total, 1),
        "scorers_total": scorers_total,
        "scorers_matched": scorers_matched,
    }


def style_metrics(samples: list[dict]) -> dict:
    by_style = {s: [] for s in STYLE_MARKERS}
    for s in samples:
        if s["style"] in by_style:
            by_style[s["style"]].append(s)
    out = {}
    for style, group in by_style.items():
        if not group:
            out[style] = {"n": 0, "activation_pct": None}
            continue
        markers = STYLE_MARKERS[style]
        hits = 0
        for s in group:
            low = s["cronica"].lower()
            if any(m in low for m in markers):
                hits += 1
        out[style] = {
            "n": len(group),
            "activation_pct": 100.0 * hits / len(group),
            "markers": markers,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("results/samples.md"))
    parser.add_argument("--out", type=Path, default=Path("results/metrics.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    md = args.samples.read_text(encoding="utf-8")
    samples = parse_samples(md)

    if not samples:
        logger.error("No samples parsed from %s. Has eval_heldout finished any?",
                     args.samples)
        return

    logger.info("Parsed %d samples across %d matches",
                len(samples), len({s["match_id"] for s in samples}))

    g = grounding_metrics(samples)
    st = style_metrics(samples)

    print("\n" + "=" * 70)
    print("  GROUNDING METRICS  (held-out matches the model never saw)")
    print("=" * 70)
    print(f"  N samples evaluated: {g['n']}")
    print(f"  Home team referenced:   {g['home_team_pct']:5.1f}%")
    print(f"  Away team referenced:   {g['away_team_pct']:5.1f}%")
    print(f"  Literal score printed:  {g['score_pct']:5.1f}%")
    print(f"  Scorer recall (last):   {g['scorers_recall_pct']:5.1f}%  "
          f"({g['scorers_matched']}/{g['scorers_total']})")
    print()
    print("=" * 70)
    print("  STYLE FIDELITY  (does the requested style show up?)")
    print("=" * 70)
    for style, m in st.items():
        if m["n"] == 0:
            print(f"  {style:30s}  (no samples)")
        else:
            print(f"  {style:30s}  {m['activation_pct']:5.1f}%  (n={m['n']})")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"grounding": g, "style": st}, indent=2))
    print(f"  Metrics JSON: {args.out}")


if __name__ == "__main__":
    main()
