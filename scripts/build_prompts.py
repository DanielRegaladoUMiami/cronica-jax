"""Build structured <STATS> prompts from Kaggle matches.

Input:  data/kaggle/matches_rich.parquet
Output: data/prompts/prompts.jsonl
  Each row: {match_id, stats_block, lang_hint}

The stats_block is the structured prompt the OpenAI model will see, and it's
also the input the JAX model will eventually learn to condition on.

Usage:
    python -m scripts.build_prompts --in data/kaggle/matches_rich.parquet \
        --out data/prompts/prompts.jsonl --max 50000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Transfermarkt competition slugs that the LLM and a Spanish-speaking audience
# both know well. Each entry maps slug -> human-readable Spanish league name.
# Anything NOT in this map is filtered out (noisy lower-tier leagues, leagues
# the LLM has limited Spanish-language coverage of, etc).
COMP_ES = {
    # Spain
    "laliga": "La Liga",
    "laliga2": "La Liga 2 (Segunda División)",
    "copa-del-rey": "Copa del Rey",
    "supercopa": "Supercopa de España",
    # Argentina
    "liga-profesional-de-futbol": "Liga Profesional Argentina",
    "torneo-apertura": "Torneo Apertura (Argentina)",
    "torneo-clausura": "Torneo Clausura (Argentina)",
    "torneo-final": "Torneo Final (Argentina)",
    "copa-argentina": "Copa Argentina",
    # Mexico
    "liga-mx-apertura": "Liga MX Apertura",
    "liga-mx-clausura": "Liga MX Clausura",
    "copa-mx": "Copa MX",
    # Brazil
    "campeonato-brasileiro-serie-a": "Brasileirao Serie A",
    "copa-do-brasil": "Copa de Brasil",
    # Colombia, Chile, Uruguay, Peru, Ecuador
    "liga-betplay-dimayor": "Liga BetPlay (Colombia)",
    "primera-division-de-chile": "Primera División de Chile",
    "primera-division-uruguay": "Primera División de Uruguay",
    "liga-1-peru": "Liga 1 (Perú)",
    "serie-a-de-ecuador": "Serie A (Ecuador)",
    # CONMEBOL international
    "copa-libertadores": "Copa Libertadores",
    "copa-sudamericana": "Copa Sudamericana",
    "copa-america": "Copa América",
    # Big European leagues — LLM and audience both know them
    "premier-league": "Premier League",
    "bundesliga": "Bundesliga",
    "serie-a": "Serie A (Italia)",
    "ligue-1": "Ligue 1",
    "liga-portugal-bwin": "Primeira Liga (Portugal)",
    "eredivisie": "Eredivisie",
    "uefa-champions-league": "Champions League",
    "uefa-europa-league": "Europa League",
    "uefa-conference-league": "Conference League",
    "fa-cup": "FA Cup",
    "dfb-pokal": "DFB-Pokal",
    "coppa-italia": "Copa Italia",
    "supercoppa-italiana": "Supercopa de Italia",
    "trophee-des-champions": "Trofeo de Campeones (Francia)",
    # Major worldwide
    "fifa-world-cup": "Copa del Mundo",
    "uefa-euro": "Eurocopa",
    "uefa-nations-league": "UEFA Nations League",
}


def _es_comp(name: str | None) -> str | None:
    """Returns Spanish league name, or None if this competition is filtered out."""
    if not name or pd.isna(name):
        return None
    return COMP_ES.get(str(name).strip().lower())


def _format_date(s: str | None) -> str:
    if not s or pd.isna(s):
        return ""
    # Already ISO if Kaggle dump is well-behaved
    return str(s)[:10]


def _parse_goals(blob: str | None, home_id: int, away_id: int,
                 home_name: str, away_name: str,
                 player_lookup: dict[int, str]) -> list[str]:
    """Return chronological list of 'MIN' Player (Team)' strings."""
    if not blob or pd.isna(blob):
        return []
    try:
        events = json.loads(blob)
    except Exception:
        return []
    parsed = []
    for e in events:
        minute = e.get("minute")
        cid = e.get("club_id")
        pid = e.get("player_in_id")
        if minute is None or cid is None:
            continue
        team = home_name if cid == home_id else away_name if cid == away_id else "?"
        player = player_lookup.get(pid, "desconocido")
        parsed.append((int(minute), player, team))
    parsed.sort(key=lambda x: x[0])
    return [f"  - {m}' {p} ({t})" for m, p, t in parsed]


def build_stats_block(row, player_lookup: dict[int, str]) -> str | None:
    """Returns the <STATS> block, or None if the match should be filtered out."""
    comp = _es_comp(row.get("competition_name"))
    if comp is None:
        return None
    date = _format_date(row.get("date"))
    local = row.get("home_club_name") or "?"
    visit = row.get("away_club_name") or "?"
    hg = int(row["home_club_goals"]) if pd.notna(row.get("home_club_goals")) else 0
    ag = int(row["away_club_goals"]) if pd.notna(row.get("away_club_goals")) else 0
    score = f"{hg}-{ag}"
    stadium = row.get("stadium") or ""
    attendance = row.get("attendance")
    referee = row.get("referee") or ""

    goals = _parse_goals(
        row.get("goals_json"),
        row.get("home_club_id"), row.get("away_club_id"),
        local, visit,
        player_lookup,
    )

    lines = [
        "<STATS>",
        f"liga: {comp}",
        f"fecha: {date}" if date else None,
        f"local: {local}",
        f"visitante: {visit}",
        f"resultado: {score}",
        "goles:" if goals else "goles: (sin detalle)",
        *goals,
        f"estadio: {stadium}" if stadium else None,
        (f"asistencia: {int(attendance):,}".replace(",", ".")
         if attendance and pd.notna(attendance) else None),
        f"árbitro: {referee}" if referee else None,
        "</STATS>",
    ]
    return "\n".join(l for l in lines if l is not None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path,
                        default=Path("data/kaggle/matches_rich.parquet"))
    parser.add_argument("--players", type=Path,
                        default=Path("data/kaggle/raw"),
                        help="Folder containing players.csv from the Kaggle dump.")
    parser.add_argument("--out", type=Path, default=Path("data/prompts/prompts.jsonl"))
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.inp)
    logger.info("Loaded %d matches", len(df))

    # Filter to leagues we map to Spanish names.
    pre = len(df)
    df = df[df["competition_name"].str.lower().isin(COMP_ES.keys())].copy()
    logger.info("Filtered to known leagues: %d -> %d", pre, len(df))
    if len(df) == 0:
        logger.error("No matches survived league filter; check COMP_ES.")
        return

    # Build player_id -> name map. Look in our data dir first, then fall back
    # to the kagglehub cache where the original CSV actually lives.
    import os
    cache_root = Path(os.path.expanduser("~/.cache/kagglehub/datasets/davidcariboo/player-scores"))
    cache_candidates = sorted(cache_root.glob("versions/*/players.csv"), reverse=True)
    players_csv = None
    for candidate in [
        args.players / "players.csv",
        args.inp.parent / "raw" / "players.csv",
        *cache_candidates,
    ]:
        if candidate.exists():
            players_csv = candidate
            break
    player_lookup: dict[int, str] = {}
    if players_csv:
        pdf = pd.read_csv(players_csv)
        for r in pdf.itertuples(index=False):
            try:
                player_lookup[int(r.player_id)] = r.name
            except Exception:
                continue
        logger.info("Loaded %d player names from %s", len(player_lookup), players_csv)
    else:
        logger.warning("players.csv not found; player names will be 'desconocido'.")

    if args.max:
        df = df.sample(min(args.max, len(df)), random_state=42).reset_index(drop=True)
        logger.info("Sampled to %d", len(df))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    by_liga: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            block = build_stats_block(row, player_lookup)
            if block is None:
                skipped += 1
                continue
            liga = COMP_ES[str(row["competition_name"]).strip().lower()]
            by_liga[liga] = by_liga.get(liga, 0) + 1
            rec = {
                "match_id": int(row["game_id"]),
                "stats_block": block,
                "competition_name": row.get("competition_name") or "",
                "liga": liga,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d prompts to %s (skipped %d)", n, args.out, skipped)
    logger.info("Coverage by liga (top 15):")
    for liga, c in sorted(by_liga.items(), key=lambda x: -x[1])[:15]:
        logger.info("  %4d  %s", c, liga)


if __name__ == "__main__":
    main()
