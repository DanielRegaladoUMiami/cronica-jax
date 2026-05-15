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


# Map common Transfermarkt competition names (English) to Spanish.
# Anything not in the map is kept verbatim — Spanish-speakers know "Premier League".
COMP_ES = {
    "Premier League": "Premier League",
    "LaLiga": "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie A",
    "Ligue 1": "Ligue 1",
    "Liga Portugal": "Liga Portugal",
    "Eredivisie": "Eredivisie",
    "Belgian Pro League": "Pro League de Bélgica",
    "Scottish Premiership": "Scottish Premiership",
    "Liga MX Apertura": "Liga MX Apertura",
    "Liga MX Clausura": "Liga MX Clausura",
    "Liga Profesional de Fútbol": "Liga Profesional Argentina",
    "Primera División": "Primera División",
    "UEFA Champions League": "Champions League",
    "UEFA Europa League": "Europa League",
    "UEFA Europa Conference League": "Conference League",
    "Copa Libertadores": "Copa Libertadores",
    "Copa Sudamericana": "Copa Sudamericana",
    "FA Cup": "FA Cup",
    "Copa del Rey": "Copa del Rey",
    "DFB-Pokal": "DFB-Pokal",
    "Coppa Italia": "Copa Italia",
}


def _es_comp(name: str | None) -> str:
    if not name or pd.isna(name):
        return "competición no especificada"
    return COMP_ES.get(name, name)


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


def build_stats_block(row, player_lookup: dict[int, str]) -> str:
    comp = _es_comp(row.get("competition_name"))
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

    # Build player_id -> name map (kagglehub stores raw csvs in cache; allow override)
    players_csv = None
    for candidate in [args.players / "players.csv",
                      args.inp.parent / "raw" / "players.csv"]:
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
    with args.out.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            block = build_stats_block(row, player_lookup)
            rec = {
                "match_id": int(row["game_id"]),
                "stats_block": block,
                "competition_name": row.get("competition_name") or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d prompts to %s", n, args.out)


if __name__ == "__main__":
    main()
