"""Líneas de apertura y cierre de juegos ya jugados (ESPN summary -> pickcenter).

Sirven para el ROI simulado del track record: comparar lo que el modelo habría
dicho antes de cada juego contra la línea de cierre real.

Uso:
    python -m src.data.odds_history              # temporada actual
    python -m src.data.odds_history --season 2025

Es incremental: solo pide los juegos que todavía no están en
data/odds_history.parquet.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import config
from .build import load_games
from .espn import EspnClient, EspnError
from .parse import parse_odds

ODDS_COLUMNS = [
    "provider",
    "spread_home",
    "spread_home_odds",
    "spread_away_odds",
    "total",
    "over_odds",
    "under_odds",
    "ml_home",
    "ml_away",
    "spread_home_open",
    "total_open",
    "ml_home_open",
    "ml_away_open",
]


def load_odds_history(path=config.ODDS_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["game_id", "season", "has_odds", *ODDS_COLUMNS])
    return pd.read_parquet(path)


def fetch_odds_history(
    season: int, *, workers: int = 6, include_fcs: bool = False, offline: bool = False, log=print
) -> pd.DataFrame:
    games = load_games()
    games = games[(games["season"] == season) & games["completed"]]
    if not include_fcs:
        games = games[~games["fcs_game"]]
    existing = load_odds_history()
    todo = games[~games["game_id"].isin(existing["game_id"])]
    log(f"{season}: {len(games)} juegos jugados, {len(todo)} sin líneas guardadas")
    if todo.empty:
        return existing

    client = EspnClient(offline=offline, pause=0.03)
    failed = 0

    def one(game_id: str) -> dict | None:
        nonlocal failed
        try:
            odds = parse_odds(client.pickcenter(game_id))
        except EspnError:
            failed += 1  # no se guarda: se reintenta en la próxima corrida
            return None
        row = {"game_id": game_id, "season": season, "has_odds": odds is not None}
        row.update({col: (odds or {}).get(col) for col in ODDS_COLUMNS})
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(one, todo["game_id"].tolist()) if r is not None]

    new = pd.DataFrame(rows, columns=["game_id", "season", "has_odds", *ODDS_COLUMNS])
    out = new if existing.empty else pd.concat([existing, new], ignore_index=True)
    out = out.drop_duplicates("game_id", keep="last").reset_index(drop=True)
    for col in (
        "spread_home_odds",
        "spread_away_odds",
        "over_odds",
        "under_odds",
        "ml_home",
        "ml_away",
        "ml_home_open",
        "ml_away_open",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("spread_home", "total", "spread_home_open", "total_open"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["has_odds"] = out["has_odds"].astype(bool)
    out["season"] = out["season"].astype("int64")
    out.to_parquet(config.ODDS_HISTORY_PATH, index=False)
    got = int(new["has_odds"].sum()) if len(new) else 0
    log(
        f"{season}: {len(new)} consultados, {got} con líneas, {failed} fallidos · API {client.stats['red']} · caché {client.stats['cache']}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Baja líneas de cierre históricas (pickcenter de ESPN)"
    )
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--include-fcs", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        out = fetch_odds_history(
            args.season, workers=args.workers, include_fcs=args.include_fcs, offline=args.offline
        )
    except EspnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    cur = out[out["season"] == args.season]
    print(f"Líneas guardadas para {args.season}: {int(cur['has_odds'].sum())} de {len(cur)} juegos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
