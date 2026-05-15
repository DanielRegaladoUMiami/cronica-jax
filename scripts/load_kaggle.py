"""Download davidcariboo/player-scores from Kaggle and filter to 'rich' matches.

A 'rich' match is one with:
  - At least 1 goal scored
  - Non-empty game_events
  - Identified competition

Outputs:
  data/kaggle/raw/*.csv          - raw download
  data/kaggle/matches_rich.parquet - one row per match + JSON events column

Usage:
    python -m scripts.load_kaggle --out data/kaggle --max-matches 50000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import kagglehub
import pandas as pd

logger = logging.getLogger(__name__)


DATASET = "davidcariboo/player-scores"


def download_dataset(out_dir: Path) -> Path:
    """Download via kagglehub. Returns path to the dataset directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s via kagglehub ...", DATASET)
    src = kagglehub.dataset_download(DATASET)
    logger.info("Downloaded to: %s", src)
    return Path(src)


def filter_rich_matches(src_dir: Path, max_matches: int | None = None) -> pd.DataFrame:
    """Join games + game_events + competitions and keep matches with goals."""
    games = pd.read_csv(src_dir / "games.csv")
    events = pd.read_csv(src_dir / "game_events.csv")
    comps = pd.read_csv(src_dir / "competitions.csv")

    logger.info("Loaded: %d games, %d events, %d competitions",
                len(games), len(events), len(comps))

    # Only keep events with type=Goals
    goals = events[events["type"] == "Goals"].copy()
    logger.info("Total goal events: %d", len(goals))

    # match_id with at least one goal
    games_with_goals = goals["game_id"].unique()
    games = games[games["game_id"].isin(games_with_goals)].copy()
    logger.info("Games with at least one goal: %d", len(games))

    # Join competition name (games.csv already has home_club_name/away_club_name)
    games = games.merge(
        comps[["competition_id", "name", "country_name"]],
        on="competition_id", how="left",
    ).rename(columns={"name": "competition_name"})

    # Aggregate goal events per game
    def event_to_dict(row):
        return {
            "minute": int(row["minute"]) if pd.notna(row.get("minute")) else None,
            "club_id": int(row["club_id"]) if pd.notna(row.get("club_id")) else None,
            "player_in_id": int(row["player_id"]) if pd.notna(row.get("player_id")) else None,
            "description": row.get("description"),
        }

    by_game = (
        goals.groupby("game_id")
        .apply(lambda d: json.dumps([event_to_dict(r) for r in d.to_dict("records")]))
        .rename("goals_json")
        .reset_index()
    )
    games = games.merge(by_game, on="game_id", how="left")

    # Keep informative columns
    cols = [
        "game_id", "date", "season",
        "competition_id", "competition_name", "country_name",
        "home_club_id", "home_club_name",
        "away_club_id", "away_club_name",
        "home_club_goals", "away_club_goals",
        "stadium", "attendance", "referee",
        "goals_json",
    ]
    cols = [c for c in cols if c in games.columns]
    out = games[cols].dropna(subset=["home_club_name", "away_club_name"])

    if max_matches is not None and len(out) > max_matches:
        out = out.sample(max_matches, random_state=42).reset_index(drop=True)
        logger.info("Sampled down to %d matches", len(out))

    logger.info("Final 'rich' matches: %d", len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/kaggle"))
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Subsample to this many matches.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src = download_dataset(args.out / "raw")
    rich = filter_rich_matches(src, max_matches=args.max_matches)

    out_path = args.out / "matches_rich.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rich.to_parquet(out_path, index=False, compression="snappy")
    logger.info("Wrote %d rows to %s (%.1f MB)",
                len(rich), out_path, out_path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
