"""Download every game of a league and build its games parquet.

Usage:
    python -m src.data.build                 # college football: current season + 2021-2025
    python -m src.data.build --league nfl    # the same for the NFL (data/nfl_games.parquet)
    python -m src.data.build --no-history    # current season only
    python -m src.data.build --refresh       # ignore the cache for the current week

College football keeps FBS teams and pools everyone else as "FCS". The NFL has
no lower division: every team with a regular-season game belongs to it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from .. import config
from .espn import EspnClient, EspnError
from .parse import parse_calendar, parse_event, parse_event_teams, parse_standings_fbs

GAME_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "week",
    "week_index",
    "date",
    "status",
    "completed",
    "neutral_site",
    "conference_game",
    "fcs_game",
    "home_id",
    "home_team",
    "home_abbr",
    "home_points",
    "home_rank",
    "home_fbs",
    "away_id",
    "away_team",
    "away_abbr",
    "away_points",
    "away_rank",
    "away_fbs",
]


def fetch_season(
    client: EspnClient, season: int, *, refresh: bool = False, log=print
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Download a full season week by week. Returns (games, teams, calendar)."""
    league = client.league
    first = client.scoreboard_week(season, 1, refresh=refresh)
    calendar = parse_calendar(first)
    if not calendar["regular"]:
        raise EspnError(f"the API returned no calendar for {season}")
    n_regular = max(w["week"] for w in calendar["regular"])

    rows: list[dict] = []
    teams: dict[str, dict] = {}
    skipped = 0

    def ingest(payload: dict, week_index: int | None) -> None:
        nonlocal skipped
        events = payload.get("events") or []
        if len(events) >= league.scoreboard_limit:
            log(
                f"  warning: the response hit the limit of {league.scoreboard_limit} games; "
                "some may be missing"
            )
        for event in events:
            row = parse_event(event, league.nicknames)
            if row is None:
                skipped += 1
                continue
            row["week_index"] = week_index if week_index is not None else row["week"]
            rows.append(row)
            for team in parse_event_teams(event, league.nicknames):
                teams[team["team_id"]] = team

    ingest(first, None)
    for week in range(2, n_regular + 1):
        ingest(client.scoreboard_week(season, week, refresh=refresh), None)
    for entry in postseason_rounds(calendar, league):
        try:
            ingest(
                client.scoreboard_week(season, entry["week"], seasontype=3, refresh=refresh),
                entry["week_index"],
            )
        except EspnError as exc:
            # A future postseason may not exist yet; that is not a pipeline error.
            log(f"  {season} postseason ({entry['label']}) not available: {exc}")

    games = pd.DataFrame(rows)
    if games.empty:
        raise EspnError(f"the API returned no games for {season}")
    # A rescheduled game can show up in two weeks: keep the played version.
    games = (
        games.sort_values(["completed", "date"])
        .drop_duplicates("game_id", keep="last")
        .sort_values(["date", "game_id"])
        .reset_index(drop=True)
    )

    if league.membership == "standings":
        fbs = parse_standings_fbs(client.standings(season, refresh=refresh))
        if len(fbs) < 100:
            raise EspnError(
                f"{season} standings returned only {len(fbs)} FBS teams; cannot classify FBS/FCS"
            )
    else:
        # No lower division: the league is whoever plays in the regular season. This also
        # leaves out all-star games, whose "teams" are conferences.
        regular = games[games["season_type"] == 2]
        fbs = dict.fromkeys(set(regular["home_id"]) | set(regular["away_id"]), "")
        if not fbs:
            raise EspnError(f"{season}: no regular-season games to tell the teams from")
    games["home_fbs"] = games["home_id"].isin(fbs)
    games["away_fbs"] = games["away_id"].isin(fbs)
    games["fcs_game"] = ~(games["home_fbs"] & games["away_fbs"])
    stray = ~(games["home_fbs"] | games["away_fbs"])
    if stray.any():
        log(f"  {int(stray.sum())} game(s) between teams outside the league dropped")
        games = games[~stray].reset_index(drop=True)

    for col in ("home_points", "away_points", "home_rank", "away_rank"):
        games[col] = games[col].astype("Int64")
    for col in ("season", "season_type", "week", "week_index"):
        games[col] = games[col].astype("int64")
    games["date"] = pd.to_datetime(games["date"], utc=True)

    team_df = pd.DataFrame(teams.values())
    team_df = team_df[team_df["team_id"].isin(set(games["home_id"]) | set(games["away_id"]))].copy()
    team_df["fbs"] = team_df["team_id"].isin(fbs)
    team_df["conference"] = team_df["team_id"].map(fbs).fillna("")
    team_df["season"] = season
    if skipped:
        log(f"  {skipped} event(s) with undefined teams skipped")
    return games[GAME_COLUMNS], team_df, calendar


def postseason_rounds(calendar: dict, league: config.League) -> list[dict]:
    """The postseason entries of a calendar, each with the `week_index` its games get.

    College football: every bowl goes in one bucket after the last regular week.
    NFL: each round is its own week (Wild Card = last week + 1 ... Super Bowl =
    last week + 4), and the Pro Bowl, which is not a real game, is left out."""
    regular = [w["week"] for w in calendar["regular"]]
    n_regular = max(regular) if regular else 0
    rounds = [e for e in calendar["postseason"] if "pro bowl" not in e["label"].lower()]
    out = []
    for position, entry in enumerate(sorted(rounds, key=lambda e: e["week"]), start=1):
        offset = position if league.playoff_rounds else 1
        out.append({**entry, "week_index": n_regular + offset})
    return out


def season_summary(games: pd.DataFrame) -> pd.DataFrame:
    g = games
    out = g.groupby("season").agg(
        games=("game_id", "size"),
        played=("completed", "sum"),
        fbs_vs_fbs=("fcs_game", lambda s: int((~s).sum())),
        vs_fcs=("fcs_game", "sum"),
        neutral=("neutral_site", "sum"),
        weeks=("week_index", "nunique"),
    )
    return out.astype(int)


def load_games(path=None, league: config.League = config.CFB) -> pd.DataFrame:
    games = pd.read_parquet(path or league.games_path)
    games["date"] = pd.to_datetime(games["date"], utc=True)
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_teams(path=None, league: config.League = config.CFB) -> pd.DataFrame:
    return pd.read_parquet(path or league.teams_path)


def build(
    season: int,
    history: list[int],
    *,
    league: config.League = config.CFB,
    refresh: bool = False,
    refresh_history: bool = False,
    offline: bool = False,
    log=print,
) -> pd.DataFrame:
    client = EspnClient(offline=offline, league=league)
    games_path, teams_path = league.games_path, league.teams_path
    existing = load_games(games_path) if games_path.exists() else None
    frames: list[pd.DataFrame] = []
    team_frames: list[pd.DataFrame] = []

    for year in sorted(set(history) | {season}):
        is_current = year == season
        if not is_current and not refresh_history and existing is not None:
            prev = existing[existing["season"] == year]
            # A past season no longer changes: what is in the parquet is reused.
            if len(prev) > league.min_season_games and prev["completed"].mean() > 0.97:
                frames.append(prev[GAME_COLUMNS])
                log(f"{year}: {len(prev)} games (reused from {games_path.name})")
                continue
        games, teams, _ = fetch_season(client, year, refresh=refresh and is_current, log=log)
        frames.append(games)
        team_frames.append(teams)
        log(f"{year}: {len(games)} games downloaded ({int(games['completed'].sum())} played)")

    all_games = (
        pd.concat(frames, ignore_index=True).sort_values(["date", "game_id"]).reset_index(drop=True)
    )
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_games.to_parquet(games_path, index=False)

    if team_frames:
        teams = pd.concat(team_frames, ignore_index=True).sort_values("season")
        if teams_path.exists():
            teams = pd.concat([load_teams(teams_path), teams], ignore_index=True)
            teams = teams.sort_values("season")
        # One record per team: the one from the most recent season.
        teams = (
            teams.drop_duplicates("team_id", keep="last").sort_values("team").reset_index(drop=True)
        )
        teams.to_parquet(teams_path, index=False)

    log(f"\nAPI requests: {client.stats['network']} · from cache: {client.stats['cache']}")
    return all_games


def _parse_history(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a league's games from ESPN and build its games parquet"
    )
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="cfb")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument(
        "--history",
        default=None,
        help="previous seasons, e.g. 2021-2025 or 2024,2025 (default: the league's history)",
    )
    parser.add_argument(
        "--no-history", action="store_true", help="download only the current season"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore the cache for the current season"
    )
    parser.add_argument(
        "--refresh-history", action="store_true", help="download past seasons again"
    )
    parser.add_argument("--offline", action="store_true", help="use the cache only (no network)")
    args = parser.parse_args(argv)

    league = config.LEAGUES[args.league]
    seasons = _parse_history(args.history) if args.history else list(league.history_seasons)
    history = [] if args.no_history else [y for y in seasons if y < args.season]
    started = datetime.now(timezone.utc)
    try:
        games = build(
            args.season,
            history,
            league=league,
            refresh=args.refresh,
            refresh_history=args.refresh_history,
            offline=args.offline,
        )
    except EspnError as exc:
        print(f"ERROR: the ESPN API failed, and data is never made up: {exc}", file=sys.stderr)
        return 1

    print("\nSummary by season:")
    print(season_summary(games).to_string())
    cur = games[games["season"] == args.season]
    done = cur[cur["completed"]]
    split = ""
    if league.membership == "standings":
        split = f", {int((~done['fcs_game']).sum())} FBS vs FBS and {int(done['fcs_game'].sum())} vs FCS"
    print(
        f"\n{league.name} {args.season}: {len(cur)} games in total, {len(done)} played "
        f"(through week {int(done['week_index'].max()) if len(done) else 0}){split}."
    )
    print(
        f"Saved to {league.games_path.relative_to(config.ROOT)} · "
        f"{(datetime.now(timezone.utc) - started).seconds}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
