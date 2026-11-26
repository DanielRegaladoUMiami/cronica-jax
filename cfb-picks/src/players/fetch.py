"""Per-player box scores, rosters and injuries from ESPN's public API.

Three things are downloaded:

- The box score of every finished game (ESPN summary -> boxscore): one row per
  player and game with his passing, rushing and receiving numbers, and one row
  per team and game with the team totals.
- Each team's roster: position, and whether the player is out, on injured
  reserve or questionable this week.
- Each team's depth chart: who starts at quarterback and the order behind him.

Usage:
    python -m src.players.fetch                      # the current season
    python -m src.players.fetch --seasons 2025 2026

It is incremental: a game already in the parquet is not requested again. If the
API fails for a game it is left out (and retried on the next run): no number
is ever made up.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import config
from ..data.build import load_games, load_teams
from ..data.espn import EspnClient, EspnError

LEAGUE = config.NFL
PLAYER_GAMES_PATH = config.DATA_DIR / "nfl_player_games.parquet"
TEAM_GAMES_PATH = config.DATA_DIR / "nfl_team_games.parquet"

# Numbers kept for every player and game.
PLAYER_STATS = [
    "pass_att",
    "pass_cmp",
    "pass_yds",
    "pass_td",
    "pass_int",
    "sacks",
    "rush_att",
    "rush_yds",
    "rush_td",
    "targets",
    "rec",
    "rec_yds",
    "rec_td",
]
TEAM_STATS = [*PLAYER_STATS, "plays", "points", "points_allowed"]

# ESPN's box score keys -> our column names, per category.
_CATEGORIES = {
    "passing": {
        "passingYards": "pass_yds",
        "passingTouchdowns": "pass_td",
        "interceptions": "pass_int",
    },
    "rushing": {
        "rushingAttempts": "rush_att",
        "rushingYards": "rush_yds",
        "rushingTouchdowns": "rush_td",
    },
    "receiving": {
        "receptions": "rec",
        "receivingYards": "rec_yds",
        "receivingTouchdowns": "rec_td",
        "receivingTargets": "targets",
    },
}

# Positions that carry, catch or throw the ball.
SKILL_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
# Injury statuses that mean "will not play".
OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful", "Suspension", "Suspended"}


def _num(value) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _pair(value, sep: str) -> tuple[float, float]:
    """'12/27' -> (12, 27); '2-15' -> (2, 15). Anything else -> (0, 0)."""
    parts = str(value).split(sep)
    if len(parts) != 2:
        return 0.0, 0.0
    return _num(parts[0]), _num(parts[1])


def trim_boxscore(payload: dict) -> dict:
    """Keep only what is used from a game summary, so the cache stays small."""
    box = payload.get("boxscore") or {}
    players = []
    for side in box.get("players") or []:
        cats = []
        for cat in side.get("statistics") or []:
            if cat.get("name") not in _CATEGORIES:
                continue
            cats.append(
                {
                    "name": cat["name"],
                    "keys": cat.get("keys") or [],
                    "athletes": [
                        {
                            "id": str((a.get("athlete") or {}).get("id")),
                            "name": (a.get("athlete") or {}).get("displayName"),
                            "stats": a.get("stats") or [],
                        }
                        for a in cat.get("athletes") or []
                    ],
                }
            )
        players.append({"team_id": str((side.get("team") or {}).get("id")), "statistics": cats})
    teams = [
        {
            "team_id": str((side.get("team") or {}).get("id")),
            "statistics": {
                s.get("name"): s.get("displayValue") for s in side.get("statistics") or []
            },
        }
        for side in box.get("teams") or []
    ]
    return {"players": players, "teams": teams}


def parse_boxscore(box: dict, game: dict) -> tuple[list[dict], list[dict]]:
    """One row per player and one per team for a finished game.

    `game` is the row of the games table (ids, week, date, points). A box score
    with no player numbers returns two empty lists: the game is simply not used.
    """
    sides = {
        str(game["home_id"]): (True, str(game["away_id"])),
        str(game["away_id"]): (False, str(game["home_id"])),
    }
    points = {
        str(game["home_id"]): (game["home_points"], game["away_points"]),
        str(game["away_id"]): (game["away_points"], game["home_points"]),
    }
    base = {
        "game_id": str(game["game_id"]),
        "season": int(game["season"]),
        "week": int(game["week"]),
        "week_index": int(game["week_index"]),
        "date": game["date"],
    }
    player_rows: list[dict] = []
    team_rows: list[dict] = []
    plays = {t["team_id"]: t.get("statistics") or {} for t in box.get("teams") or []}
    for side in box.get("players") or []:
        team_id = side.get("team_id")
        if team_id not in sides:
            continue
        home, opp = sides[team_id]
        by_player: dict[str, dict] = {}
        for cat in side.get("statistics") or []:
            keys = cat.get("keys") or []
            mapping = _CATEGORIES[cat["name"]]
            for ath in cat.get("athletes") or []:
                row = by_player.setdefault(
                    ath["id"],
                    {
                        "athlete_id": ath["id"],
                        "name": ath.get("name"),
                        **dict.fromkeys(PLAYER_STATS, 0.0),
                    },
                )
                stats = dict(zip(keys, ath.get("stats") or []))
                for key, col in mapping.items():
                    row[col] = _num(stats.get(key))
                if cat["name"] == "passing":
                    row["pass_cmp"], row["pass_att"] = _pair(
                        stats.get("completions/passingAttempts"), "/"
                    )
                    row["sacks"] = _pair(stats.get("sacks-sackYardsLost"), "-")[0]
        if not by_player:
            continue
        team = {**base, "team_id": team_id, "opp_id": opp, "home": home}
        for row in by_player.values():
            player_rows.append({**team, **row})
        totals = {col: sum(r[col] for r in by_player.values()) for col in PLAYER_STATS}
        pts, pts_allowed = points[team_id]
        team_rows.append(
            {
                **team,
                **totals,
                "plays": _num((plays.get(team_id) or {}).get("totalOffensivePlays")),
                "points": float(pts) if pd.notna(pts) else None,
                "points_allowed": float(pts_allowed) if pd.notna(pts_allowed) else None,
            }
        )
    if len(team_rows) != 2:  # a box score with only one side is not trusted
        return [], []
    return player_rows, team_rows


def boxscore(client: EspnClient, game_id: str) -> dict:
    return client.get(
        f"{LEAGUE.cache_prefix}boxscore/{game_id}",
        f"{LEAGUE.site_url}/summary",
        {"event": game_id},
        max_age=None,
        transform=trim_boxscore,
    )


def load_player_games(path=None) -> pd.DataFrame:
    path = path or PLAYER_GAMES_PATH
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def load_team_games(path=None) -> pd.DataFrame:
    path = path or TEAM_GAMES_PATH
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def fetch_player_games(
    seasons: list[int], *, workers: int = 6, offline: bool = False, log=print
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the box scores that are missing and save both tables."""
    games = load_games(league=LEAGUE)
    games = games[games["season"].isin(seasons) & games["completed"]]
    players, teams = load_player_games(), load_team_games()
    have = set(teams["game_id"]) if len(teams) else set()
    todo = games[~games["game_id"].isin(have)]
    log(f"Box scores: {len(games)} games played in {seasons}, {len(todo)} to download")
    if todo.empty:
        return players, teams

    client = EspnClient(offline=offline, pause=0.03, league=LEAGUE)
    failed = 0

    def one(game: dict) -> tuple[list[dict], list[dict]]:
        nonlocal failed
        try:
            return parse_boxscore(boxscore(client, game["game_id"]), game)
        except EspnError:
            failed += 1  # not saved: it is retried on the next run
            return [], []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, todo.to_dict("records")))
    new_players = pd.DataFrame([r for p, _ in results for r in p])
    new_teams = pd.DataFrame([r for _, t in results for r in t])
    empty = sum(1 for p, _ in results if not p) - failed
    if len(new_teams):
        players = pd.concat([players, new_players], ignore_index=True)
        teams = pd.concat([teams, new_teams], ignore_index=True)
        players = players.drop_duplicates(["game_id", "athlete_id"], keep="last")
        teams = teams.drop_duplicates(["game_id", "team_id"], keep="last")
        players = players.sort_values(["date", "game_id", "team_id", "athlete_id"])
        teams = teams.sort_values(["date", "game_id", "team_id"])
        players.reset_index(drop=True).to_parquet(PLAYER_GAMES_PATH, index=False)
        teams.reset_index(drop=True).to_parquet(TEAM_GAMES_PATH, index=False)
    log(
        f"Box scores: {len(new_teams) // 2} games added, {empty} without player numbers,"
        f" {failed} failed · API {client.stats['network']} · cache {client.stats['cache']}"
    )
    return players, teams


# ------------------------------------------------------------ rosters and depth
def parse_roster(payload: dict, team_id: str) -> list[dict]:
    """Every player on a team: position and this week's injury status."""
    rows = []
    for group in payload.get("athletes") or []:
        for ath in group.get("items") or []:
            injuries = ath.get("injuries") or []
            status = injuries[0].get("status") if injuries else None
            if group.get("position") == "suspended":
                status = "Suspended"
            elif group.get("position") == "injuredReserveOrOut" and status not in OUT_STATUSES:
                status = "Out"
            rows.append(
                {
                    "athlete_id": str(ath.get("id")),
                    "name": ath.get("displayName"),
                    "team_id": str(team_id),
                    "position": (ath.get("position") or {}).get("abbreviation"),
                    "jersey": ath.get("jersey"),
                    "group": group.get("position"),
                    "injury_status": status,
                    "injury_date": injuries[0].get("date") if injuries else None,
                    "practice_squad": group.get("position") == "practiceSquad",
                }
            )
    return rows


def parse_depth(payload: dict, team_id: str) -> list[dict]:
    """The offensive depth chart: position slot, order and player."""
    rows = []
    for chart in payload.get("depthchart") or []:
        positions = chart.get("positions") or {}
        if "qb" not in positions:
            continue
        for slot, block in positions.items():
            for order, ath in enumerate(block.get("athletes") or [], start=1):
                rows.append(
                    {
                        "team_id": str(team_id),
                        "slot": slot,
                        "order": order,
                        "athlete_id": str(ath.get("id")),
                        "name": ath.get("displayName"),
                    }
                )
    return rows


def fetch_rosters(client: EspnClient | None = None, *, max_age: float = 3 * 3600, log=print):
    """Rosters and depth charts of every team (refreshed every few hours)."""
    client = client or EspnClient(league=LEAGUE, pause=0.03)
    teams = load_teams(league=LEAGUE)
    season = int(teams["season"].max())
    ids = sorted(teams[teams["season"] == season]["team_id"].astype(str).unique())
    roster_rows: list[dict] = []
    depth_rows: list[dict] = []
    failed: list[str] = []

    def one(team_id: str):
        try:
            roster = client.get(
                f"{LEAGUE.cache_prefix}roster/{team_id}",
                f"{LEAGUE.site_url}/teams/{team_id}/roster",
                max_age=max_age,
                transform=lambda p: {"athletes": _trim_roster(p)},
            )
            depth = client.get(
                f"{LEAGUE.cache_prefix}depth/{team_id}",
                f"{LEAGUE.site_url}/teams/{team_id}/depthcharts",
                max_age=max_age,
                transform=_trim_depth,
            )
        except EspnError:
            failed.append(team_id)
            return [], []
        return parse_roster(roster, team_id), parse_depth(depth, team_id)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for roster, depth in pool.map(one, ids):
            roster_rows.extend(roster)
            depth_rows.extend(depth)
    log(
        f"Rosters: {len(ids) - len(failed)} of {len(ids)} teams"
        + (f" ({len(failed)} failed: no injury data for them)" if failed else "")
    )
    return pd.DataFrame(roster_rows), pd.DataFrame(depth_rows), failed


def _trim_roster(payload: dict) -> list[dict]:
    keep = ("id", "displayName", "jersey", "position", "injuries")
    out = []
    for group in payload.get("athletes") or []:
        items = []
        for ath in group.get("items") or []:
            item = {k: ath.get(k) for k in keep}
            item["position"] = {"abbreviation": (ath.get("position") or {}).get("abbreviation")}
            item["injuries"] = [
                {"status": i.get("status"), "date": i.get("date")}
                for i in ath.get("injuries") or []
            ]
            items.append(item)
        out.append({"position": group.get("position"), "items": items})
    return out


def _trim_depth(payload: dict) -> dict:
    charts = []
    for chart in payload.get("depthchart") or []:
        positions = {
            slot: {
                "athletes": [
                    {"id": a.get("id"), "displayName": a.get("displayName")}
                    for a in block.get("athletes") or []
                ]
            }
            for slot, block in (chart.get("positions") or {}).items()
        }
        charts.append({"name": chart.get("name"), "positions": positions})
    return {"depthchart": charts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download per-player NFL box scores from ESPN")
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    seasons = args.seasons or [config.current_season()]
    try:
        players, teams = fetch_player_games(seasons, workers=args.workers, offline=args.offline)
    except EspnError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Saved: {len(players)} player rows and {len(teams)} team rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
