"""Download soccer matches and historical odds from ESPN.

Usage:
    python -m src.soccer.data              # matches for every league, current season + history
    python -m src.soccer.data --odds       # also closing odds for matches already played

Matches come from the monthly scoreboard (11 requests per league and season).
ESPN only keeps odds on the scoreboard for matches still to be played, so the
closing odds of finished matches come from each match's summary, one request
per match, cached forever.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..data.espn import EspnClient, EspnError, _strip_links
from ..data.parse import dig, parse_american, parse_line
from . import config

MATCH_COLUMNS = [
    "match_id",
    "league",
    "season",
    "date",
    "week_start",
    "status",
    "completed",
    "home_id",
    "home_team",
    "home_short",
    "home_abbr",
    "home_goals",
    "away_id",
    "away_team",
    "away_short",
    "away_abbr",
    "away_goals",
]
ODDS_COLUMNS = [
    "provider",
    "ml_home",
    "ml_draw",
    "ml_away",
    "total",
    "over_odds",
    "under_odds",
    "spread_home",
    "spread_home_odds",
    "spread_away_odds",
    "ml_home_open",
    "ml_draw_open",
    "ml_away_open",
    "total_open",
]
# Which book to trust first when a match has several. "Live" feeds are in-play prices.
PROVIDER_PREFERENCE = (
    "draftkings",
    "espnbet",
    "bet365",
    "caesars",
    "unibet",
    "betfair",
    "sugarhouse",
    "pointsbet",
    "westgate",
    "mgm",
    "holland",
    "titanbets",
)


# ------------------------------------------------------------------- parsing
def parse_status(event: dict) -> str:
    """final | scheduled | in_progress | postponed | canceled | unknown."""
    st = dig(event, "status", "type", default={}) or {}
    name = str(st.get("name") or "")
    if "CANCEL" in name or "ABANDON" in name:
        return "canceled"
    if "POSTPONED" in name or "SUSPENDED" in name:
        return "postponed"
    state = st.get("state")
    if st.get("completed") and state == "post":
        return "final"
    if state == "in":
        return "in_progress"
    if state == "pre":
        return "scheduled"
    return "unknown"


def week_start(date: datetime) -> datetime:
    """The Tuesday that opens a match's week. A weekend round (Fri-Mon) stays in one week."""
    day = date.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return day - timedelta(days=(day.weekday() - 1) % 7)


def _goals(competitor: dict) -> int | None:
    try:
        return int(float(competitor.get("score")))
    except (TypeError, ValueError):
        return None


def parse_team(competitor: dict) -> dict | None:
    team = competitor.get("team") or {}
    tid = str(team.get("id") or "")
    name = team.get("displayName") or team.get("location") or team.get("name")
    if not (tid.isdigit() and int(tid) > 0) or not name or str(name).upper() == "TBD":
        return None
    return {
        "team_id": tid,
        "team": str(name),
        "short": str(team.get("shortDisplayName") or name),
        "abbr": str(team.get("abbreviation") or name)[:5],
        "color": str(team.get("color") or ""),
        "alt_color": str(team.get("alternateColor") or ""),
    }


def parse_match(event: dict, league: str) -> dict | None:
    """One scoreboard event -> flat row. None if a team is missing or the event is malformed."""
    comp = dig(event, "competitions", 0)
    if not isinstance(comp, dict):
        return None
    sides: dict[str, dict] = {}
    for competitor in comp.get("competitors") or []:
        side = competitor.get("homeAway")
        team = parse_team(competitor)
        if side in ("home", "away") and team is not None:
            sides[side] = {**team, "goals": _goals(competitor)}
    if set(sides) != {"home", "away"}:
        return None
    try:
        date = datetime.fromisoformat(str(event.get("date")).replace("Z", "+00:00"))
        date = date.astimezone(timezone.utc)
    except ValueError:
        return None
    status = parse_status(event)
    home, away = sides["home"], sides["away"]
    if status == "final" and (home["goals"] is None or away["goals"] is None):
        status = "unknown"  # a "final" without a score cannot be used
    completed = status == "final"
    return {
        "match_id": str(event.get("id")),
        "league": league,
        "season": dig(event, "season", "year"),
        "date": date,
        "week_start": week_start(date),
        "status": status,
        "completed": completed,
        "home_id": home["team_id"],
        "home_team": home["team"],
        "home_short": home["short"],
        "home_abbr": home["abbr"],
        "home_goals": home["goals"] if completed else None,
        "away_id": away["team_id"],
        "away_team": away["team"],
        "away_short": away["short"],
        "away_abbr": away["abbr"],
        "away_goals": away["goals"] if completed else None,
    }


def parse_match_teams(event: dict) -> list[dict]:
    out = []
    for competitor in dig(event, "competitions", 0, "competitors", default=[]) or []:
        team = parse_team(competitor)
        if team is not None:
            out.append(team)
    return out


def parse_odds_entry(entry: dict) -> dict:
    """Normalize one ESPN odds block for a soccer match (three-way moneyline, total, handicap).

    Works for the scoreboard's `odds[i]`, the summary's `pickcenter[i]` and the
    summary's older `odds[i]` entries. `spread_home` is the HOME team's handicap.
    Anything missing stays None.
    """

    def ml(side: str, legacy: tuple, which: str = "close") -> int | None:
        value = parse_american(dig(entry, "moneyline", side, which, "odds"))
        if value is None and which == "close":
            value = parse_american(dig(entry, *legacy))
        return value

    spread_home = parse_line(dig(entry, "pointSpread", "home", "close", "line"))
    if spread_home is None:
        away = parse_line(dig(entry, "pointSpread", "away", "close", "line"))
        spread_home = -away if away is not None else parse_line(entry.get("spread"))
    total = parse_line(dig(entry, "total", "over", "close", "line"))
    if total is None:
        total = parse_line(entry.get("overUnder"))

    def price(*paths: tuple) -> int | None:
        for path in paths:
            value = parse_american(dig(entry, *path))
            if value is not None:
                return value
        return None

    return {
        "provider": dig(entry, "provider", "name"),
        "ml_home": ml("home", ("homeTeamOdds", "moneyLine")),
        "ml_draw": ml("draw", ("drawOdds", "moneyLine")),
        "ml_away": ml("away", ("awayTeamOdds", "moneyLine")),
        "total": total,
        "over_odds": price(("total", "over", "close", "odds"), ("overOdds",)),
        "under_odds": price(("total", "under", "close", "odds"), ("underOdds",)),
        "spread_home": spread_home,
        "spread_home_odds": price(
            ("pointSpread", "home", "close", "odds"), ("homeTeamOdds", "spreadOdds")
        ),
        "spread_away_odds": price(
            ("pointSpread", "away", "close", "odds"), ("awayTeamOdds", "spreadOdds")
        ),
        "ml_home_open": ml("home", (), "open"),
        "ml_draw_open": ml("draw", (), "open"),
        "ml_away_open": ml("away", (), "open"),
        "total_open": parse_line(dig(entry, "total", "over", "open", "line")),
    }


def three_way_overround(odds: dict) -> float | None:
    """Sum of the three implied probabilities (1.0 = no bookmaker margin)."""
    prices = [odds.get("ml_home"), odds.get("ml_draw"), odds.get("ml_away")]
    if any(p is None for p in prices):
        return None
    return sum(100 / (p + 100) if p > 0 else -p / (-p + 100) for p in prices)


def is_plausible(odds: dict) -> bool:
    """A usable pre-match quote: all three prices, with a margin a bookmaker would post."""
    over = three_way_overround(odds)
    return over is not None and 1.0 <= over <= 1.2


def parse_odds(entries) -> dict | None:
    """Pick the best pre-match quote from a list of ESPN odds blocks. None if there is none."""
    if not isinstance(entries, list):
        return None
    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(dig(entry, "provider", "name", default="")).lower()
        if "live" in name:
            continue
        odds = parse_odds_entry(entry)
        if is_plausible(odds):
            compact = name.replace(" ", "")
            pref = next(
                (i for i, key in enumerate(PROVIDER_PREFERENCE) if key in compact),
                len(PROVIDER_PREFERENCE),
            )
            filled = sum(odds.get(k) is not None for k in ("total", "over_odds", "spread_home"))
            candidates.append((pref, -filled, len(candidates), odds))
    return min(candidates)[3] if candidates else None


def parse_summary_odds(summary: dict) -> dict | None:
    """Closing odds from a match summary: the pickcenter block first, then the older list."""
    return parse_odds(summary.get("pickcenter")) or parse_odds(summary.get("odds"))


# ------------------------------------------------------------------ fetching
def _month_settled(yyyymm: str):
    """A month can be cached forever once it is over and every match in it is decided."""
    year, month = int(yyyymm[:4]), int(yyyymm[4:])
    month_end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)

    def settled(payload: dict) -> bool:
        if datetime.now(timezone.utc) < month_end + timedelta(days=3):
            return False
        for event in payload.get("events") or []:
            if parse_status(event) not in ("final", "canceled", "postponed"):
                return False
        return True

    return settled


def fetch_month(client: EspnClient, league: str, yyyymm: str, *, refresh: bool = False) -> dict:
    return client.get(
        f"soccer/{league}/{yyyymm}",
        f"{config.ESPN_SOCCER}/{league}/scoreboard",
        {"dates": yyyymm, "limit": 300},
        max_age=900,
        settled=_month_settled(yyyymm),
        refresh=refresh,
    )


def fetch_league_season(
    client: EspnClient, league: str, season: int, *, refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """All matches of one league season. Returns (matches, teams, live odds by match id)."""
    rows: list[dict] = []
    teams: dict[str, dict] = {}
    odds: dict[str, dict] = {}
    for yyyymm in config.season_months(season):
        payload = fetch_month(client, league, yyyymm, refresh=refresh)
        for event in payload.get("events") or []:
            row = parse_match(event, league)
            if row is None or row["season"] != season:
                continue
            rows.append(row)
            for team in parse_match_teams(event):
                teams[team["team_id"]] = {**team, "league": league, "season": season}
            if row["status"] == "scheduled":
                quote = parse_odds(dig(event, "competitions", 0, "odds"))
                if quote:
                    odds[row["match_id"]] = quote
    matches = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    if len(matches):
        # A postponed match shows up again on its new date: keep the played version.
        matches = (
            matches.sort_values(["completed", "date"])
            .drop_duplicates("match_id", keep="last")
            .sort_values(["date", "match_id"])
            .reset_index(drop=True)
        )
    return matches, pd.DataFrame(teams.values()), odds


def _typed(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    for col in ("date", "week_start"):
        matches[col] = pd.to_datetime(matches[col], utc=True)
    for col in ("home_goals", "away_goals"):
        matches[col] = pd.to_numeric(matches[col], errors="coerce").astype("Int64")
    matches["season"] = matches["season"].astype("int64")
    matches["completed"] = matches["completed"].astype(bool)
    # ESPN leaves a few old matches as "scheduled" with no score. They are not fixtures.
    stale = (matches["status"] == "scheduled") & (
        matches["date"] < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
    )
    matches.loc[stale, "status"] = "unknown"
    return matches.sort_values(["date", "match_id"]).reset_index(drop=True)


def load_matches(path=config.MATCHES_PATH) -> pd.DataFrame:
    return _typed(pd.read_parquet(path))


def load_teams(path=config.TEAMS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def build(
    season: int,
    history: list[int],
    *,
    refresh: bool = False,
    offline: bool = False,
    client: EspnClient | None = None,
    log=print,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Download every league for the given seasons. Returns (all matches, live odds)."""
    client = client or EspnClient(offline=offline)
    existing = load_matches() if config.MATCHES_PATH.exists() else None
    frames, team_frames, live_odds = [], [], {}
    for league, name in config.LEAGUES.items():
        for year in sorted(set(history) | {season}):
            if year != season and existing is not None:
                prev = existing[(existing["league"] == league) & (existing["season"] == year)]
                # A finished season never changes: reuse what is already in the parquet.
                if len(prev) >= 300 and prev["completed"].mean() > 0.99:
                    frames.append(prev[MATCH_COLUMNS])
                    continue
            matches, teams, odds = fetch_league_season(
                client, league, year, refresh=refresh and year == season
            )
            if matches.empty:
                raise EspnError(f"ESPN returned no matches for {name} {year}")
            frames.append(matches)
            team_frames.append(teams)
            if year == season:
                live_odds.update(odds)
        done = sum(int(f["completed"].sum()) for f in frames if (f["league"] == league).all())
        log(f"{name}: {done} matches played across the seasons loaded")

    all_matches = _typed(pd.concat(frames, ignore_index=True))
    config.MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_matches.to_parquet(config.MATCHES_PATH, index=False)
    if team_frames:
        teams = pd.concat(team_frames, ignore_index=True)
        if config.TEAMS_PATH.exists():
            teams = pd.concat([load_teams(), teams], ignore_index=True)
        # One record per team: the most recent season seen.
        teams = teams.sort_values("season").drop_duplicates("team_id", keep="last")
        teams.sort_values(["league", "team"]).reset_index(drop=True).to_parquet(
            config.TEAMS_PATH, index=False
        )
    log(f"API requests: {client.stats['network']} · from cache: {client.stats['cache']}")
    return all_matches, live_odds


# ------------------------------------------------------------- closing odds
# Provider-seasons whose archived goals line is not a pre-match line. In 2023-24
# ESPN's old DraftKings feed listed "over/under 4.5" for nine matches in ten and
# kept updating it during play: the matches where that line was priced near even
# averaged 5.7 goals. Its moneylines and handicaps for the same season track the
# model like every other season's, so only the goals line is dropped.
UNRELIABLE_TOTALS = {("DraftKings (old)", 2023)}
_TOTAL_COLUMNS = ["total", "over_odds", "under_odds", "total_open"]


def drop_unreliable_totals(odds: pd.DataFrame) -> pd.DataFrame:
    """Blank the goals line of the provider-seasons listed in UNRELIABLE_TOTALS."""
    odds = odds.copy()
    for provider, season in UNRELIABLE_TOTALS:
        bad = (odds["provider"] == provider) & (odds["season"] == season)
        odds.loc[bad, _TOTAL_COLUMNS] = pd.NA
    return odds


def load_odds_history(path=config.ODDS_HISTORY_PATH, *, raw: bool = False) -> pd.DataFrame:
    """Closing odds of finished matches. `raw=True` returns the file as downloaded."""
    if not path.exists():
        return pd.DataFrame(columns=["match_id", "league", "season", "has_odds", *ODDS_COLUMNS])
    odds = pd.read_parquet(path)
    return odds if raw else drop_unreliable_totals(odds)


def fetch_odds_history(
    seasons: list[int] | None = None, *, workers: int = 6, offline: bool = False, log=print
) -> pd.DataFrame:
    """Closing odds for finished matches that are not in the parquet yet (incremental)."""
    matches = load_matches()
    matches = matches[matches["completed"]]
    if seasons is not None:
        matches = matches[matches["season"].isin(seasons)]
    existing = load_odds_history(raw=True)  # the file keeps what ESPN sent, unfiltered
    todo = matches[~matches["match_id"].isin(existing["match_id"])]
    log(f"{len(matches)} matches played, {len(todo)} without stored odds")
    if todo.empty:
        return drop_unreliable_totals(existing)

    client = EspnClient(offline=offline, pause=0.03)
    failed = 0

    def one(row) -> dict | None:
        nonlocal failed
        try:
            summary = client.get(
                f"soccer_summary/{row.league}/{row.match_id}",
                f"{config.ESPN_SOCCER}/{row.league}/summary",
                {"event": row.match_id},
                max_age=None,
                transform=lambda p: {
                    "pickcenter": _strip_links(p.get("pickcenter") or []),
                    "odds": [
                        _strip_links({k: v for k, v in o.items() if k != "bettingOdds"})
                        for o in (p.get("odds") or [])
                        if isinstance(o, dict)
                    ],
                },
            )
        except EspnError:
            failed += 1  # not stored: it is retried on the next run
            return None
        odds = parse_summary_odds(summary)
        out = {
            "match_id": row.match_id,
            "league": row.league,
            "season": row.season,
            "has_odds": odds is not None,
        }
        out.update({col: (odds or {}).get(col) for col in ODDS_COLUMNS})
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(one, list(todo.itertuples())) if r is not None]

    new = pd.DataFrame(rows, columns=["match_id", "league", "season", "has_odds", *ODDS_COLUMNS])
    out = new if existing.empty else pd.concat([existing, new], ignore_index=True)
    out = out.drop_duplicates("match_id", keep="last").reset_index(drop=True)
    for col in ODDS_COLUMNS:
        if col == "provider":
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        is_line = col in ("total", "spread_home", "total_open")
        out[col] = numeric if is_line else numeric.astype("Int64")
    out["has_odds"] = out["has_odds"].astype(bool)
    out["season"] = out["season"].astype("int64")
    out.to_parquet(config.ODDS_HISTORY_PATH, index=False)
    got = int(new["has_odds"].sum()) if len(new) else 0
    log(
        f"{len(new)} looked up, {got} with odds, {failed} failed"
        f" · API {client.stats['network']} · cache {client.stats['cache']}"
    )
    return drop_unreliable_totals(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download soccer matches (and odds) from ESPN")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument("--no-history", action="store_true", help="only the current season")
    parser.add_argument("--odds", action="store_true", help="also download closing odds")
    parser.add_argument("--refresh", action="store_true", help="ignore the current season's cache")
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    args = parser.parse_args(argv)

    history = [] if args.no_history else [y for y in config.HISTORY_SEASONS if y < args.season]
    try:
        matches, live = build(args.season, history, refresh=args.refresh, offline=args.offline)
        if args.odds:
            fetch_odds_history(offline=args.offline)
    except EspnError as exc:
        print(f"ERROR: the ESPN API failed and no data is made up: {exc}", file=sys.stderr)
        return 1
    summary = (
        matches.groupby(["season", "league"])
        .agg(matches=("match_id", "size"), played=("completed", "sum"))
        .unstack("league")
    )
    print("\nMatches per season and league (total / played):")
    print(summary.to_string())
    cur = matches[matches["season"] == args.season]
    print(
        f"\nSeason {args.season}: {len(cur)} matches, {int(cur['completed'].sum())} played,"
        f" {len(live)} upcoming with odds. Saved to"
        f" {config.MATCHES_PATH.relative_to(config.base.ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
