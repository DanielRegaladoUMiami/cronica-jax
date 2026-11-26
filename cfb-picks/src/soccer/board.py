"""Champions League, Europa League and Conference League: market odds vs Kalshi prices.

There is no model of ours here: nothing in this project has been tested on
European cup matches. What the board does is honest arithmetic:

1. Take the sportsbook's three-way odds (home / draw / away) from ESPN and remove
   the bookmaker's margin. What is left is the market's probability of each result.
2. Take Kalshi's prices for the same match. On Kalshi you can buy "Yes" or "No" on
   each result, so "No on team X" wins when X draws or loses.
3. Compare. A contract is worth its probability: "No on X" is worth 1 - P(X wins).
   After Kalshi's fee, the difference is what the price gives away or costs.

A "cheap" contract means Kalshi and the sportsbook disagree. It is not a proven edge.

Usage:
    python -m src.soccer.board              # the next 7 days
    python -m src.soccer.board --days 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from .. import config
from ..data import kalshi
from ..data.espn import EspnClient, EspnError
from ..data.parse import parse_american, parse_status

BOARD_PATH = config.SITE_DATA_DIR / "soccer.json"
NOTES_PATH = config.DATA_DIR / "soccer_notes.csv"
OUTCOMES = ("home", "draw", "away")


@dataclass(frozen=True)
class Competition:
    key: str
    name: str
    espn_path: str
    kalshi_series: str


COMPETITIONS = (
    Competition("ucl", "Champions League", "soccer/uefa.champions", "KXUCLGAME"),
    Competition("uel", "Europa League", "soccer/uefa.europa", "KXUELGAME"),
    Competition("uecl", "Conference League", "soccer/uefa.europa.conf", "KXUECLGAME"),
)


# ------------------------------------------------------------- probabilities
def implied(odds: float) -> float:
    """American odds -> the raw probability they imply (margin still inside)."""
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def shin(raw: list[float]) -> list[float]:
    """Remove the bookmaker's margin with Shin's method.

    Splitting the margin evenly overstates long shots: bookmakers load more margin
    on them. Shin's method assumes a share `z` of the money is better informed and
    solves for the probabilities that add up to 1. With no margin it returns `raw`.
    """
    total = sum(raw)
    if total <= 1.0 or any(r <= 0 for r in raw):
        return [r / total for r in raw]

    def probs(z: float) -> list[float]:
        return [
            (math.sqrt(z * z + 4.0 * (1.0 - z) * r * r / total) - z) / (2.0 * (1.0 - z))
            for r in raw
        ]

    lo, hi = 0.0, 0.5
    for _ in range(80):
        z = (lo + hi) / 2.0
        if sum(probs(z)) > 1.0:
            lo = z
        else:
            hi = z
    out = probs((lo + hi) / 2.0)
    scale = sum(out)
    return [p / scale for p in out]


# ------------------------------------------------------------------ ESPN side
def _moneyline(block: dict | None, when: str) -> float | None:
    return parse_american(((block or {}).get(when) or {}).get("odds"))


def parse_match(event: dict, comp: Competition) -> dict | None:
    """One match from ESPN's scoreboard, with the sportsbook's odds if there are any."""
    comps = event.get("competitions") or []
    if not comps:
        return None
    game = comps[0]
    sides = {c.get("homeAway"): c for c in game.get("competitors") or []}
    if "home" not in sides or "away" not in sides:
        return None

    def team(side: dict) -> dict:
        info = side.get("team") or {}
        records = side.get("records") or []
        score = side.get("score")
        return {
            "id": str(info.get("id")),
            "name": info.get("displayName") or info.get("name"),
            "short": info.get("shortDisplayName") or info.get("displayName"),
            "abbr": info.get("abbreviation"),
            "record": records[0].get("summary") if records else None,
            "form": side.get("form") or None,
            "score": int(score) if str(score or "").isdigit() else None,
        }

    status = parse_status(event)
    match = {
        "id": str(event.get("id")),
        "competition": comp.key,
        "kickoff": event.get("date"),
        "status": status,
        "venue": ((game.get("venue") or {}).get("address") or {}).get("city"),
        "home": team(sides["home"]),
        "away": team(sides["away"]),
        "book": None,
        "fair": None,
        "fair_open": None,
        "kalshi": None,
        "contracts": [],
        "note": None,
    }
    odds = (game.get("odds") or [None])[0] or {}
    line = odds.get("moneyline") or {}
    close = [_moneyline(line.get(k), "close") for k in OUTCOMES]
    opened = [_moneyline(line.get(k), "open") for k in OUTCOMES]
    if all(v is not None and v != 0 for v in close):
        total = odds.get("total") or {}
        match["book"] = {
            "provider": (odds.get("provider") or {}).get("name"),
            "ml": dict(zip(OUTCOMES, close)),
            "total": odds.get("overUnder"),
            "over": _moneyline(total.get("over"), "close"),
            "under": _moneyline(total.get("under"), "close"),
        }
        match["fair"] = dict(zip(OUTCOMES, shin([implied(v) for v in close])))
        if all(v is not None and v != 0 for v in opened):
            match["fair_open"] = dict(zip(OUTCOMES, shin([implied(v) for v in opened])))
    return match


def fetch_matches(
    client: EspnClient, comp: Competition, start: datetime, end: datetime
) -> list[dict]:
    """Every match between two dates. ESPN's soccer scoreboard takes one day at a time."""
    found: dict[str, dict] = {}
    day = start
    while day.date() <= end.date():
        payload = client.get(
            f"soccer/{comp.key}/scoreboard_{day:%Y%m%d}",
            f"{config.ESPN_HOST}/site/v2/sports/{comp.espn_path}/scoreboard",
            {"dates": f"{day:%Y%m%d}", "limit": 100},
            max_age=600,
        )
        for event in payload.get("events") or []:
            match = parse_match(event, comp)
            if match is not None:
                found[match["id"]] = match
        day += timedelta(days=1)
    return list(found.values())


def fetch_table(client: EspnClient, comp: Competition) -> dict[str, dict]:
    """League-phase table: team id -> rank and points. Empty if ESPN has none."""
    try:
        payload = client.get(
            f"soccer/{comp.key}/standings",
            f"{config.ESPN_HOST}/v2/sports/{comp.espn_path}/standings",
            max_age=3 * 3600,
        )
    except EspnError:
        return {}
    table: dict[str, dict] = {}
    for child in payload.get("children") or []:
        for entry in (child.get("standings") or {}).get("entries") or []:
            stats = {s.get("name"): s.get("value") for s in entry.get("stats") or []}
            if stats.get("rank") is None:
                continue
            table[str((entry.get("team") or {}).get("id"))] = {
                "rank": int(stats["rank"]),
                "points": int(stats.get("points") or 0),
                "goal_diff": int(stats.get("pointDifferential") or 0),
            }
    return table


# ---------------------------------------------------------------- Kalshi side
_DROP_TOKENS = {
    "fc", "cf", "sc", "ac", "as", "afc", "sk", "fk", "nk", "bk", "if", "ff", "cd", "sv", "kv",
    "club", "de", "the", "and",
}  # fmt: skip
_ALIASES = {
    "psg": "paris saint germain",
    "inter": "internazionale",
    "inter milan": "internazionale",
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham hotspur",
    "atletico": "atletico madrid",
    "leverkusen": "bayer leverkusen",
    "brugge": "club brugge",
    "union sg": "union st gilloise",
    "be'er sheva": "hapoel be'er sheva",
}


def _plain(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    text = _ALIASES.get(text.strip(), text)
    return re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", "")).strip()


def name_score(a: str, b: str) -> float:
    """How alike two club names are, 0 to 1 ("AC Milan" vs "Milan" -> 1)."""
    pa, pb = _plain(a), _plain(b)
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0
    ta = {t for t in pa.split() if t not in _DROP_TOKENS} or set(pa.split())
    tb = {t for t in pb.split() if t not in _DROP_TOKENS} or set(pb.split())
    if ta == tb:
        return 1.0
    if ta <= tb or tb <= ta:
        return 0.8
    overlap = len(ta & tb) / len(ta | tb)
    alike = SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return max(overlap, 0.85 * alike)


def _event_day(ticker: str) -> str | None:
    """'KXUELGAME-26NOV26OLYACM' -> '2026-11-26'."""
    found = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", ticker or "")
    if not found:
        return None
    try:
        day = datetime.strptime("".join(found.groups()), "%y%b%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return f"{day:%Y-%m-%d}"


def _event_sides(event: dict) -> dict | None:
    """Split a Kalshi match event into its home, away and draw markets."""
    title = event.get("title") or ""
    if " vs " not in title:
        return None
    home_name, away_name = (part.strip() for part in title.split(" vs ", 1))
    draw = None
    teams = []
    for raw in event.get("markets") or []:
        market = kalshi.parse_market(raw)
        code = (market["ticker"] or "").rsplit("-", 1)[-1]
        if code == "TIE" or (market["label"] or "").lower() in {"tie", "draw"}:
            draw = market
        else:
            teams.append((code, market))
    if draw is None or len(teams) != 2:
        return None
    # The title is "home vs away"; pair each team market with the side it names.
    first = max(teams, key=lambda t: name_score(t[1]["label"] or "", home_name))
    second = teams[1] if first is teams[0] else teams[0]
    return {
        "event": event.get("event_ticker"),
        "day": _event_day(event.get("event_ticker") or ""),
        "home_name": home_name,
        "away_name": away_name,
        "home_code": first[0],
        "away_code": second[0],
        "home": first[1],
        "away": second[1],
        "draw": draw,
    }


def match_kalshi(matches: list[dict], events: list[dict], floor: float = 0.55) -> dict[str, dict]:
    """Pair ESPN matches with Kalshi events by day and club names.

    Each pair needs both clubs to look alike; an ambiguous or weak pairing is left
    without a Kalshi price rather than guessed.
    """
    sides = [s for s in (_event_sides(ev) for ev in events) if s is not None]
    scored = []
    for match in matches:
        day = (match.get("kickoff") or "")[:10]
        for idx, side in enumerate(sides):
            if side["day"] and day and abs(_days_apart(side["day"], day)) > 1:
                continue
            home = max(
                name_score(match["home"]["name"], side["home_name"]),
                name_score(match["home"]["short"], side["home_name"]),
                name_score(match["home"]["name"], side["home"]["label"] or ""),
                0.9 if match["home"]["abbr"] == side["home_code"] else 0.0,
            )
            away = max(
                name_score(match["away"]["name"], side["away_name"]),
                name_score(match["away"]["short"], side["away_name"]),
                name_score(match["away"]["name"], side["away"]["label"] or ""),
                0.9 if match["away"]["abbr"] == side["away_code"] else 0.0,
            )
            # One clear club plus a plausible partner is enough; two weak ones are not.
            score = (home + away) / 2.0 if min(home, away) >= 0.3 else 0.0
            if score >= floor:
                scored.append((score, match["id"], idx))
    scored.sort(reverse=True)
    out: dict[str, dict] = {}
    used: set[int] = set()
    for _score, match_id, idx in scored:
        if match_id in out or idx in used:
            continue
        out[match_id] = sides[idx]
        used.add(idx)
    return out


def _days_apart(a: str, b: str) -> int:
    return (datetime.fromisoformat(a) - datetime.fromisoformat(b)).days


# ------------------------------------------------------------------ contracts
def contracts(match: dict, fee_type: str | None) -> list[dict]:
    """The six things you can buy on a match, each with its fair value and edge.

    edge       = fair value - price you pay now - fee
    edge_rest  = the same if you leave an order at the current bid and it fills
    """
    fair, markets = match.get("fair"), (match.get("kalshi") or {}).get("markets")
    if not fair or not markets:
        return []
    names = {"home": match["home"]["short"], "away": match["away"]["short"]}
    out = []
    for outcome in OUTCOMES:
        market = markets.get(outcome) or {}
        if not market.get("quoted"):
            continue
        for side in ("yes", "no"):
            value = fair[outcome] if side == "yes" else 1.0 - fair[outcome]
            ask, bid = market[f"{side}_ask"], market[f"{side}_bid"]
            if outcome == "draw":
                text = "Draw" if side == "yes" else "No draw (either team wins)"
            else:
                text = f"{names[outcome]} win" if side == "yes" else f"{names[outcome]} do not win"
            fee = kalshi.taker_fee(ask)
            out.append(
                {
                    "outcome": outcome,
                    "side": side,
                    "text": text,
                    "price": ask,
                    "bid": bid,
                    "fair": value,
                    "fee": fee,
                    "edge": value - ask - fee,
                    "edge_rest": value - bid - kalshi.maker_fee(bid, fee_type),
                    "volume": market.get("volume"),
                    "ticker": market.get("ticker"),
                }
            )
    return out


def load_notes(path=None) -> list[dict]:
    """Hand-written team news: date (YYYY-MM-DD), team, note."""
    path = path or NOTES_PATH
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh) if (row.get("team") or "").strip()]


def attach_notes(matches: list[dict], notes: list[dict]) -> None:
    for match in matches:
        day = (match.get("kickoff") or "")[:10]
        found = []
        for row in notes:
            if (row.get("date") or "").strip() not in ("", day):
                continue
            for side in ("home", "away"):
                if name_score(row["team"], match[side]["name"]) >= 0.9:
                    found.append(f"{match[side]['short']}: {row.get('note', '').strip()}")
        match["note"] = " · ".join(found) or None


# ---------------------------------------------------------------------- board
def build_board(now: datetime | None = None, days: int = 7, log=print) -> dict:
    now = now or datetime.now(timezone.utc)
    start, end = now - timedelta(days=1), now + timedelta(days=days)
    client = EspnClient(pause=0.05)
    market = kalshi.KalshiClient()
    notes = load_notes()
    board = {
        "sport": "soccer",
        "generated_at": now.isoformat(timespec="seconds"),
        "fee": {"taker_rate": kalshi.TAKER_RATE, "maker_rate": kalshi.MAKER_RATE},
        "competitions": [],
        "matches": [],
        "warnings": [],
    }
    for comp in COMPETITIONS:
        try:
            matches = fetch_matches(client, comp, start, end)
        except EspnError as exc:
            board["warnings"].append(f"{comp.name}: ESPN did not answer ({exc}). No matches shown.")
            continue
        table = fetch_table(client, comp)
        for match in matches:
            for side in ("home", "away"):
                match[side].update(table.get(match[side]["id"], {}))
        fee_type = None
        pairs: dict[str, dict] = {}
        try:
            fee_type = market.series(comp.kalshi_series).get("fee_type")
            pairs = match_kalshi(matches, market.events(comp.kalshi_series))
        except kalshi.KalshiError as exc:
            board["warnings"].append(
                f"{comp.name}: Kalshi did not answer ({exc}). No prices shown."
            )
        for match in matches:
            pair = pairs.get(match["id"])
            if pair is not None:
                match["kalshi"] = {
                    "event": pair["event"],
                    "fee_type": fee_type,
                    "markets": {k: pair[k] for k in OUTCOMES},
                }
            match["contracts"] = contracts(match, fee_type)
        attach_notes(matches, notes)
        upcoming = [m for m in matches if m["status"] in ("scheduled", "in_progress")]
        board["competitions"].append(
            {
                "key": comp.key,
                "name": comp.name,
                "matches": len(matches),
                "upcoming": len(upcoming),
                "with_odds": sum(1 for m in upcoming if m["fair"]),
                "with_kalshi": sum(1 for m in upcoming if m["kalshi"]),
            }
        )
        board["matches"].extend(matches)
        log(
            f"{comp.name}: {len(matches)} matches, {len(upcoming)} to play,"
            f" {sum(1 for m in upcoming if m['fair'])} with odds,"
            f" {sum(1 for m in upcoming if m['kalshi'])} with Kalshi prices"
        )
    board["matches"].sort(key=lambda m: (m["kickoff"] or "", m["competition"], m["id"]))
    return board


LIKELY_PRICE = 0.30  # under it a contract is a long shot: never listed as a good price


def best_contracts(board: dict, limit: int = 12) -> list[tuple[dict, dict]]:
    """The contracts whose price is furthest below fair value, among matches not yet played.

    Long shots are left out. A one-cent gap on an 11-cent contract is noise, and on
    Kalshi cheap contracts win less often than their price implies.
    """
    rows = [
        (c, m)
        for m in board["matches"]
        if m["status"] == "scheduled"
        for c in m["contracts"]
        if LIKELY_PRICE <= c["price"] <= 0.95
    ]
    rows.sort(key=lambda r: r[0]["edge"], reverse=True)
    return rows[:limit]


def print_summary(board: dict) -> None:
    for warning in board["warnings"]:
        print("WARNING:", warning)
    print("\nBest prices against the sportsbook (after Kalshi's fee):")
    for contract, match in best_contracts(board):
        print(
            f"  {contract['edge'] * 100:+5.1f}c  {contract['text']:<34s} {contract['price'] * 100:3.0f}c"
            f" (fair {contract['fair'] * 100:4.1f}c) · {match['home']['short']} v {match['away']['short']}"
            f" · {match['competition'].upper()} {match['kickoff'][:16]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="European soccer: market odds vs Kalshi prices")
    parser.add_argument("--days", type=int, default=7, help="how many days ahead to look")
    args = parser.parse_args(argv)
    board = build_board(days=args.days)
    if not board["matches"] and board["warnings"]:
        print("ERROR: no data, and none is made up:", "; ".join(board["warnings"]), file=sys.stderr)
        return 1
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_PATH.write_text(json.dumps(board, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print_summary(board)
    print(f"\nExported: {BOARD_PATH.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
