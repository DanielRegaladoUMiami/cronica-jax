"""What the market says about each player: DraftKings lines and Kalshi prices.

- DraftKings (through ESPN): one line per player and statistic, such as
  "passing yards 279.5". ESPN publishes the line but not its price, so it is used
  as the market's midpoint for yardage, and shown as a reference for the rest.
- Kalshi: a ladder of yes/no contracts per player, such as "250+ passing yards"
  at 65 cents. That price is the market's probability, and it is what a
  projection has to beat (after the fee) to be worth anything.

If a source does not answer, its numbers are simply missing: nothing is filled in.
"""

from __future__ import annotations

import re
import unicodedata

from .. import config
from ..data import kalshi
from ..data.espn import EspnClient, EspnError

CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

# ESPN's prop type id -> our statistic.
DK_TYPES = {
    "8": "pass_yds",
    "9": "pass_cmp",
    "10": "pass_td",
    "15": "pass_int",
    "16": "pass_att",
    "11": "rush_att",
    "12": "rush_yds",
    "13": "rec_yds",
    "14": "rec",
}
KALSHI_SERIES = {
    "KXNFLPASSYDS": "pass_yds",
    "KXNFLPASSTDS": "pass_td",
    "KXNFLPASSINT": "pass_int",
    "KXNFLPASSATT": "pass_att",
    "KXNFLPASSCOMP": "pass_cmp",
    "KXNFLRSHYDS": "rush_yds",
    "KXNFLRSHATT": "rush_att",
    "KXNFLRECYDS": "rec_yds",
    "KXNFLREC": "rec",
    "KXNFLTD": "td",
}
# Kalshi's team codes that differ from ESPN's abbreviations.
KALSHI_TEAM = {"JAC": "JAX", "WAS": "WSH", "LA": "LAR"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def plain_name(name: str) -> str:
    """'Amon-Ra St. Brown' and 'Amon-Ra St. Brown Jr.' -> 'amonra st brown'."""
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    tokens = re.sub(r"[^a-z ]+", "", text.replace("-", "")).split()
    return " ".join(t for t in tokens if t not in _SUFFIXES)


# ------------------------------------------------------------------ DraftKings
def _athlete_id(ref: str | None) -> str | None:
    found = re.search(r"/athletes/(\d+)", ref or "")
    return found.group(1) if found else None


def trim_props(payload: dict) -> dict:
    """Keep the player lines that are used: (athlete, statistic, line, opening line)."""
    rows = []
    for item in payload.get("items") or []:
        stat = DK_TYPES.get(str((item.get("type") or {}).get("id")))
        athlete = _athlete_id((item.get("athlete") or {}).get("$ref"))
        line = ((item.get("current") or {}).get("target") or {}).get("value")
        if stat is None or athlete is None or not isinstance(line, (int, float)):
            continue
        opened = ((item.get("open") or {}).get("target") or {}).get("value")
        rows.append(
            [
                athlete,
                stat,
                float(line),
                float(opened) if isinstance(opened, (int, float)) else None,
            ]
        )
    return {"count": payload.get("count"), "pages": payload.get("pageCount") or 1, "rows": rows}


def draftkings_lines(client: EspnClient, game_id: str, max_age: float = 900) -> dict:
    """{(athlete id, statistic): {"line": 279.5, "open": 275.5}} for one game."""
    out: dict[tuple[str, str], dict] = {}
    url = f"{CORE}/events/{game_id}/competitions/{game_id}/odds"
    listing = client.get(f"nfl/props/{game_id}_providers", url, {"limit": 20}, max_age=max_age)
    providers = [
        str((item.get("provider") or {}).get("id"))
        for item in listing.get("items") or []
        if item.get("propBets")
    ]
    if not providers:
        return out
    provider = providers[0]
    page, pages = 1, 1
    while page <= pages and page <= 6:
        data = client.get(
            f"nfl/props/{game_id}_{provider}_p{page}",
            f"{url}/{provider}/propBets",
            {"limit": 1000, "page": page},
            max_age=max_age,
            transform=trim_props,
        )
        pages = int(data.get("pages") or 1)
        for athlete, stat, line, opened in data.get("rows") or []:
            out[(athlete, stat)] = {"line": line, "open": opened}
        page += 1
    return out


# ---------------------------------------------------------------------- Kalshi
def split_teams(tail: str, known: set[str]) -> tuple[str, str] | None:
    """'CHIDET' -> ('CHI', 'DET'), with Kalshi's own codes mapped to ESPN's."""
    for cut in (2, 3, 4):
        first, second = tail[:cut], tail[cut:]
        a, b = KALSHI_TEAM.get(first, first), KALSHI_TEAM.get(second, second)
        if a in known and b in known:
            return a, b
    return None


def event_teams(event_ticker: str, known: set[str]) -> tuple[str, str] | None:
    """'KXNFLPASSYDS-26NOV26CHIDET' -> ('CHI', 'DET'): away team first."""
    found = re.search(r"-\d{2}[A-Z]{3}\d{2}([A-Z]+)$", event_ticker or "")
    return split_teams(found.group(1), known) if found else None


def kalshi_ladders(client: kalshi.KalshiClient, known: set[str], log=print) -> dict:
    """{(away, home): {plain player name: {statistic: [rungs]}}} for every open NFL game.

    A rung is one contract: {"t": 250, "yes_bid", "yes_ask", "no_bid", "no_ask", "volume", "ticker"}.
    """
    out: dict[tuple[str, str], dict] = {}
    failed = []
    for series, stat in KALSHI_SERIES.items():
        try:
            events = client.events(series)
        except kalshi.KalshiError:
            failed.append(series)
            continue
        for event in events:
            teams = event_teams(event.get("event_ticker") or "", known)
            if teams is None:
                continue
            game = out.setdefault(teams, {})
            for raw in event.get("markets") or []:
                market = kalshi.parse_market(raw)
                label = market["label"] or ""
                if ":" not in label or market["strike"] is None or not market["quoted"]:
                    continue
                name = plain_name(label.split(":", 1)[0])
                if not name or "dst" in name.split():
                    continue
                game.setdefault(name, {}).setdefault(stat, []).append(
                    {
                        "t": round(market["strike"] + 0.5),
                        "yes_bid": market["yes_bid"],
                        "yes_ask": market["yes_ask"],
                        "no_bid": market["no_bid"],
                        "no_ask": market["no_ask"],
                        "volume": market["volume"],
                        "ticker": market["ticker"],
                    }
                )
    for game in out.values():
        for stats in game.values():
            for rungs in stats.values():
                rungs.sort(key=lambda r: r["t"])
    if failed:
        log(f"Kalshi: no answer for {', '.join(failed)}; those prices are left out")
    return out


def kalshi_games(client: kalshi.KalshiClient, known: set[str]) -> dict:
    """{(away, home): {"away": market, "home": market}}: who wins, in Kalshi's prices."""
    out: dict[tuple[str, str], dict] = {}
    for event in client.events("KXNFLGAME"):
        teams = event_teams(event.get("event_ticker") or "", known)
        if teams is None:
            continue
        sides = {}
        for raw in event.get("markets") or []:
            market = kalshi.parse_market(raw)
            code = (market["ticker"] or "").rsplit("-", 1)[-1]
            code = KALSHI_TEAM.get(code, code)
            if code == teams[0]:
                sides["away"] = market
            elif code == teams[1]:
                sides["home"] = market
        if len(sides) == 2:
            out[teams] = sides
    return out


def espn_client() -> EspnClient:
    return EspnClient(league=config.NFL, pause=0.05)


__all__ = [
    "DK_TYPES",
    "KALSHI_SERIES",
    "EspnError",
    "draftkings_lines",
    "espn_client",
    "event_teams",
    "kalshi_games",
    "kalshi_ladders",
    "plain_name",
    "split_teams",
    "trim_props",
]
