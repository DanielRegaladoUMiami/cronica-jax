"""This week's odds: your lines (data/odds_manual.csv) merged with ESPN's.

Merge rules, per market (moneyline, spread, total):
- The LINE and the PRICE used for a bet are yours when you captured them;
  otherwise they are ESPN's (DraftKings).
- The no-vig market probability comes from a two-sided quote from one book:
  yours if you captured both sides; otherwise ESPN's (same line); with only
  one side available, a typical vig is removed from it.
- A missing price is never invented silently: when -110 (the standard price
  for spreads and totals) is assumed, it is flagged as "assumed".
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .. import config
from ..model.prob import DEFAULT_OVERROUND, american_to_prob
from .parse import parse_american, parse_line

MANUAL_COLUMNS = [
    "season",
    "week",
    "away",
    "home",
    "ml_away",
    "ml_home",
    "spread_away",
    "spread_away_odds",
    "spread_home_odds",
    "total",
    "over_odds",
    "under_odds",
    "book",
    "captured_at",
]
STANDARD_PRICE = -110
DEFAULT_BOOK = "my book"

# Common ways of writing a team -> ESPN's short name (both already normalized).
ALIASES = {
    "miami fl": "miami",
    "miami florida": "miami",
    "miami ohio": "miami oh",
    "pitt": "pittsburgh",
    "southern california": "usc",
    "southern cal": "usc",
    "hawaii": "hawai i",
    "mississippi": "ole miss",
    "north carolina state": "nc state",
    "appalachian state": "app state",
    "connecticut": "uconn",
    "umass": "massachusetts",
    "louisiana monroe": "ul monroe",
    "ulm": "ul monroe",
    "louisiana lafayette": "louisiana",
    "central florida": "ucf",
    "brigham young": "byu",
    "texas christian": "tcu",
    "southern methodist": "smu",
    "louisiana state": "lsu",
    "sam houston state": "sam houston",
    "florida intl": "florida international",
    "fiu": "florida international",
}


def normalize_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"\bst\b\.?", "state", text)
    text = re.sub(r"[^a-z0-9&]+", " ", text).strip()
    return ALIASES.get(text, text)


def team_lookup(teams: pd.DataFrame) -> dict[str, str]:
    """Normalized name -> team_id. Accepts the short name, the full name and, when
    only one team has it, the abbreviation and the place ("Kansas City"; not "New York")."""
    lookup: dict[str, str] = {}
    abbr_count = teams["abbr"].str.lower().value_counts()
    places = teams["location"] if "location" in teams.columns else pd.Series("", index=teams.index)
    place_count = places.fillna("").str.lower().value_counts()
    for row, place in zip(teams.itertuples(), places.fillna(""), strict=True):
        lookup[normalize_team(row.team)] = row.team_id
        lookup.setdefault(normalize_team(row.display_name), row.team_id)
        if abbr_count.get(str(row.abbr).lower(), 0) == 1:
            lookup.setdefault(normalize_team(row.abbr), row.team_id)
        if place and place_count.get(place.lower(), 0) == 1:
            lookup.setdefault(normalize_team(place), row.team_id)
    return lookup


def load_manual_odds(
    path: Path | None = None,
    season: int | None = None,
    week: int | None = None,
    league: config.League = config.CFB,
) -> list[dict]:
    """Read the league's odds_manual.csv. Empty fields come back as None."""
    path = path or league.odds_manual_path
    if not Path(path).exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if not (raw.get("away") or "").strip() or not (raw.get("home") or "").strip():
                continue
            try:
                row_season, row_week = int(raw["season"]), int(raw["week"])
            except (KeyError, TypeError, ValueError):
                continue
            if (season is not None and row_season != season) or (
                week is not None and row_week != week
            ):
                continue
            rows.append(
                {
                    "season": row_season,
                    "week": row_week,
                    "away": raw["away"].strip(),
                    "home": raw["home"].strip(),
                    "ml_away": parse_american(raw.get("ml_away")),
                    "ml_home": parse_american(raw.get("ml_home")),
                    "spread_away": parse_line(raw.get("spread_away")),
                    "spread_away_odds": parse_american(raw.get("spread_away_odds")),
                    "spread_home_odds": parse_american(raw.get("spread_home_odds")),
                    "total": parse_line(raw.get("total")),
                    "over_odds": parse_american(raw.get("over_odds")),
                    "under_odds": parse_american(raw.get("under_odds")),
                    "book": (raw.get("book") or "").strip() or DEFAULT_BOOK,
                    "captured_at": (raw.get("captured_at") or "").strip(),
                }
            )
    return rows


def match_manual(
    rows: list[dict], week_games: pd.DataFrame, teams: pd.DataFrame
) -> tuple[dict[str, dict], list[str]]:
    """Match each manual row to a game this week. Returns ({game_id: row}, warnings)."""
    lookup = team_lookup(teams)
    by_pair = {(g.away_id, g.home_id): g.game_id for g in week_games.itertuples()}
    matched: dict[str, dict] = {}
    warnings: list[str] = []
    for row in rows:
        label = f"{row['away']} @ {row['home']}"
        away_id = lookup.get(normalize_team(row["away"]))
        home_id = lookup.get(normalize_team(row["home"]))
        if away_id is None or home_id is None:
            missing = row["away"] if away_id is None else row["home"]
            warnings.append(f"odds_manual: unknown team '{missing}' ({label}); row ignored")
            continue
        game_id = by_pair.get((away_id, home_id))
        if game_id is None:
            if (home_id, away_id) in by_pair:
                warnings.append(
                    f"odds_manual: {label} exists but with home and away swapped; row ignored"
                )
            else:
                warnings.append(f"odds_manual: {label} is not on this week's schedule; row ignored")
            continue
        matched[game_id] = row
    return matched, warnings


# ------------------------------------------------------------------- markets
def _quote(odds: int | None, source: str) -> dict | None:
    return None if odds is None else {"odds": int(odds), "source": source}


def _fair_pair(
    a: dict | None, b: dict | None, ref: tuple[int | None, int | None] | None
) -> tuple[float | None, float | None, dict | None]:
    """No-vig probabilities for the two sides of a market.

    a, b: the quotes a bet would be placed at. ref: ESPN's two-sided quote for the
    same line (or None). Returns (p_a, p_b, how the vig was removed)."""
    real_a = a is not None and a["source"] != "assumed"
    real_b = b is not None and b["source"] != "assumed"
    if real_a and real_b and a["source"] == b["source"]:
        pa, pb = american_to_prob(a["odds"]), american_to_prob(b["odds"])
        return pa / (pa + pb), pb / (pa + pb), {"method": "two_sided", "source": a["source"]}
    if ref is not None and ref[0] is not None and ref[1] is not None:
        pa, pb = american_to_prob(ref[0]), american_to_prob(ref[1])
        return pa / (pa + pb), pb / (pa + pb), {"method": "two_sided", "source": "espn"}
    if real_a or real_b:
        known, is_a = (a, True) if real_a else (b, False)
        p = min(american_to_prob(known["odds"]) / DEFAULT_OVERROUND, 0.999)
        how = {"method": "one_sided", "source": known["source"]}
        return (p, 1 - p, how) if is_a else (1 - p, p, how)
    if a is not None and b is not None:  # both assumed at -110
        return 0.5, 0.5, {"method": "assumed", "source": None}
    return None, None, None


def build_markets(manual: dict | None, espn: dict | None) -> dict:
    """Merge your lines with ESPN's for one game.

    Internal convention: home_line is the HOME team's spread (negative = home favorite).
    """
    manual = manual or {}
    espn = espn or {}
    book = manual.get("book") or DEFAULT_BOOK

    # ---------------- moneyline
    ml_home = _quote(manual.get("ml_home"), book) or _quote(espn.get("ml_home"), "espn")
    ml_away = _quote(manual.get("ml_away"), book) or _quote(espn.get("ml_away"), "espn")
    p_home, p_away, how = _fair_pair(ml_home, ml_away, (espn.get("ml_home"), espn.get("ml_away")))
    ml = {"home": ml_home, "away": ml_away, "p_home": p_home, "p_away": p_away, "devig": how}

    # ---------------- spread
    espn_line = espn.get("spread_home")
    if manual.get("spread_away") is not None:
        line, line_src = -float(manual["spread_away"]), book
    elif espn_line is not None:
        line, line_src = float(espn_line), "espn"
    else:
        line, line_src = None, None
    spread = {
        "home_line": line,
        "line_source": line_src,
        "home": None,
        "away": None,
        "p_home": None,
        "p_away": None,
        "devig": None,
    }
    if line is not None:
        same = espn_line is not None and abs(float(espn_line) - line) < 1e-9
        sp_home = _quote(manual.get("spread_home_odds"), book) if line_src == book else None
        sp_away = _quote(manual.get("spread_away_odds"), book) if line_src == book else None
        if same:
            sp_home = sp_home or _quote(espn.get("spread_home_odds"), "espn")
            sp_away = sp_away or _quote(espn.get("spread_away_odds"), "espn")
        sp_home = sp_home or _quote(STANDARD_PRICE, "assumed")
        sp_away = sp_away or _quote(STANDARD_PRICE, "assumed")
        ref = (espn.get("spread_home_odds"), espn.get("spread_away_odds")) if same else None
        ph, pa, how = _fair_pair(sp_home, sp_away, ref)
        spread.update({"home": sp_home, "away": sp_away, "p_home": ph, "p_away": pa, "devig": how})

    # ---------------- total
    espn_total = espn.get("total")
    if manual.get("total") is not None:
        tline, tsrc = float(manual["total"]), book
    elif espn_total is not None:
        tline, tsrc = float(espn_total), "espn"
    else:
        tline, tsrc = None, None
    total = {
        "line": tline,
        "line_source": tsrc,
        "over": None,
        "under": None,
        "p_over": None,
        "p_under": None,
        "devig": None,
    }
    if tline is not None:
        same = espn_total is not None and abs(float(espn_total) - tline) < 1e-9
        over = _quote(manual.get("over_odds"), book) if tsrc == book else None
        under = _quote(manual.get("under_odds"), book) if tsrc == book else None
        if same:
            over = over or _quote(espn.get("over_odds"), "espn")
            under = under or _quote(espn.get("under_odds"), "espn")
        over = over or _quote(STANDARD_PRICE, "assumed")
        under = under or _quote(STANDARD_PRICE, "assumed")
        ref = (espn.get("over_odds"), espn.get("under_odds")) if same else None
        po, pu, how = _fair_pair(over, under, ref)
        total.update({"over": over, "under": under, "p_over": po, "p_under": pu, "devig": how})

    return {
        "ml": ml,
        "spread": spread,
        "total": total,
        "has_manual": bool(manual),
        "espn": espn or None,
    }


def compare_sources(
    manual: dict | None, espn: dict | None, home: str = "Home", away: str = "Away"
) -> list[str]:
    """Notable differences between your book and ESPN's provider (to catch typos and shop prices)."""
    if not manual or not espn:
        return []
    book = manual.get("book") or DEFAULT_BOOK
    provider = espn.get("provider") or "ESPN"
    notes = []
    for key, team in (("ml_home", home), ("ml_away", away)):
        mine, theirs = manual.get(key), espn.get(key)
        if mine is None or theirs is None:
            continue
        if (mine < 0) != (theirs < 0) and min(abs(mine), abs(theirs)) > 125:
            notes.append(
                f"{team} moneyline: {book} has {mine:+d} but {provider} has {theirs:+d}."
                " They disagree on the favorite, so check the entry."
            )
        elif abs(american_to_prob(mine) - american_to_prob(theirs)) >= 0.02:
            better = provider if american_to_prob(theirs) < american_to_prob(mine) else book
            notes.append(
                f"{team} moneyline: {book} {mine:+d} vs {provider} {theirs:+d} ({better} pays better)"
            )
    if manual.get("spread_away") is not None and espn.get("spread_home") is not None:
        mine, theirs = -float(manual["spread_away"]), float(espn["spread_home"])
        if abs(mine - theirs) >= 1.0:
            notes.append(f"Spread: {book} has {home} {mine:+g}, {provider} has {home} {theirs:+g}")
    if (
        manual.get("total") is not None
        and espn.get("total") is not None
        and abs(manual["total"] - espn["total"]) >= 1.0
    ):
        notes.append(f"Total: {book} {manual['total']:g} vs {provider} {espn['total']:g}")
    return notes
