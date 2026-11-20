"""Odds de la semana: tus líneas (data/odds_manual.csv) + las de ESPN.

Reglas de combinación, por mercado (moneyline, spread, total):
- La LÍNEA y el MOMIO que se usan para apostar son los tuyos si los capturaste;
  si no, los de ESPN (DraftKings).
- La probabilidad del mercado SIN VIG se calcula con una cotización de dos lados
  de una sola fuente: tu casa si capturaste ambos lados; si no, ESPN (misma
  línea); si no hay dos lados, se le quita un vig típico al único lado que hay.
- Un precio que falta nunca se inventa en silencio: si se asume -110 (estándar
  de spread/total) queda marcado como "asumido".
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

# Formas comunes de escribir un equipo -> nombre corto de ESPN (ya normalizados).
ALIASES = {
    "miami fl": "miami",
    "miami florida": "miami",
    "miami ohio": "miami oh",
    "pitt": "pittsburgh",
    "southern california": "usc",
    "southern cal": "usc",
    "hawaii": "hawai i",
    "ole miss": "ole miss",
    "mississippi": "ole miss",
    "nc state": "nc state",
    "north carolina state": "nc state",
    "appalachian state": "app state",
    "connecticut": "uconn",
    "massachusetts": "massachusetts",
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
    "san jose st": "san jose state",
}


def normalize_team(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"\bst\b\.?", "state", text)
    text = re.sub(r"[^a-z0-9&]+", " ", text).strip()
    return ALIASES.get(text, text)


def team_lookup(teams: pd.DataFrame) -> dict[str, str]:
    """nombre normalizado -> team_id. Acepta nombre corto, nombre completo y abreviatura (si es única)."""
    lookup: dict[str, str] = {}
    abbr_count = teams["abbr"].str.lower().value_counts()
    for row in teams.itertuples():
        lookup[normalize_team(row.team)] = row.team_id
        lookup.setdefault(normalize_team(row.display_name), row.team_id)
        if abbr_count.get(str(row.abbr).lower(), 0) == 1:
            lookup.setdefault(normalize_team(row.abbr), row.team_id)
    return lookup


def load_manual_odds(
    path: Path = config.ODDS_MANUAL_PATH, season: int | None = None, week: int | None = None
) -> list[dict]:
    """Lee odds_manual.csv. Los campos vacíos quedan en None."""
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
                    "book": (raw.get("book") or "").strip() or "manual",
                    "captured_at": (raw.get("captured_at") or "").strip(),
                }
            )
    return rows


def match_manual(
    rows: list[dict], week_games: pd.DataFrame, teams: pd.DataFrame
) -> tuple[dict[str, dict], list[str]]:
    """Empareja cada fila manual con un juego de la semana. Devuelve ({game_id: fila}, avisos)."""
    lookup = team_lookup(teams)
    by_pair = {(g.away_id, g.home_id): g.game_id for g in week_games.itertuples()}
    matched: dict[str, dict] = {}
    warnings: list[str] = []
    for row in rows:
        label = f"{row['away']} @ {row['home']}"
        away_id, home_id = (
            lookup.get(normalize_team(row["away"])),
            lookup.get(normalize_team(row["home"])),
        )
        if away_id is None or home_id is None:
            missing = row["away"] if away_id is None else row["home"]
            warnings.append(
                f"odds_manual: no reconozco el equipo '{missing}' ({label}); la fila se ignoró"
            )
            continue
        game_id = by_pair.get((away_id, home_id))
        if game_id is None:
            if (home_id, away_id) in by_pair:
                warnings.append(
                    f"odds_manual: {label} existe pero con local y visitante invertidos; la fila se ignoró"
                )
            else:
                warnings.append(
                    f"odds_manual: {label} no está en los juegos de la semana; la fila se ignoró"
                )
            continue
        matched[game_id] = row
    return matched, warnings


# ------------------------------------------------------------------ mercados
def _quote(odds: int | None, source: str) -> dict | None:
    return None if odds is None else {"momio": int(odds), "fuente": source}


def _fair_pair(
    a: dict | None, b: dict | None, ref: tuple[int | None, int | None] | None
) -> tuple[float | None, float | None, str | None]:
    """Probabilidades sin vig de los dos lados de un mercado.

    a, b: cotizaciones que se usarían para apostar. ref: cotización de dos lados de
    ESPN para la misma línea (o None). Devuelve (p_a, p_b, método)."""
    real_a = a is not None and a["fuente"] != "asumido"
    real_b = b is not None and b["fuente"] != "asumido"
    if real_a and real_b and a["fuente"] == b["fuente"]:
        pa, pb = american_to_prob(a["momio"]), american_to_prob(b["momio"])
        return pa / (pa + pb), pb / (pa + pb), f"dos lados ({a['fuente']})"
    if ref is not None and ref[0] is not None and ref[1] is not None:
        pa, pb = american_to_prob(ref[0]), american_to_prob(ref[1])
        return pa / (pa + pb), pb / (pa + pb), "dos lados (espn)"
    if real_a or real_b:
        known, is_a = (a, True) if real_a else (b, False)
        p = min(american_to_prob(known["momio"]) / DEFAULT_OVERROUND, 0.999)
        return (
            (p, 1 - p, f"un lado ({known['fuente']}), vig estimado")
            if is_a
            else (1 - p, p, f"un lado ({known['fuente']}), vig estimado")
        )
    if a is not None and b is not None:  # ambos asumidos a -110
        return 0.5, 0.5, "precio asumido"
    return None, None, None


def build_markets(manual: dict | None, espn: dict | None) -> dict:
    """Combina tus líneas con las de ESPN para un juego.

    Convención interna: spread_home = línea del local (negativa = local favorito).
    """
    manual = manual or {}
    espn = espn or {}
    book = manual.get("book") or "manual"

    # ---------------- moneyline
    ml_home = _quote(manual.get("ml_home"), book) or _quote(espn.get("ml_home"), "espn")
    ml_away = _quote(manual.get("ml_away"), book) or _quote(espn.get("ml_away"), "espn")
    p_home, p_away, how = _fair_pair(ml_home, ml_away, (espn.get("ml_home"), espn.get("ml_away")))
    ml = {
        "local": ml_home,
        "visita": ml_away,
        "p_local": p_home,
        "p_visita": p_away,
        "sin_vig": how,
    }

    # ---------------- spread
    espn_line = espn.get("spread_home")
    if manual.get("spread_away") is not None:
        line, line_src = -float(manual["spread_away"]), book
    elif espn_line is not None:
        line, line_src = float(espn_line), "espn"
    else:
        line, line_src = None, None
    spread = {
        "linea_local": line,
        "fuente_linea": line_src,
        "local": None,
        "visita": None,
        "p_local": None,
        "p_visita": None,
        "sin_vig": None,
    }
    if line is not None:
        same = espn_line is not None and abs(float(espn_line) - line) < 1e-9
        sp_home = _quote(manual.get("spread_home_odds"), book) if line_src == book else None
        sp_away = _quote(manual.get("spread_away_odds"), book) if line_src == book else None
        if same:
            sp_home = sp_home or _quote(espn.get("spread_home_odds"), "espn")
            sp_away = sp_away or _quote(espn.get("spread_away_odds"), "espn")
        sp_home = sp_home or _quote(STANDARD_PRICE, "asumido")
        sp_away = sp_away or _quote(STANDARD_PRICE, "asumido")
        ref = (espn.get("spread_home_odds"), espn.get("spread_away_odds")) if same else None
        ph, pa, how = _fair_pair(sp_home, sp_away, ref)
        spread.update(
            {"local": sp_home, "visita": sp_away, "p_local": ph, "p_visita": pa, "sin_vig": how}
        )

    # ---------------- total
    espn_total = espn.get("total")
    if manual.get("total") is not None:
        tline, tsrc = float(manual["total"]), book
    elif espn_total is not None:
        tline, tsrc = float(espn_total), "espn"
    else:
        tline, tsrc = None, None
    total = {
        "linea": tline,
        "fuente_linea": tsrc,
        "over": None,
        "under": None,
        "p_over": None,
        "p_under": None,
        "sin_vig": None,
    }
    if tline is not None:
        same = espn_total is not None and abs(float(espn_total) - tline) < 1e-9
        over = _quote(manual.get("over_odds"), book) if tsrc == book else None
        under = _quote(manual.get("under_odds"), book) if tsrc == book else None
        if same:
            over = over or _quote(espn.get("over_odds"), "espn")
            under = under or _quote(espn.get("under_odds"), "espn")
        over = over or _quote(STANDARD_PRICE, "asumido")
        under = under or _quote(STANDARD_PRICE, "asumido")
        ref = (espn.get("over_odds"), espn.get("under_odds")) if same else None
        po, pu, how = _fair_pair(over, under, ref)
        total.update({"over": over, "under": under, "p_over": po, "p_under": pu, "sin_vig": how})

    return {
        "ml": ml,
        "spread": spread,
        "total": total,
        "tiene_manual": bool(manual),
        "espn": espn or None,
    }


def compare_sources(manual: dict | None, espn: dict | None) -> list[str]:
    """Diferencias notables entre tu casa y ESPN/DraftKings (para validar capturas y comparar precios)."""
    if not manual or not espn:
        return []
    notes = []
    for side, label in (("ml_home", "ML local"), ("ml_away", "ML visitante")):
        mine, theirs = manual.get(side), espn.get(side)
        if mine is None or theirs is None:
            continue
        if (mine < 0) != (theirs < 0) and min(abs(mine), abs(theirs)) > 125:
            notes.append(
                f"{label}: tu casa {mine:+d} y DraftKings {theirs:+d} no coinciden en el favorito; revisa la captura"
            )
        elif abs(american_to_prob(mine) - american_to_prob(theirs)) >= 0.02:
            better = (
                "DraftKings paga mejor"
                if american_to_prob(theirs) < american_to_prob(mine)
                else "tu casa paga mejor"
            )
            notes.append(f"{label}: tu casa {mine:+d} vs DraftKings {theirs:+d} ({better})")
    if manual.get("spread_away") is not None and espn.get("spread_home") is not None:
        mine, theirs = -float(manual["spread_away"]), float(espn["spread_home"])
        if abs(mine - theirs) >= 1.0:
            notes.append(f"Spread del local: tu casa {mine:+g} vs DraftKings {theirs:+g}")
    if (
        manual.get("total") is not None
        and espn.get("total") is not None
        and abs(manual["total"] - espn["total"]) >= 1.0
    ):
        notes.append(f"Total: tu casa {manual['total']:g} vs DraftKings {espn['total']:g}")
    return notes
