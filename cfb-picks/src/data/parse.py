"""Parsers puros del JSON de ESPN. Sin red ni disco: fáciles de testear.

Regla general: si un dato no viene o no se entiende, se devuelve None.
Nunca se rellena con un valor inventado.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any


def dig(obj: Any, *path: Any, default: Any = None) -> Any:
    """Acceso seguro a JSON anidado: dig(d, "a", 0, "b")."""
    for key in path:
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, list) and isinstance(key, int) and -len(obj) <= key < len(obj):
            obj = obj[key]
        else:
            return default
        if obj is None:
            return default
    return obj


# ------------------------------------------------------------------ números
_EMPTY = {"", "-", "--", "OFF", "N/A", "NA", "NONE", "NULL", "NAN"}


def parse_american(value: Any) -> int | None:
    """Momio americano -> int. "+185" -> 185, "-110" -> -110, "EVEN" -> 100.

    Devuelve None si falta o no es un momio válido (|momio| < 100 no existe).
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
    else:
        text = str(value).strip().upper().replace("−", "-").replace("–", "-")
        if text in _EMPTY:
            return None
        if text in {"EVEN", "EV", "PK"}:
            return 100
        try:
            num = float(text.replace("+", ""))
        except ValueError:
            return None
    if math.isnan(num) or abs(num) < 100:
        return None
    return round(num)


def parse_line(value: Any) -> float | None:
    """Línea de puntos -> float. "+5.5" -> 5.5, "o46.5" -> 46.5, "PK" -> 0.0."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if math.isnan(float(value)) else float(value)
    text = str(value).strip().upper().replace("−", "-").replace("–", "-")
    if text in _EMPTY:
        return None
    if text in {"PK", "PICK", "EVEN", "EV"}:
        return 0.0
    match = re.fullmatch(r"[OU]?\s*([+-]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


# -------------------------------------------------------------------- juegos
def parse_status(event: dict) -> str:
    """final | scheduled | in_progress | postponed | canceled | forfeit | unknown."""
    st = dig(event, "status", "type", default={}) or {}
    name = str(st.get("name") or "")
    if "CANCEL" in name:
        return "canceled"
    if "POSTPONED" in name:
        return "postponed"
    if "FORFEIT" in name:
        return "forfeit"
    state = st.get("state")
    if st.get("completed") and state == "post":
        return "final"
    if state == "in":
        return "in_progress"
    if state == "pre":
        return "scheduled"
    return "unknown"


def _team_id(competitor: dict) -> str | None:
    tid = str(dig(competitor, "team", "id", default="") or "")
    return tid if tid.isdigit() and int(tid) > 0 else None


def _score(competitor: dict) -> int | None:
    raw = competitor.get("score")
    if isinstance(raw, dict):  # el endpoint de schedule lo trae como {"value": 24.0}
        raw = raw.get("value")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _rank(competitor: dict) -> int | None:
    rank = dig(competitor, "curatedRank", "current")
    return int(rank) if isinstance(rank, (int, float)) and 0 < rank <= 25 else None


def parse_team(competitor: dict) -> dict | None:
    team = competitor.get("team") or {}
    tid = _team_id(competitor)
    if tid is None:
        return None
    name = team.get("location") or team.get("shortDisplayName") or team.get("displayName")
    if not name or str(name).upper() == "TBD":
        return None
    return {
        "team_id": tid,
        "team": str(name),
        "abbr": str(team.get("abbreviation") or name),
        "display_name": str(team.get("displayName") or name),
        "color": str(team.get("color") or ""),
        "alt_color": str(team.get("alternateColor") or ""),
        "conference_id": str(team.get("conferenceId") or ""),
    }


def parse_event(event: dict) -> dict | None:
    """Un evento del scoreboard -> fila plana. None si el juego no es utilizable
    (falta un equipo, equipo por definir, estructura inesperada)."""
    comp = dig(event, "competitions", 0)
    if not isinstance(comp, dict):
        return None
    sides: dict[str, dict] = {}
    for competitor in comp.get("competitors") or []:
        side = competitor.get("homeAway")
        team = parse_team(competitor)
        if side in ("home", "away") and team is not None:
            sides[side] = {**team, "points": _score(competitor), "rank": _rank(competitor)}
    if set(sides) != {"home", "away"}:
        return None
    try:
        date = datetime.fromisoformat(str(event.get("date")).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None

    status = parse_status(event)
    home, away = sides["home"], sides["away"]
    hp, ap = home["points"], away["points"]
    # Un final sin marcador (o 0-0, imposible con tiempo extra) no sirve para entrenar.
    if status == "final" and (hp is None or ap is None or (hp == 0 and ap == 0)):
        status = "unknown"
    completed = status == "final"

    return {
        "game_id": str(event.get("id")),
        "season": dig(event, "season", "year"),
        "season_type": dig(event, "season", "type"),
        "week": dig(event, "week", "number"),
        "date": date,
        "status": status,
        "completed": completed,
        "neutral_site": bool(comp.get("neutralSite")),
        "conference_game": bool(comp.get("conferenceCompetition")),
        "home_id": home["team_id"],
        "home_team": home["team"],
        "home_abbr": home["abbr"],
        "home_points": hp if completed else None,
        "home_rank": home["rank"],
        "away_id": away["team_id"],
        "away_team": away["team"],
        "away_abbr": away["abbr"],
        "away_points": ap if completed else None,
        "away_rank": away["rank"],
    }


def parse_event_teams(event: dict) -> list[dict]:
    out = []
    for competitor in dig(event, "competitions", 0, "competitors", default=[]) or []:
        team = parse_team(competitor)
        if team is not None:
            out.append(team)
    return out


def parse_calendar(payload: dict) -> dict:
    """Calendario de la temporada: semanas regulares y de postemporada con sus fechas."""
    out: dict[str, list[dict]] = {"regular": [], "postseason": []}
    for block in dig(payload, "leagues", 0, "calendar", default=[]) or []:
        if not isinstance(block, dict):
            continue
        kind = {"2": "regular", "3": "postseason"}.get(str(block.get("value")))
        if kind is None:
            continue
        for entry in block.get("entries") or []:
            try:
                out[kind].append(
                    {
                        "week": int(entry["value"]),
                        "label": str(entry.get("label") or ""),
                        "start": datetime.fromisoformat(entry["startDate"].replace("Z", "+00:00")),
                        "end": datetime.fromisoformat(entry["endDate"].replace("Z", "+00:00")),
                    }
                )
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
    return out


def parse_standings_fbs(payload: dict) -> dict[str, str]:
    """Standings FBS -> {team_id: conferencia}. Recorre conferencias y divisiones."""
    teams: dict[str, str] = {}

    def walk(node: dict, conference: str | None) -> None:
        for entry in dig(node, "standings", "entries", default=[]) or []:
            tid = str(dig(entry, "team", "id", default="") or "")
            if tid.isdigit():
                teams[tid] = conference or str(node.get("name") or "")
        for child in node.get("children") or []:
            walk(child, conference or str(child.get("name") or child.get("abbreviation") or ""))

    for child in payload.get("children") or []:
        walk(child, str(child.get("name") or child.get("abbreviation") or ""))
    return teams


# ---------------------------------------------------------------------- odds
_PROVIDER_PREFERENCE = ("draftkings", "espn bet", "caesars", "consensus")


def _spread_from_details(
    details: Any, home_abbr: str | None, away_abbr: str | None
) -> float | None:
    """ "BYU -5.5" -> línea del local, usando qué equipo aparece como favorito."""
    if not details or not home_abbr or not away_abbr:
        return None
    text = str(details).strip().upper()
    if text in {"EVEN", "PK", "PICK"}:
        return 0.0
    match = re.fullmatch(r"(.+?)\s+([+-]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    abbr, num = match.group(1).strip(), float(match.group(2))
    if abbr == home_abbr.upper():
        return num
    if abbr == away_abbr.upper():
        return -num
    return None


def parse_odds_entry(entry: dict) -> dict:
    """Normaliza un bloque de odds de ESPN (scoreboard `odds[i]` o summary `pickcenter[i]`).

    Convención: `spread_home` es la línea del LOCAL (negativa = local favorito).
    Los campos que no vienen quedan en None.
    """
    home_abbr = dig(entry, "homeTeamOdds", "team", "abbreviation")
    away_abbr = dig(entry, "awayTeamOdds", "team", "abbreviation")

    def spread(which: str) -> float | None:
        home = parse_line(dig(entry, "pointSpread", "home", which, "line"))
        away = parse_line(dig(entry, "pointSpread", "away", which, "line"))
        if home is None and away is not None:
            home = -away
        return home

    spread_home = spread("close")
    if spread_home is None:
        spread_home = _spread_from_details(entry.get("details"), home_abbr, away_abbr)
    if spread_home is None:
        spread_home = parse_line(entry.get("spread"))  # ESPN lo publica como línea del local

    total = parse_line(dig(entry, "total", "over", "close", "line"))
    if total is None:
        total = parse_line(entry.get("overUnder"))

    def price(*paths: tuple) -> int | None:
        for path in paths:
            val = parse_american(dig(entry, *path))
            if val is not None:
                return val
        return None

    return {
        "provider": dig(entry, "provider", "name"),
        "spread_home": spread_home,
        "spread_home_odds": price(
            ("pointSpread", "home", "close", "odds"), ("homeTeamOdds", "spreadOdds")
        ),
        "spread_away_odds": price(
            ("pointSpread", "away", "close", "odds"), ("awayTeamOdds", "spreadOdds")
        ),
        "total": total,
        "over_odds": price(("total", "over", "close", "odds"), ("overOdds",)),
        "under_odds": price(("total", "under", "close", "odds"), ("underOdds",)),
        "ml_home": price(("moneyline", "home", "close", "odds"), ("homeTeamOdds", "moneyLine")),
        "ml_away": price(("moneyline", "away", "close", "odds"), ("awayTeamOdds", "moneyLine")),
        "spread_home_open": spread("open"),
        "total_open": parse_line(dig(entry, "total", "over", "open", "line")),
        "ml_home_open": parse_american(dig(entry, "moneyline", "home", "open", "odds")),
        "ml_away_open": parse_american(dig(entry, "moneyline", "away", "open", "odds")),
    }


def has_any_line(odds: dict | None) -> bool:
    return bool(odds) and any(
        odds.get(k) is not None for k in ("spread_home", "total", "ml_home", "ml_away")
    )


def parse_odds(entries: Any) -> dict | None:
    """Elige el mejor proveedor de una lista de odds y lo normaliza. None si no hay líneas."""
    if not isinstance(entries, list):
        return None
    parsed = [parse_odds_entry(e) for e in entries if isinstance(e, dict)]
    parsed = [p for p in parsed if has_any_line(p)]
    if not parsed:
        return None

    def rank(p: dict) -> tuple:
        name = str(p.get("provider") or "").lower()
        pref = next(
            (i for i, key in enumerate(_PROVIDER_PREFERENCE) if key in name),
            len(_PROVIDER_PREFERENCE),
        )
        filled = sum(p.get(k) is not None for k in ("spread_home", "total", "ml_home", "ml_away"))
        return (pref, -filled)

    return min(parsed, key=rank)


def parse_event_odds(event: dict) -> dict | None:
    """Odds de un evento del scoreboard (`competitions[0].odds`)."""
    return parse_odds(dig(event, "competitions", 0, "odds"))
