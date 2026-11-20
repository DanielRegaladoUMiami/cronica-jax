"""Rutas y constantes del proyecto."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"

GAMES_PATH = DATA_DIR / "games.parquet"
TEAMS_PATH = DATA_DIR / "teams.parquet"
ODDS_HISTORY_PATH = DATA_DIR / "odds_history.parquet"
ODDS_MANUAL_PATH = DATA_DIR / "odds_manual.csv"
NOTES_PATH = DATA_DIR / "notas.csv"
PARAMS_PATH = ROOT / "src" / "model" / "params.json"

# API público de ESPN (sin key).
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
FBS_GROUP = 80
SCOREBOARD_LIMIT = 300

# Temporadas previas: solo se usan para calibrar hiperparámetros y como prior
# de pretemporada. La temporada actual siempre se evalúa fuera de muestra.
HISTORY_SEASONS = [2021, 2022, 2023, 2024, 2025]


def current_season(now: datetime | None = None) -> int:
    """La temporada se nombra por el año en que empieza (los bowls de enero son del año anterior)."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1
