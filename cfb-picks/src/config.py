"""Paths, constants and the leagues the pipeline knows.

Everything that differs between college football and the NFL lives in a
`League`: where ESPN keeps it, how its files are named, how its postseason is
organized. The rest of the code takes a league and stays the same for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"

# ESPN's public API (no key needed).
ESPN_HOST = "https://site.api.espn.com/apis"


@dataclass(frozen=True)
class League:
    key: str  # "cfb" or "nfl": what --league takes and what the site calls the sport
    name: str
    espn_path: str  # the sport's path inside ESPN's API
    prefix: str  # prefix of its data files, cache folder and site JSON ("" = the original names)
    scoreboard_params: dict = field(default_factory=dict)  # extra query parameters
    standings_params: dict = field(default_factory=dict)
    scoreboard_limit: int = 100
    # Who belongs to the league each season. "standings": the teams in ESPN's standings
    # (FBS; everyone else is pooled as FCS). "schedule": every team with a regular-season game.
    membership: str = "standings"
    # True: each postseason round is its own week. False: bowls are one bucket after the last week.
    playoff_rounds: bool = False
    # True: teams go by nickname ("Bills"), because cities repeat (two in New York, two in LA).
    nicknames: bool = False
    min_season_games: int = (
        600  # a finished season this complete in the parquet is not re-downloaded
    )
    big_favorite: float = 14.0  # projected margin that counts as a clear favorite in the backtest
    min_points: float = 3.0  # floor for a team's projected score
    history_seasons: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)
    rules: dict = field(default_factory=dict)  # overrides of model.rules.Rules for this league
    prototype: dict | None = None  # reference numbers of the original prototype, if there was one

    # ------------------------------------------------------------------ ESPN
    @property
    def site_url(self) -> str:
        return f"{ESPN_HOST}/site/v2/sports/{self.espn_path}"

    @property
    def standings_url(self) -> str:
        return f"{ESPN_HOST}/v2/sports/{self.espn_path}/standings"

    @property
    def cache_prefix(self) -> str:
        return f"{self.prefix}/" if self.prefix else ""

    # ----------------------------------------------------------------- files
    def _data(self, name: str) -> Path:
        return DATA_DIR / (f"{self.prefix}_{name}" if self.prefix else name)

    @property
    def games_path(self) -> Path:
        return self._data("games.parquet")

    @property
    def teams_path(self) -> Path:
        return self._data("teams.parquet")

    @property
    def odds_history_path(self) -> Path:
        return self._data("odds_history.parquet")

    @property
    def odds_manual_path(self) -> Path:
        return self._data("odds_manual.csv")

    @property
    def notes_path(self) -> Path:
        return self._data("notes.csv")

    @property
    def parlays_path(self) -> Path:
        return self._data("parlays.csv")

    @property
    def params_path(self) -> Path:
        name = f"{self.prefix}_params.json" if self.prefix else "params.json"
        return ROOT / "src" / "model" / name

    def site_json(self, name: str) -> Path:
        """site/data/picks.json for college football, site/data/nfl_picks.json for the NFL."""
        return SITE_DATA_DIR / (f"{self.prefix}_{name}" if self.prefix else name)


CFB = League(
    key="cfb",
    name="College football",
    espn_path="football/college-football",
    prefix="",
    scoreboard_params={"groups": 80},  # 80 = FBS
    standings_params={"group": 80},
    scoreboard_limit=300,
    prototype={"accuracy": 0.73, "accuracy_big": 0.92, "mae_margin": 15.5, "mae_total": 20.0},
)

NFL = League(
    key="nfl",
    name="NFL",
    espn_path="football/nfl",
    prefix="nfl",
    membership="schedule",
    playoff_rounds=True,
    nicknames=True,
    min_season_games=270,  # 272 regular-season games plus the playoffs
    big_favorite=7.0,
    # NFL lines are tighter than college ones, so the red flags trip earlier. Set on the
    # betting backtest: with these, flagged picks lost about twice what the plays did.
    rules={"gap_margin": 4.0, "gap_total": 4.0, "move_spread": 1.0, "move_total": 3.0},
)

LEAGUES = {league.key: league for league in (CFB, NFL)}


def current_season(now: datetime | None = None) -> int:
    """A season is named after the year it starts in (January games belong to the previous year)."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1
