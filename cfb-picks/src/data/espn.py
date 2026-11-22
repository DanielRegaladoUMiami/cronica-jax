"""Minimal client for ESPN's public API, with an on-disk cache.

Cache policy:
- A week that has already been played (all games final and more than 2 days
  old) is never requested again.
- The current week is refreshed if the copy is older than `max_age` seconds.
- If the API fails, `EspnError` is raised. Made-up data is never returned.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from .. import config


class EspnError(RuntimeError):
    """The API did not respond, or responded with something unusable."""


_UNSETTLED_OK = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_FORFEIT"}


def scoreboard_is_settled(payload: dict, now: datetime | None = None) -> bool:
    """True if the week will no longer change (it can be cached forever)."""
    events = payload.get("events") or []
    if not events:
        return False
    now = now or datetime.now(timezone.utc)
    latest = None
    for ev in events:
        st = (ev.get("status") or {}).get("type") or {}
        if not (st.get("completed") or st.get("name") in _UNSETTLED_OK):
            return False
        try:
            d = datetime.fromisoformat(str(ev.get("date")).replace("Z", "+00:00"))
        except ValueError:
            return False
        latest = d if latest is None or d > latest else latest
    return latest is not None and latest < now - timedelta(days=2)


def _strip_links(obj: Any) -> Any:
    """Strip links/logos from an odds block to keep the cache small."""
    drop = {"link", "links", "logo", "logos", "footer", "header"}
    if isinstance(obj, dict):
        return {k: _strip_links(v) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip_links(x) for x in obj]
    return obj


class EspnClient:
    def __init__(
        self,
        cache_dir: Path | str = config.CACHE_DIR,
        pause: float = 0.1,
        timeout: float = 30.0,
        retries: int = 4,
        offline: bool = False,
        league: config.League = config.CFB,
    ) -> None:
        self.league = league
        self.cache_dir = Path(cache_dir)
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self.stats = {"network": 0, "cache": 0}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "cfb-picks/1.0", "Accept": "application/json"})
        self._lock = threading.Lock()
        self._last_call = 0.0

    # -------------------------------------------------------------- network
    def _throttle(self) -> None:
        with self._lock:
            wait = self.pause - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def fetch_json(self, url: str, params: dict | None = None) -> dict:
        if self.offline:
            raise EspnError(f"offline mode and no cache for {url} {params}")
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                # A 403 from the proxy is a network policy, not a transient failure.
                if "403" in str(exc):
                    raise EspnError(f"the network blocks {url}: {exc}") from exc
                last_err = exc
            except requests.RequestException as exc:
                last_err = exc
            else:
                if resp.status_code == 200:
                    try:
                        self.stats["network"] += 1
                        return resp.json()
                    except ValueError as exc:
                        last_err = exc
                elif resp.status_code in (429, 500, 502, 503, 504):
                    last_err = EspnError(f"HTTP {resp.status_code}")
                else:
                    raise EspnError(f"HTTP {resp.status_code} from {resp.url}")
            time.sleep(1.5 * (2**attempt))
        raise EspnError(f"no response from {url} {params}: {last_err}")

    # ---------------------------------------------------------------- cache
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json.gz"

    def read_cache(self, key: str) -> tuple[Any, float] | tuple[None, None]:
        path = self._path(key)
        if not path.exists():
            return None, None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError, EOFError):
            return None, None
        return payload, time.time() - path.stat().st_mtime

    def write_cache(self, key: str, payload: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        tmp.replace(path)

    def get(
        self,
        key: str,
        url: str,
        params: dict | None = None,
        *,
        max_age: float | None = None,
        settled: Callable[[Any], bool] | None = None,
        refresh: bool = False,
        transform: Callable[[dict], Any] | None = None,
    ) -> Any:
        """Return the cached response if it is still fresh; otherwise request it.

        max_age=None means "forever". A true `settled(payload)` also makes it
        permanent. `transform` trims the response before it is saved.
        """
        cached, age = self.read_cache(key)
        if cached is not None and not refresh:
            fresh = max_age is None or age <= max_age or (settled is not None and settled(cached))
            if fresh or self.offline:
                self.stats["cache"] += 1
                return cached
        try:
            payload = self.fetch_json(url, params)
        except EspnError:
            if cached is not None and self.offline:
                return cached
            raise
        if transform is not None:
            payload = transform(payload)
        self.write_cache(key, payload)
        return payload

    # ------------------------------------------------------------ endpoints
    def scoreboard_week(
        self,
        season: int,
        week: int,
        seasontype: int = 2,
        *,
        max_age: float = 900,
        refresh: bool = False,
    ) -> dict:
        """All the league's games in a week (seasontype 2 = regular season, 3 = postseason)."""
        league = self.league
        params = {
            "dates": season,
            "seasontype": seasontype,
            "week": week,
            **league.scoreboard_params,
            "limit": league.scoreboard_limit,
        }
        return self.get(
            f"{league.cache_prefix}scoreboard/{season}_t{seasontype}_w{week:02d}",
            f"{league.site_url}/scoreboard",
            params,
            max_age=max_age,
            settled=scoreboard_is_settled,
            refresh=refresh,
        )

    def scoreboard_date(
        self, yyyymmdd: str, *, max_age: float = 900, refresh: bool = False
    ) -> dict:
        """The league's games on a specific day (YYYYMMDD)."""
        league = self.league
        params = {"dates": yyyymmdd, **league.scoreboard_params, "limit": league.scoreboard_limit}
        return self.get(
            f"{league.cache_prefix}scoreboard_day/{yyyymmdd}",
            f"{league.site_url}/scoreboard",
            params,
            max_age=max_age,
            settled=scoreboard_is_settled,
            refresh=refresh,
        )

    def standings(self, season: int, *, refresh: bool = False) -> dict:
        """Standings for the season. For college football: the source for which teams are FBS."""
        league = self.league
        past = season < config.current_season()
        return self.get(
            f"{league.cache_prefix}standings/{season}",
            league.standings_url,
            {"season": season, **league.standings_params},
            max_age=None if past else 7 * 86400,
            refresh=refresh,
        )

    def pickcenter(self, event_id: str, *, refresh: bool = False) -> list:
        """Lines (opening and closing) for a game. Only the `pickcenter` block is stored."""
        payload = self.get(
            f"{self.league.cache_prefix}pickcenter/{event_id}",
            f"{self.league.site_url}/summary",
            {"event": event_id},
            max_age=None,
            refresh=refresh,
            transform=lambda p: {"pickcenter": _strip_links(p.get("pickcenter") or [])},
        )
        return payload.get("pickcenter") or []
