"""Cliente mínimo para el API público de ESPN, con caché en disco.

Política de caché:
- Una semana ya jugada (todos los juegos finales y con más de 2 días) no se
  vuelve a pedir nunca.
- La semana en curso se refresca si la copia tiene más de `max_age` segundos.
- Si el API falla, se lanza `EspnError`. Nunca se devuelven datos inventados.
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
    """El API no respondió o respondió algo que no se puede usar."""


_UNSETTLED_OK = {"STATUS_CANCELED", "STATUS_POSTPONED", "STATUS_FORFEIT"}


def scoreboard_is_settled(payload: dict, now: datetime | None = None) -> bool:
    """True si la semana ya no va a cambiar (se puede cachear para siempre)."""
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
    """Quita enlaces/logos de un bloque de odds para que la caché pese poco."""
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
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        self.offline = offline
        self.stats = {"red": 0, "cache": 0}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "cfb-picks/1.0", "Accept": "application/json"})
        self._lock = threading.Lock()
        self._last_call = 0.0

    # ------------------------------------------------------------------ red
    def _throttle(self) -> None:
        with self._lock:
            wait = self.pause - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def fetch_json(self, url: str, params: dict | None = None) -> dict:
        if self.offline:
            raise EspnError(f"modo offline y sin caché para {url} {params}")
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                # Un 403 del proxy es una política de red, no un fallo transitorio.
                if "403" in str(exc):
                    raise EspnError(f"la red bloquea {url}: {exc}") from exc
                last_err = exc
            except requests.RequestException as exc:
                last_err = exc
            else:
                if resp.status_code == 200:
                    try:
                        self.stats["red"] += 1
                        return resp.json()
                    except ValueError as exc:
                        last_err = exc
                elif resp.status_code in (429, 500, 502, 503, 504):
                    last_err = EspnError(f"HTTP {resp.status_code}")
                else:
                    raise EspnError(f"HTTP {resp.status_code} en {resp.url}")
            time.sleep(1.5 * (2**attempt))
        raise EspnError(f"sin respuesta de {url} {params}: {last_err}")

    # ---------------------------------------------------------------- caché
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
        """Devuelve la respuesta desde caché si sigue vigente; si no, la pide.

        max_age=None significa "para siempre". `settled(payload)` verdadero
        también la hace permanente. `transform` reduce la respuesta antes de
        guardarla.
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
        """Todos los juegos FBS de una semana (seasontype 2 = regular, 3 = postemporada)."""
        params = {
            "dates": season,
            "seasontype": seasontype,
            "week": week,
            "groups": config.FBS_GROUP,
            "limit": config.SCOREBOARD_LIMIT,
        }
        return self.get(
            f"scoreboard/{season}_t{seasontype}_w{week:02d}",
            f"{config.ESPN_SITE}/scoreboard",
            params,
            max_age=max_age,
            settled=scoreboard_is_settled,
            refresh=refresh,
        )

    def scoreboard_date(
        self, yyyymmdd: str, *, max_age: float = 900, refresh: bool = False
    ) -> dict:
        """Juegos FBS de un día concreto (YYYYMMDD)."""
        params = {"dates": yyyymmdd, "groups": config.FBS_GROUP, "limit": config.SCOREBOARD_LIMIT}
        return self.get(
            f"scoreboard_dia/{yyyymmdd}",
            f"{config.ESPN_SITE}/scoreboard",
            params,
            max_age=max_age,
            settled=scoreboard_is_settled,
            refresh=refresh,
        )

    def standings(self, season: int, *, refresh: bool = False) -> dict:
        """Standings FBS de la temporada: de aquí sale qué equipos son FBS ese año."""
        past = season < config.current_season()
        return self.get(
            f"standings/{season}",
            config.ESPN_STANDINGS,
            {"season": season, "group": config.FBS_GROUP},
            max_age=None if past else 7 * 86400,
            refresh=refresh,
        )

    def pickcenter(self, event_id: str, *, refresh: bool = False) -> list:
        """Líneas (apertura y cierre) de un juego. Solo se guarda el bloque `pickcenter`."""
        payload = self.get(
            f"pickcenter/{event_id}",
            f"{config.ESPN_SITE}/summary",
            {"event": event_id},
            max_age=None,
            refresh=refresh,
            transform=lambda p: {"pickcenter": _strip_links(p.get("pickcenter") or [])},
        )
        return payload.get("pickcenter") or []
