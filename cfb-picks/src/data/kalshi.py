"""Public market data from Kalshi (no key needed), and its fee.

A Kalshi market is a yes/no contract that pays $1.00. Buying "Yes" at 41 cents
wins 59 cents if it happens; buying "No" on "team X wins" wins if X draws or
loses. Prices are therefore probabilities: a contract is worth buying only if
the chance of it paying is higher than its price plus the fee.

The fee is 7% x price x (1 - price) per contract (about 1.7 cents at 50 cents,
almost nothing near 0 or 100), rounded up to the next cent on the whole order.
"""

from __future__ import annotations

import time
from typing import Any

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
TAKER_RATE = 0.07
MAKER_RATE = 0.0175  # only on series whose fee type is "quadratic_with_maker_fees"


class KalshiError(RuntimeError):
    """Kalshi did not respond, or responded with something unusable."""


def taker_fee(price: float) -> float:
    """Fee per contract when you take the price on the screen."""
    return TAKER_RATE * price * (1.0 - price)


def maker_fee(price: float, fee_type: str | None) -> float:
    """Fee per contract when your resting order gets filled."""
    if fee_type == "quadratic_with_maker_fees":
        return MAKER_RATE * price * (1.0 - price)
    return 0.0


def _price(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if 0.0 <= out <= 1.0 else None


def parse_market(market: dict) -> dict:
    """The fields that matter from one Kalshi market, with prices in dollars (0 to 1)."""
    yes_bid, yes_ask = _price(market.get("yes_bid_dollars")), _price(market.get("yes_ask_dollars"))
    no_bid, no_ask = _price(market.get("no_bid_dollars")), _price(market.get("no_ask_dollars"))
    # An empty book shows 0.00 bid and 1.00 ask: that is not a price.
    quoted = (
        yes_bid is not None
        and yes_ask is not None
        and 0.0 < yes_ask < 1.0
        and 0.0 < yes_bid < 1.0
        and yes_ask >= yes_bid
    )
    try:
        volume = float(market.get("volume_fp") or 0.0)
    except (TypeError, ValueError):
        volume = 0.0
    strike = market.get("floor_strike")
    return {
        "ticker": market.get("ticker"),
        "title": market.get("title"),
        "label": market.get("yes_sub_title"),
        "status": market.get("status"),
        "strike": float(strike) if isinstance(strike, (int, float)) else None,
        "yes_bid": yes_bid if quoted else None,
        "yes_ask": yes_ask if quoted else None,
        "no_bid": no_bid if quoted else None,
        "no_ask": no_ask if quoted else None,
        "volume": volume,
        "quoted": quoted,
    }


class KalshiClient:
    def __init__(self, pause: float = 0.25, timeout: float = 30.0, retries: int = 4) -> None:
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        self.requests = 0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "cfb-picks/1.0", "Accept": "application/json"})
        self._last = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            wait = self.pause - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self._session.get(f"{API}{path}", params=params, timeout=self.timeout)
            except requests.exceptions.ProxyError as exc:
                if "403" in str(exc):
                    raise KalshiError(f"the network blocks Kalshi: {exc}") from exc
                last_err = exc
            except requests.RequestException as exc:
                last_err = exc
            else:
                if resp.status_code == 200:
                    self.requests += 1
                    try:
                        return resp.json()
                    except ValueError as exc:
                        last_err = exc
                elif resp.status_code in (429, 500, 502, 503, 504):
                    last_err = KalshiError(f"HTTP {resp.status_code}")
                else:
                    raise KalshiError(f"HTTP {resp.status_code} from {resp.url}")
            time.sleep(1.0 * (2**attempt))
        raise KalshiError(f"no response from Kalshi {path} {params}: {last_err}")

    def events(self, series_ticker: str, status: str = "open") -> list[dict]:
        """Every event of a series with its markets (a game is one event)."""
        out: list[dict] = []
        cursor = None
        for _ in range(50):
            params = {
                "series_ticker": series_ticker,
                "status": status,
                "with_nested_markets": "true",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            payload = self._get("/events", params)
            out.extend(payload.get("events") or [])
            cursor = payload.get("cursor")
            if not cursor:
                break
        return out

    def series(self, series_ticker: str) -> dict:
        return self._get(f"/series/{series_ticker}").get("series") or {}
