"""Backtest de apuestas: aplica la regla JUGADA/PASAR/EVITAR semana a semana
contra las líneas de cierre reales y registra qué habría pasado.

Es walk-forward igual que el backtest de predicción: los ratings de la semana W
solo usan semanas < W, y el chequeo de estabilidad se hace con los datos que
había en ese momento.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.odds import build_markets
from .data import SeasonArrays
from .prob import american_to_decimal
from .rules import Rules, evaluate_game

_ODDS_FIELDS = (
    "spread_home",
    "spread_home_odds",
    "spread_away_odds",
    "total",
    "over_odds",
    "under_odds",
    "ml_home",
    "ml_away",
    "spread_home_open",
    "total_open",
    "ml_home_open",
    "ml_away_open",
)


def odds_row_to_dict(row) -> dict:
    """Fila de odds_history.parquet -> dict con None en lo que falta."""
    out = {"provider": getattr(row, "provider", None)}
    for field in _ODDS_FIELDS:
        value = getattr(row, field, None)
        if value is None or pd.isna(value):
            out[field] = None
        else:
            out[field] = (
                float(value)
                if field in ("spread_home", "total", "spread_home_open", "total_open")
                else int(value)
            )
    return out


def drop_last_games(
    sa: SeasonArrays, mask: np.ndarray, teams: tuple[int, ...], n: int = 2
) -> np.ndarray:
    """Copia de `mask` sin los últimos n juegos de cada equipo (los arreglos están en orden cronológico)."""
    out = mask.copy()
    for team in teams:
        played = np.flatnonzero(mask & sa.completed & ((sa.h == team) | (sa.a == team)))
        out[played[-n:]] = False
    return out


def settle(tipo: str, lado: str, linea: float | None, margin: float, total: float) -> int:
    """1 = gana, 0 = push, -1 = pierde."""
    if tipo == "ml":
        value = margin if lado == "local" else -margin
    elif tipo == "spread":
        value = (margin + linea) if lado == "local" else -(margin + linea)
    else:
        value = (total - linea) if lado == "over" else (linea - total)
    return int(np.sign(value))


def simulate_bets(
    model,
    seasons: dict[int, SeasonArrays],
    odds: pd.DataFrame,
    sigma_m: float,
    sigma_t: float,
    rules: Rules,
    eval_seasons,
) -> pd.DataFrame:
    odds_by_id = {row.game_id: odds_row_to_dict(row) for row in odds[odds["has_odds"]].itertuples()}
    eval_seasons = set(eval_seasons)
    rows: list[dict] = []
    prev_state = prev_sa = None
    for season in sorted(seasons):
        sa = seasons[season]
        prior = model.prior(prev_state, prev_sa, sa)
        if season in eval_seasons:
            for week in np.unique(sa.week[sa.completed]):
                train = sa.completed & (sa.week < week)
                test = [
                    g
                    for g in np.flatnonzero(sa.completed & (sa.week == week) & ~sa.fcs)
                    if sa.game_id[g] in odds_by_id
                ]
                if not test:
                    continue
                state = model.fit(sa, train, prior, now_week=int(week))
                for g in test:
                    h, a, neutral = int(sa.h[g]), int(sa.a[g]), bool(sa.neutral[g])
                    m, t = model.predict(state, [h], [a], [neutral])
                    alt = model.fit(
                        sa, drop_last_games(sa, train, (h, a)), prior, now_week=int(week)
                    )
                    m2, t2 = model.predict(alt, [h], [a], [neutral])
                    markets = build_markets(None, odds_by_id[sa.game_id[g]])
                    total_pred = None if math.isnan(float(t[0])) else float(t[0])
                    ev = evaluate_game(
                        margin=float(m[0]),
                        total=total_pred,
                        margin_alt=float(m2[0]),
                        total_alt=None if total_pred is None else float(t2[0]),
                        markets=markets,
                        sigma_m=sigma_m,
                        sigma_t=sigma_t,
                        rules=rules,
                        min_games_played=int(min(state.games[h], state.games[a])),
                    )
                    margin, total = float(sa.hp[g] - sa.ap[g]), float(sa.hp[g] + sa.ap[g])
                    for tipo in ("ml", "spread", "total"):
                        lado = ev[tipo]["pick"]
                        side = ev[tipo].get(lado) if lado else None
                        if not side or side["edge"] is None:
                            continue
                        linea = (
                            markets["spread"]["linea_local"]
                            if tipo == "spread"
                            else markets["total"]["linea"]
                            if tipo == "total"
                            else None
                        )
                        res = settle(tipo, lado, linea, margin, total)
                        profit = (
                            (american_to_decimal(side["momio"]) - 1.0)
                            if res > 0
                            else (0.0 if res == 0 else -1.0)
                        )
                        rows.append(
                            {
                                "season": season,
                                "week": int(week),
                                "game_id": sa.game_id[g],
                                "tipo": tipo,
                                "lado": lado,
                                "linea": linea,
                                "momio": side["momio"],
                                "p_modelo": side["p_modelo"],
                                "p_mercado": side["p_mercado"],
                                "edge": side["edge"],
                                "ev": side["ev"],
                                "estable": side["estable"],
                                "etiqueta": side["etiqueta"],
                                "stake": side["stake"],
                                "resultado": res,
                                "profit": profit,
                                "desacuerdo": ev["desacuerdo_total"]
                                if tipo == "total"
                                else ev["desacuerdo_margen"],
                            }
                        )
        if sa.completed.any():
            prev_state = model.fit(
                sa, np.ones(len(sa.week), dtype=bool), prior, now_week=int(sa.week.max()) + 1
            )
            prev_sa = sa
    return pd.DataFrame(rows)


def record(bets: pd.DataFrame) -> dict:
    """Resumen de un conjunto de apuestas a unidad plana."""
    n = len(bets)
    if n == 0:
        return {
            "n": 0,
            "ganadas": 0,
            "perdidas": 0,
            "push": 0,
            "pct": None,
            "unidades": 0.0,
            "roi": None,
            "roi_error": None,
        }
    won, lost = int((bets["resultado"] > 0).sum()), int((bets["resultado"] < 0).sum())
    profit = bets["profit"].to_numpy(dtype=float)
    return {
        "n": n,
        "ganadas": won,
        "perdidas": lost,
        "push": n - won - lost,
        "pct": won / max(won + lost, 1),
        "unidades": float(profit.sum()),
        "roi": float(profit.mean()),
        "roi_error": float(profit.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
    }
