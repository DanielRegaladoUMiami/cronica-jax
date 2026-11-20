"""Probabilidades: normal, momios americanos, vig, Kelly y parlays.

Todo es aritmética pura (sin scipy) para que sea fácil de testear.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x):
    """CDF de la normal estándar. Acepta escalar o arreglo."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / _SQRT2))
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(arr / _SQRT2))


# ------------------------------------------------------------ del modelo
def p_home_win(margin: float, sigma: float) -> float:
    """P(gana el local) si el margen real ~ Normal(margen proyectado, sigma)."""
    return float(norm_cdf(margin / sigma))


def p_cover(margin: float, spread_home: float, sigma: float) -> float:
    """P(el local cubre). spread_home es la línea del local (-7.5 = favorito por 7.5).
    Cubre si margen_real + spread_home > 0."""
    return float(norm_cdf((margin + spread_home) / sigma))


def p_over(total: float, line: float, sigma: float) -> float:
    return float(norm_cdf((total - line) / sigma))


# ------------------------------------------------------------- momios
def american_to_decimal(odds: float) -> float:
    """Momio americano -> decimal (pago total por unidad apostada). -110 -> 1.909, +150 -> 2.5."""
    odds = float(odds)
    if abs(odds) < 100:
        raise ValueError(f"momio americano inválido: {odds}")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def american_to_prob(odds: float) -> float:
    """Probabilidad implícita CON vig. -110 -> 0.5238, +150 -> 0.40."""
    return 1.0 / american_to_decimal(odds)


def prob_to_american(p: float) -> int | None:
    """Momio justo para una probabilidad (para mostrar 'la línea del modelo')."""
    if not 0.0 < p < 1.0:
        return None
    if p >= 0.5:
        return round(-100.0 * p / (1.0 - p))
    return round(100.0 * (1.0 - p) / p)


DEFAULT_OVERROUND = 1.045  # vig típico de un mercado de dos lados (-110/-110 da 1.0476)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Quita el vig repartiéndolo en proporción (método multiplicativo).
    -110/-110 -> (0.5, 0.5)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def devig_one_side(odds: float, overround: float = DEFAULT_OVERROUND) -> float:
    """Sin el otro lado no se puede medir el vig: se asume un sobre-redondeo."""
    return min(american_to_prob(odds) / overround, 0.999)


def overround(odds_a: float, odds_b: float) -> float:
    return american_to_prob(odds_a) + american_to_prob(odds_b)


# ------------------------------------------------------------ apuestas
def expected_value(p: float, odds: float) -> float:
    """Ganancia esperada por unidad apostada al momio real (con vig)."""
    return p * american_to_decimal(odds) - 1.0


def kelly_fraction(p: float, odds: float) -> float:
    """Fracción de Kelly completa: (b*p - q) / b, con b = decimal - 1. Nunca negativa."""
    b = american_to_decimal(odds) - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def suggested_stake(p: float, odds: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """Kelly fraccional (1/4 por defecto) con tope como fracción del bankroll."""
    return min(fraction * kelly_fraction(p, odds), cap)


def bet_profit(odds: float, won: bool) -> float:
    """Resultado de apostar 1 unidad."""
    return american_to_decimal(odds) - 1.0 if won else -1.0


def parlay(probs: Sequence[float], odds: Iterable[float | None]) -> dict:
    """Parlay de piernas independientes.

    prob        = producto de las probabilidades del modelo
    decimal     = producto de los momios decimales (lo que paga la casa)
    prob_casa   = 1 / decimal (probabilidad implícita del boleto, con vig)
    ev          = ganancia esperada por unidad
    Si falta el momio de alguna pierna, no se calcula el pago.
    """
    prob = float(np.prod(list(probs))) if len(probs) else 0.0
    odds = list(odds)
    if odds and all(o is not None for o in odds):
        decimal = float(np.prod([american_to_decimal(o) for o in odds]))
        return {
            "prob": prob,
            "decimal": decimal,
            "prob_casa": 1.0 / decimal,
            "ev": prob * decimal - 1.0,
        }
    return {"prob": prob, "decimal": None, "prob_casa": None, "ev": None}
