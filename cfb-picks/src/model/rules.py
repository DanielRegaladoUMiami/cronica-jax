"""Reglas de decisión: de una proyección y unas líneas a JUGADA / PASAR / EVITAR.

La misma función se usa para los picks de la semana y para el backtest de
apuestas, así el track record mide exactamente la regla que ves en el sitio.

JUGADA  = edge >= 5%, valor esperado positivo al momio real, y sin alertas.
EVITAR  = hay edge en papel, pero el pick es inestable o el mercado está diciendo
          algo que el modelo no sabe (lesiones, clima, noticias).
PASAR   = no hay edge suficiente.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .prob import expected_value, kelly_fraction, norm_cdf


@dataclass(frozen=True)
class Rules:
    edge_min: float = 0.05  # edge mínimo (probabilidad del modelo - mercado sin vig)
    kelly_fraction: float = 0.25  # 1/4 de Kelly
    stake_cap: float = 0.02  # tope: 2% del bankroll por jugada
    gap_margin: float = (
        5.0  # pts de diferencia modelo vs spread del mercado -> EVITAR (ML y spread)
    )
    gap_total: float = 7.0  # pts de diferencia modelo vs total del mercado -> EVITAR
    move_spread: float = 3.0  # movimiento de la línea (apertura -> ahora) en contra del pick
    move_total: float = 4.0
    min_games: int = 3  # juegos mínimos de cada equipo para confiar en su rating

    def as_dict(self) -> dict:
        return asdict(self)


def norm_ppf(p: float) -> float:
    """Inversa de la normal estándar por bisección (suficiente para convertir prob -> puntos)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _side(
    p_model: float,
    p_alt: float | None,
    p_market: float | None,
    quote: dict | None,
    rules: Rules,
    alerts: list[str],
) -> dict:
    """Evalúa un lado de un mercado."""
    out = {
        "p_modelo": p_model,
        "p_mercado": p_market,
        "edge": None,
        "momio": quote["momio"] if quote else None,
        "fuente": quote["fuente"] if quote else None,
        "ev": None,
        "estable": None,
        "kelly": 0.0,
        "stake": 0.0,
        "etiqueta": "SIN LÍNEA",
        "motivos": [],
    }
    if p_market is None or quote is None:
        return out
    edge = p_model - p_market
    ev = expected_value(p_model, quote["momio"])
    out["edge"], out["ev"] = edge, ev
    if p_alt is not None:
        out["estable"] = bool(p_alt - p_market > 0) if edge > 0 else None
    if edge < rules.edge_min or ev <= 0:
        out["etiqueta"] = "PASAR"
        return out
    motivos = list(alerts)
    if out["estable"] is False:
        motivos.insert(0, "inestable: el pick cambia de lado al quitar los últimos 2 partidos")
    if quote["fuente"] == "asumido":
        motivos.append("precio asumido (-110): confirma el momio real")
    blocking = [m for m in motivos if not m.startswith("precio asumido")]
    out["motivos"] = motivos
    if blocking:
        out["etiqueta"] = "EVITAR"
        return out
    out["etiqueta"] = "JUGADA"
    out["kelly"] = kelly_fraction(p_model, quote["momio"])
    out["stake"] = min(rules.kelly_fraction * out["kelly"], rules.stake_cap)
    return out


def market_margin(markets: dict, sigma_m: float) -> float | None:
    """Margen del local según el mercado: spread de ESPN, o el spread que se apuesta,
    o (si solo hay moneyline) la probabilidad sin vig convertida a puntos."""
    espn = markets.get("espn") or {}
    if espn.get("spread_home") is not None:
        return -float(espn["spread_home"])
    if markets["spread"]["linea_local"] is not None:
        return -float(markets["spread"]["linea_local"])
    if markets["ml"]["p_local"] is not None:
        return sigma_m * norm_ppf(markets["ml"]["p_local"])
    return None


def evaluate_game(
    *,
    margin: float,
    total: float | None,
    margin_alt: float | None,
    total_alt: float | None,
    markets: dict,
    sigma_m: float,
    sigma_t: float,
    rules: Rules,
    min_games_played: int = 99,
) -> dict:
    """Evalúa los seis lados apostables de un juego (ML, spread y total)."""
    espn = markets.get("espn") or {}
    mkt_margin = market_margin(markets, sigma_m)
    gap = (
        None if mkt_margin is None else margin - mkt_margin
    )  # >0: el modelo ve mejor al local que el mercado
    few = min_games_played < rules.min_games

    # Movimiento de la línea desde la apertura (solo lo publica ESPN).
    move_spread = None
    if espn.get("spread_home") is not None and espn.get("spread_home_open") is not None:
        move_spread = -(
            float(espn["spread_home"]) - float(espn["spread_home_open"])
        )  # >0: se movió hacia el local
    move_total = None
    if espn.get("total") is not None and espn.get("total_open") is not None:
        move_total = float(espn["total"]) - float(espn["total_open"])  # >0: subió

    def side_alerts(toward_home: bool) -> list[str]:
        alerts = []
        if few:
            alerts.append(f"pocos datos: algún equipo tiene menos de {rules.min_games} juegos")
        if gap is not None and abs(gap) >= rules.gap_margin and (gap > 0) == toward_home:
            alerts.append(f"el mercado sabe algo: el modelo difiere {abs(gap):.1f} pts del spread")
        if (
            move_spread is not None
            and abs(move_spread) >= rules.move_spread
            and (move_spread > 0) != toward_home
        ):
            alerts.append(
                f"la línea se movió {abs(move_spread):.1f} pts en contra desde la apertura"
            )
        return alerts

    p_home = float(norm_cdf(margin / sigma_m))
    p_home_alt = None if margin_alt is None else float(norm_cdf(margin_alt / sigma_m))
    ml = markets["ml"]
    out_ml = {
        "local": _side(p_home, p_home_alt, ml["p_local"], ml["local"], rules, side_alerts(True)),
        "visita": _side(
            1 - p_home,
            None if p_home_alt is None else 1 - p_home_alt,
            ml["p_visita"],
            ml["visita"],
            rules,
            side_alerts(False),
        ),
    }

    sp = markets["spread"]
    out_sp = {"local": None, "visita": None}
    if sp["linea_local"] is not None:
        line = float(sp["linea_local"])
        pc = float(norm_cdf((margin + line) / sigma_m))
        pc_alt = None if margin_alt is None else float(norm_cdf((margin_alt + line) / sigma_m))
        out_sp = {
            "local": _side(pc, pc_alt, sp["p_local"], sp["local"], rules, side_alerts(True)),
            "visita": _side(
                1 - pc,
                None if pc_alt is None else 1 - pc_alt,
                sp["p_visita"],
                sp["visita"],
                rules,
                side_alerts(False),
            ),
        }

    tot = markets["total"]
    out_tot = {"over": None, "under": None}
    tgap = None
    if tot["linea"] is not None and total is not None:
        tline = float(tot["linea"])
        ref_total = float(espn["total"]) if espn.get("total") is not None else tline
        tgap = total - ref_total

        def total_alerts(over: bool) -> list[str]:
            alerts = []
            if few:
                alerts.append(f"pocos datos: algún equipo tiene menos de {rules.min_games} juegos")
            if abs(tgap) >= rules.gap_total and (tgap > 0) == over:
                alerts.append(
                    f"el mercado sabe algo: el modelo difiere {abs(tgap):.1f} pts del total"
                )
            if (
                move_total is not None
                and abs(move_total) >= rules.move_total
                and (move_total > 0) != over
            ):
                alerts.append(
                    f"el total se movió {abs(move_total):.1f} pts en contra desde la apertura (¿clima, lesiones?)"
                )
            return alerts

        po = float(norm_cdf((total - tline) / sigma_t))
        po_alt = None if total_alt is None else float(norm_cdf((total_alt - tline) / sigma_t))
        out_tot = {
            "over": _side(po, po_alt, tot["p_over"], tot["over"], rules, total_alerts(True)),
            "under": _side(
                1 - po,
                None if po_alt is None else 1 - po_alt,
                tot["p_under"],
                tot["under"],
                rules,
                total_alerts(False),
            ),
        }

    def best(pair: dict, keys: tuple[str, str]) -> str | None:
        sides = [(k, pair[k]) for k in keys if pair.get(k)]
        if not sides:
            return None
        with_edge = [(k, s) for k, s in sides if s["edge"] is not None]
        if with_edge:
            return max(with_edge, key=lambda ks: ks[1]["edge"])[0]
        return max(sides, key=lambda ks: ks[1]["p_modelo"])[0]

    return {
        "ml": {**out_ml, "pick": best(out_ml, ("local", "visita"))},
        "spread": {**out_sp, "pick": best(out_sp, ("local", "visita"))},
        "total": {**out_tot, "pick": best(out_tot, ("over", "under"))},
        "margen_mercado": mkt_margin,
        "desacuerdo_margen": gap,
        "desacuerdo_total": tgap,
        "mov_spread": move_spread,
        "mov_total": move_total,
    }
