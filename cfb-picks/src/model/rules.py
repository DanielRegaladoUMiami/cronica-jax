"""Decision rules: from a projection and a set of lines to PLAY / PASS / AVOID.

The same function labels this week's picks and the betting backtest, so the
track record measures exactly the rule shown on the site.

PLAY   = edge of at least 5%, positive expected value at the real price, no red flags.
AVOID  = there is an edge on paper, but the pick is unstable or the market is
         telling us something the model cannot know (injuries, weather, news).
PASS   = not enough edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .prob import expected_value, kelly_fraction, norm_cdf

PLAY, AVOID, PASS, NO_LINE = "PLAY", "AVOID", "PASS", "NO LINE"


@dataclass(frozen=True)
class Rules:
    edge_min: float = 0.05  # minimum edge (model probability minus no-vig market probability)
    kelly_fraction: float = 0.25  # quarter Kelly
    stake_cap: float = 0.02  # never more than 2% of bankroll on one play
    gap_margin: float = (
        5.0  # points between model and market spread that trigger AVOID (ML, spread)
    )
    gap_total: float = 7.0  # same, for totals
    move_spread: float = 3.0  # line move (open -> now) against the pick that triggers AVOID
    move_total: float = 4.0
    min_games: int = 3  # games each team needs before its rating is trusted

    def as_dict(self) -> dict:
        return asdict(self)


def norm_ppf(p: float) -> float:
    """Inverse standard normal by bisection (plenty for turning a probability into points)."""
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
    unstable: str = "unstable: the pick flips without each team's last 2 games",
) -> dict:
    """Evaluate one side of a market. `rules` only needs edge_min, kelly_fraction and stake_cap."""
    out = {
        "p_model": p_model,
        "p_market": p_market,
        "edge": None,
        "odds": quote["odds"] if quote else None,
        "source": quote["source"] if quote else None,
        "ev": None,
        "stable": None,
        "kelly": 0.0,
        "stake": 0.0,
        "label": NO_LINE,
        "reasons": [],
    }
    if p_market is None or quote is None:
        return out
    edge = p_model - p_market
    ev = expected_value(p_model, quote["odds"])
    out["edge"], out["ev"] = edge, ev
    if p_alt is not None:
        out["stable"] = bool(p_alt - p_market > 0) if edge > 0 else None
    if edge < rules.edge_min or ev <= 0:
        out["label"] = PASS
        return out
    reasons = list(alerts)
    if out["stable"] is False:
        reasons.insert(0, unstable)
    if quote["source"] == "assumed":
        reasons.append("assumed price (-110): confirm the real odds")
    blocking = [r for r in reasons if not r.startswith("assumed price")]
    out["reasons"] = reasons
    if blocking:
        out["label"] = AVOID
        return out
    out["label"] = PLAY
    out["kelly"] = kelly_fraction(p_model, quote["odds"])
    out["stake"] = min(rules.kelly_fraction * out["kelly"], rules.stake_cap)
    return out


def market_margin(markets: dict, sigma_m: float) -> float | None:
    """Home margin according to the market: ESPN's spread, else the spread being bet,
    else (moneyline only) the no-vig probability converted to points."""
    espn = markets.get("espn") or {}
    if espn.get("spread_home") is not None:
        return -float(espn["spread_home"])
    if markets["spread"]["home_line"] is not None:
        return -float(markets["spread"]["home_line"])
    if markets["ml"]["p_home"] is not None:
        return sigma_m * norm_ppf(markets["ml"]["p_home"])
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
    """Evaluate the six bettable sides of a game (moneyline, spread and total)."""
    espn = markets.get("espn") or {}
    mkt_margin = market_margin(markets, sigma_m)
    # gap > 0: the model likes the home team more than the market does
    gap = None if mkt_margin is None else margin - mkt_margin
    few = min_games_played < rules.min_games
    thin = f"thin data: a team has played fewer than {rules.min_games} games"

    # Line movement since the open (only ESPN publishes the opening line).
    spread_move = None
    if espn.get("spread_home") is not None and espn.get("spread_home_open") is not None:
        # > 0: the line moved toward the home team
        spread_move = -(float(espn["spread_home"]) - float(espn["spread_home_open"]))
    total_move = None
    if espn.get("total") is not None and espn.get("total_open") is not None:
        total_move = float(espn["total"]) - float(espn["total_open"])  # > 0: the total went up

    def side_alerts(toward_home: bool) -> list[str]:
        alerts = []
        if few:
            alerts.append(thin)
        if gap is not None and abs(gap) >= rules.gap_margin and (gap > 0) == toward_home:
            alerts.append(f"the market disagrees: the model is {abs(gap):.1f} pts off the spread")
        if (
            spread_move is not None
            and abs(spread_move) >= rules.move_spread
            and (spread_move > 0) != toward_home
        ):
            alerts.append(
                f"the line moved {abs(spread_move):.1f} pts against this side since it opened"
            )
        return alerts

    p_home = float(norm_cdf(margin / sigma_m))
    p_home_alt = None if margin_alt is None else float(norm_cdf(margin_alt / sigma_m))
    ml = markets["ml"]
    out_ml = {
        "home": _side(p_home, p_home_alt, ml["p_home"], ml["home"], rules, side_alerts(True)),
        "away": _side(
            1 - p_home,
            None if p_home_alt is None else 1 - p_home_alt,
            ml["p_away"],
            ml["away"],
            rules,
            side_alerts(False),
        ),
    }

    sp = markets["spread"]
    out_sp = {"home": None, "away": None}
    if sp["home_line"] is not None:
        line = float(sp["home_line"])
        pc = float(norm_cdf((margin + line) / sigma_m))
        pc_alt = None if margin_alt is None else float(norm_cdf((margin_alt + line) / sigma_m))
        out_sp = {
            "home": _side(pc, pc_alt, sp["p_home"], sp["home"], rules, side_alerts(True)),
            "away": _side(
                1 - pc,
                None if pc_alt is None else 1 - pc_alt,
                sp["p_away"],
                sp["away"],
                rules,
                side_alerts(False),
            ),
        }

    tot = markets["total"]
    out_tot = {"over": None, "under": None}
    tgap = None
    if tot["line"] is not None and total is not None:
        tline = float(tot["line"])
        ref_total = float(espn["total"]) if espn.get("total") is not None else tline
        tgap = total - ref_total

        def total_alerts(over: bool) -> list[str]:
            alerts = []
            if few:
                alerts.append(thin)
            if abs(tgap) >= rules.gap_total and (tgap > 0) == over:
                alerts.append(
                    f"the market disagrees: the model is {abs(tgap):.1f} pts off the total"
                )
            if (
                total_move is not None
                and abs(total_move) >= rules.move_total
                and (total_move > 0) != over
            ):
                alerts.append(
                    f"the total moved {abs(total_move):.1f} pts against this side since it opened"
                    " (weather or injuries?)"
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
        return max(sides, key=lambda ks: ks[1]["p_model"])[0]

    return {
        "ml": {**out_ml, "pick": best(out_ml, ("home", "away"))},
        "spread": {**out_sp, "pick": best(out_sp, ("home", "away"))},
        "total": {**out_tot, "pick": best(out_tot, ("over", "under"))},
        "market_margin": mkt_margin,
        "margin_gap": gap,
        "total_gap": tgap,
        "spread_move": spread_move,
        "total_move": total_move,
    }
