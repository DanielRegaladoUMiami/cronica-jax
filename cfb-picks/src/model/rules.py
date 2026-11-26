"""Decision rules: from a projection and a set of lines to PLAY / PASS / AVOID.

The same function labels this week's picks and the betting backtest, so the
track record measures exactly the rule shown on the site.

The model and the market each give a probability. Their difference is a
DISAGREEMENT, not yet an advantage: the backtest measures how much of it came
true (`trust`), and only that part counts.

p_model = the model on its own.
p_fair  = the market's probability moved toward the model's by `trust`. With the
          trust measured so far (0: when the two disagreed, the market was right)
          it is simply the market's probability.
edge    = p_model - market: how far the model is from the market.
value   = p_fair - market: the part of that the evidence backs.

PLAY   = value of at least 5%, positive expected value at the real price, no red flags.
AVOID  = the model sees 5%+ on paper, but the pick is unstable or the market is
         telling us something the model cannot know (injuries, weather, news).
PASS   = no value: the model agrees with the market, or disagrees without the
         evidence to back it.

The moneyline pick is the team more likely to win (by p_fair). `lean` is the side
the model likes more than the market does, which is what the betting backtest follows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .prob import anchored, expected_value, kelly_fraction, norm_cdf

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
    # How much of the model's disagreement with the market to believe, per market:
    # 1 = all of it (the model alone), 0 = none (the market's number). The picks
    # pipeline measures it on the betting backtest; 1 leaves the model untouched.
    trust_ml: float = 1.0
    trust_spread: float = 1.0
    trust_total: float = 1.0

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
    trust: float = 1.0,
) -> dict:
    """Evaluate one side of a market."""
    out = {
        "p_model": p_model,
        "p_fair": p_model,
        "p_market": p_market,
        "edge": None,
        "value": None,
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
    p_fair = anchored(p_model, p_market, trust)
    value = p_fair - p_market
    ev = expected_value(p_fair, quote["odds"])
    out.update(edge=edge, p_fair=p_fair, value=value, ev=ev)
    if p_alt is not None:
        out["stable"] = bool(p_alt - p_market > 0) if edge > 0 else None
    if edge < rules.edge_min or expected_value(p_model, quote["odds"]) <= 0:
        out["label"] = PASS
        return out
    reasons = list(alerts)
    if out["stable"] is False:
        reasons.insert(0, "unstable: the pick flips without each team's last 2 games")
    if quote["source"] == "assumed":
        reasons.append("assumed price (-110): confirm the real odds")
    blocking = [r for r in reasons if not r.startswith("assumed price")]
    out["reasons"] = reasons
    if blocking:
        out["label"] = AVOID
        return out
    if value < rules.edge_min or ev <= 0:
        # The model disagrees with the market, but the record says not to believe it.
        out["label"] = PASS
        out["reasons"] = [
            *reasons,
            "the model disagrees with the market; when it has, the market has been right",
        ]
        return out
    out["label"] = PLAY
    out["kelly"] = kelly_fraction(p_fair, quote["odds"])
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
        "home": _side(
            p_home, p_home_alt, ml["p_home"], ml["home"], rules, side_alerts(True), rules.trust_ml
        ),
        "away": _side(
            1 - p_home,
            None if p_home_alt is None else 1 - p_home_alt,
            ml["p_away"],
            ml["away"],
            rules,
            side_alerts(False),
            rules.trust_ml,
        ),
    }

    sp = markets["spread"]
    out_sp = {"home": None, "away": None}
    if sp["home_line"] is not None:
        line = float(sp["home_line"])
        pc = float(norm_cdf((margin + line) / sigma_m))
        pc_alt = None if margin_alt is None else float(norm_cdf((margin_alt + line) / sigma_m))
        out_sp = {
            "home": _side(
                pc, pc_alt, sp["p_home"], sp["home"], rules, side_alerts(True), rules.trust_spread
            ),
            "away": _side(
                1 - pc,
                None if pc_alt is None else 1 - pc_alt,
                sp["p_away"],
                sp["away"],
                rules,
                side_alerts(False),
                rules.trust_spread,
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
            "over": _side(
                po, po_alt, tot["p_over"], tot["over"], rules, total_alerts(True), rules.trust_total
            ),
            "under": _side(
                1 - po,
                None if po_alt is None else 1 - po_alt,
                tot["p_under"],
                tot["under"],
                rules,
                total_alerts(False),
                rules.trust_total,
            ),
        }

    def lean(pair: dict, keys: tuple[str, str]) -> str | None:
        """The side the model likes more than the market does."""
        sides = [(k, pair[k]) for k in keys if pair.get(k)]
        if not sides:
            return None
        with_edge = [(k, s) for k, s in sides if s["edge"] is not None]
        if with_edge:
            return max(with_edge, key=lambda ks: ks[1]["edge"])[0]
        return max(sides, key=lambda ks: ks[1]["p_model"])[0]

    def likely(pair: dict, keys: tuple[str, str]) -> str | None:
        """The side more likely to win, by the probability the evidence supports."""
        sides = [(k, pair[k]) for k in keys if pair.get(k)]
        return max(sides, key=lambda ks: ks[1]["p_fair"])[0] if sides else None

    ml_lean = lean(out_ml, ("home", "away"))
    sp_lean = lean(out_sp, ("home", "away"))
    tot_lean = lean(out_tot, ("over", "under"))
    return {
        # A moneyline pick answers "who wins": the likelier team, not the bigger disagreement.
        "ml": {**out_ml, "pick": likely(out_ml, ("home", "away")), "lean": ml_lean},
        # A spread or a total is priced as a coin flip, so the pick is the model's lean.
        "spread": {**out_sp, "pick": sp_lean, "lean": sp_lean},
        "total": {**out_tot, "pick": tot_lean, "lean": tot_lean},
        "market_margin": mkt_margin,
        "margin_gap": gap,
        "total_gap": tgap,
        "spread_move": spread_move,
        "total_move": total_move,
    }
