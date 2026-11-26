"""Manual lines: team names, merging with ESPN, and the labeling rules."""

import pandas as pd
import pytest
from src.data.odds import (
    build_markets,
    compare_sources,
    load_manual_odds,
    match_manual,
    normalize_team,
)
from src.model.prob import american_to_prob, anchored
from src.model.rules import Rules, evaluate_game
from src.model.simulate import MIN_BETS, fit_trust

ESPN = {
    "provider": "DraftKings",
    "spread_home": -12.5,
    "spread_home_odds": -108,
    "spread_away_odds": -112,
    "total": 41.5,
    "over_odds": -105,
    "under_odds": -115,
    "ml_home": -535,
    "ml_away": 400,
    "spread_home_open": -8.5,
    "total_open": 44.5,
}
STEADY = dict(ESPN, spread_home_open=-12.5, total_open=41.5)  # the line has not moved


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Miami (FL)", "miami"),
        ("Miami", "miami"),
        ("Miami (OH)", "miami oh"),
        ("San José State", "san jose state"),
        ("San Jose St.", "san jose state"),
        ("Texas A&M", "texas a&m"),
        ("Hawai'i", "hawai i"),
        ("Hawaii", "hawai i"),
        ("  Boise   State ", "boise state"),
        ("Pitt", "pittsburgh"),
        ("Ole Miss", "ole miss"),
    ],
)
def test_normalize_team(raw, expected):
    assert normalize_team(raw) == expected


def _teams():
    return pd.DataFrame(
        [
            {"team_id": "2390", "team": "Miami", "abbr": "MIA", "display_name": "Miami Hurricanes"},
            {"team_id": "259", "team": "Virginia Tech", "abbr": "VT", "display_name": "VT Hokies"},
            {"team_id": "150", "team": "Duke", "abbr": "DUKE", "display_name": "Duke Blue Devils"},
            {"team_id": "228", "team": "Clemson", "abbr": "CLEM", "display_name": "Clemson Tigers"},
        ]
    )


def _week():
    return pd.DataFrame(
        [
            {"game_id": "g1", "away_id": "259", "home_id": "2390"},
            {"game_id": "g2", "away_id": "228", "home_id": "150"},
        ]
    )


def test_match_manual_rows_to_games():
    rows = [{"away": "Virginia Tech", "home": "Miami (FL)"}, {"away": "Clemson", "home": "Duke"}]
    matched, warnings = match_manual(rows, _week(), _teams())
    assert set(matched) == {"g1", "g2"} and warnings == []


def test_match_manual_never_guesses():
    rows = [
        {"away": "Duke", "home": "Clemson"},
        {"away": "Georgia", "home": "Duke"},
        {"away": "Miami (FL)", "home": "Duke"},
    ]
    matched, warnings = match_manual(rows, _week(), _teams())
    assert matched == {}
    assert "swapped" in warnings[0]
    assert "unknown team" in warnings[1]
    assert "not on this week's schedule" in warnings[2]


def test_load_manual_odds_reads_blanks_as_none(tmp_path):
    path = tmp_path / "odds.csv"
    path.write_text(
        "season,week,away,home,ml_away,ml_home,spread_away,spread_away_odds,spread_home_odds,"
        "total,over_odds,under_odds,book,captured_at\n"
        "2026,12,Clemson,Duke,,-475,,,,,,,my book,2026-11-20\n"
        "2026,12,LSU,Tennessee,-110,-110,+1.5,-120,,45.5,,,,2026-11-20\n"
        "2026,11,Old,Game,-110,-110,,,,,,,my book,2026-11-13\n"
    )
    rows = load_manual_odds(path, season=2026, week=12)
    assert len(rows) == 2
    assert rows[0]["ml_home"] == -475 and rows[0]["ml_away"] is None and rows[0]["total"] is None
    assert rows[1]["spread_away"] == 1.5 and rows[1]["spread_away_odds"] == -120
    assert rows[1]["over_odds"] is None
    assert rows[1]["book"] == "my book"  # default when the book column is empty


def test_manual_price_with_market_probability_from_two_sided_quote():
    manual = {"ml_home": -475, "book": "my book"}  # only one side was captured
    markets = build_markets(manual, ESPN)
    ml = markets["ml"]
    assert ml["home"] == {"odds": -475, "source": "my book"}  # the bet uses YOUR price
    assert ml["away"] == {"odds": 400, "source": "espn"}  # the missing side comes from ESPN
    fair = american_to_prob(-535) / (american_to_prob(-535) + american_to_prob(400))
    assert ml["p_home"] == pytest.approx(fair)  # no-vig from both sides of ONE book
    assert ml["p_home"] + ml["p_away"] == pytest.approx(1.0)
    assert ml["devig"] == {"method": "two_sided", "source": "espn"}
    assert markets["spread"]["home_line"] == -12.5
    assert markets["spread"]["line_source"] == "espn"


def test_manual_spread_uses_your_line_and_flags_assumed_price():
    manual = {"spread_away": 1.5, "spread_away_odds": -120, "total": 45.5, "book": "my book"}
    # ESPN has a DIFFERENT spread (home +1.5) and the same total
    espn = dict(ESPN, spread_home=1.5, total=45.5)
    markets = build_markets(manual, espn)
    sp = markets["spread"]
    assert sp["home_line"] == -1.5 and sp["line_source"] == "my book"  # away +1.5 = home -1.5
    assert sp["away"] == {"odds": -120, "source": "my book"}
    assert sp["home"]["source"] == "assumed"  # that side was not captured
    assert sp["p_away"] == pytest.approx(american_to_prob(-120) / 1.045)
    assert sp["devig"] == {"method": "one_sided", "source": "my book"}
    tot = markets["total"]
    assert tot["line"] == 45.5 and tot["over"]["source"] == "espn"  # same line: ESPN's prices
    assert tot["p_over"] + tot["p_under"] == pytest.approx(1.0)


def test_no_odds_at_all_leaves_fields_empty():
    markets = build_markets(None, None)
    assert markets["ml"]["home"] is None and markets["ml"]["p_home"] is None
    assert markets["spread"]["home_line"] is None and markets["total"]["line"] is None
    assert markets["has_manual"] is False


def test_compare_sources_flags_price_gaps():
    notes = compare_sources(
        {"ml_away": -1300, "spread_away": 1.5, "total": 37.5, "book": "my book"},
        {"ml_away": -950, "spread_home": 1.5, "total": 38.5, "provider": "DraftKings"},
        home="Michigan State",
        away="Oregon",
    )
    assert "Oregon moneyline: my book -1300 vs DraftKings -950 (DraftKings pays better)" in notes
    assert any(n.startswith("Spread: my book has Michigan State -1.5") for n in notes)
    assert "Total: my book 37.5 vs DraftKings 38.5" in notes
    assert compare_sources({"ml_home": -110}, {"ml_home": -112}) == []


def test_compare_sources_flags_a_flipped_favorite():
    notes = compare_sources({"ml_home": -300}, {"ml_home": 250}, home="Duke")
    assert len(notes) == 1 and "disagree on the favorite" in notes[0]


# -------------------------------------------------------------------- labels
def _evaluate(margin, margin_alt, espn=ESPN, total=44.0, total_alt=44.0, **kw):
    return evaluate_game(
        margin=margin,
        total=total,
        margin_alt=margin_alt,
        total_alt=total_alt,
        markets=build_markets(None, espn),
        sigma_m=16.0,
        sigma_t=16.0,
        rules=Rules(),
        **kw,
    )


def test_small_edge_is_a_pass():
    out = _evaluate(12.5, 12.5)  # the model agrees with the market
    assert out["spread"]["home"]["label"] == "PASS" and out["spread"]["away"]["label"] == "PASS"
    assert out["spread"]["home"]["stake"] == 0.0


def test_stable_edge_without_alerts_is_a_play():
    out = _evaluate(16.5, 16.0, espn=STEADY)  # 4 pts better than the market, and stable
    side = out["spread"]["home"]
    assert side["label"] == "PLAY" and side["stable"] is True
    assert side["edge"] >= 0.05 and 0 < side["stake"] <= 0.02
    assert out["spread"]["pick"] == "home"


def test_unstable_pick_is_avoided():
    out = _evaluate(16.5, 10.0, espn=STEADY)  # without the last 2 games the pick flips
    side = out["spread"]["home"]
    assert side["label"] == "AVOID" and side["stable"] is False and side["stake"] == 0.0
    assert side["reasons"][0].startswith("unstable")


def test_big_disagreement_with_market_is_avoided():
    out = _evaluate(20.5, 20.5, espn=STEADY)  # 8 pts apart: the market knows something
    assert out["spread"]["home"]["label"] == "AVOID"
    assert out["margin_gap"] == pytest.approx(8.0)
    assert any("market disagrees" in r for r in out["spread"]["home"]["reasons"])


def test_line_moving_against_the_pick_is_avoided():
    # the model prefers the away side, but the line moved 4 pts toward the home team
    out = _evaluate(8.5, 8.5)
    side = out["spread"]["away"]
    assert side["label"] == "AVOID" and any("moved" in r for r in side["reasons"])
    assert out["spread_move"] == pytest.approx(4.0)


def test_few_games_blocks_plays():
    out = _evaluate(16.5, 16.0, espn=STEADY, min_games_played=1)
    assert out["spread"]["home"]["label"] == "AVOID"
    assert out["spread"]["home"]["reasons"][0].startswith("thin data")


def test_without_lines_there_is_only_the_model_probability():
    out = evaluate_game(
        margin=7.0,
        total=50.0,
        margin_alt=7.0,
        total_alt=50.0,
        markets=build_markets(None, None),
        sigma_m=16.0,
        sigma_t=16.0,
        rules=Rules(),
    )
    assert out["ml"]["home"]["label"] == "NO LINE" and out["ml"]["home"]["edge"] is None
    assert out["ml"]["pick"] == "home" and out["spread"]["pick"] is None


# ------------------------------------------------- the model against the market
MARKET_ONLY = {"trust_ml": 0.0, "trust_spread": 0.0, "trust_total": 0.0}


def _checked(margin, margin_alt=None, espn=STEADY, **trust):
    return evaluate_game(
        margin=margin,
        total=44.0,
        margin_alt=margin if margin_alt is None else margin_alt,
        total_alt=44.0,
        markets=build_markets(None, espn),
        sigma_m=16.0,
        sigma_t=16.0,
        rules=Rules(**trust),
    )


def test_anchored_probability_moves_from_the_market_toward_the_model():
    # The case that started it: the market says 12% and the model 16%.
    assert anchored(0.16, 0.12, 1.0) == 0.16
    assert anchored(0.16, 0.12, 0.0) == 0.12
    assert 0.12 < anchored(0.16, 0.12, 0.5) < 0.16
    assert anchored(0.5, 0.5, 0.3) == pytest.approx(0.5)


def test_a_gap_the_record_does_not_back_is_not_a_play():
    alone = _checked(16.5, 16.0)  # the model on its own: 4 points better than the market
    assert alone["spread"]["home"]["label"] == "PLAY"
    assert alone["spread"]["home"]["p_fair"] == alone["spread"]["home"]["p_model"]
    out = _checked(16.5, 16.0, **MARKET_ONLY)
    side = out["spread"]["home"]
    assert side["label"] == "PASS" and side["stake"] == 0.0
    assert side["p_fair"] == pytest.approx(side["p_market"]) and side["value"] == pytest.approx(0.0)
    assert side["edge"] == pytest.approx(alone["spread"]["home"]["edge"])  # the gap is still shown
    assert "the market has been right" in side["reasons"][-1]
    # A red flag is still a red flag.
    assert _checked(20.5, **MARKET_ONLY)["spread"]["home"]["label"] == "AVOID"
    # Half the trust: the chance sits between the two, and only that part counts as value.
    half = _checked(16.5, 16.0, trust_spread=0.5)["spread"]["home"]
    assert side["p_market"] < half["p_fair"] < half["p_model"]
    assert half["value"] == pytest.approx(half["edge"] / 2, abs=0.01)


def test_moneyline_pick_is_the_likelier_team_not_the_bigger_gap():
    # The market makes the home team an 81% favorite; the model has it closer.
    out = _checked(4.0, **MARKET_ONLY)
    ml = out["ml"]
    assert ml["lean"] == "away" and ml["away"]["edge"] > 0.05  # the model alone likes the underdog
    assert ml["pick"] == "home"  # but the pick is who is likelier to win
    assert (
        ml["home"]["p_fair"] == pytest.approx(ml["home"]["p_market"])
        and ml["home"]["p_fair"] > 0.75
    )
    assert ml["away"]["label"] in ("PASS", "AVOID") and ml["home"]["label"] == "PASS"
    # A spread is priced as a coin flip, so its pick stays the model's lean.
    assert out["spread"]["pick"] == out["spread"]["lean"] == "away"


def _bets(n, follow, seed=3):
    """Synthetic moneyline bets whose results follow the market's number or the model's."""
    rng = __import__("numpy").random.default_rng(seed)
    market = rng.uniform(0.15, 0.70, n)
    model = __import__("numpy").clip(market + rng.uniform(0.0, 0.25, n), 0.05, 0.95)
    truth = market if follow == "market" else model
    won = rng.uniform(size=n) < truth
    return pd.DataFrame(
        {
            "market": "ml",
            "p_market": market,
            "p_model": model,
            "result": [1 if w else -1 for w in won],
        }
    )


def test_trust_is_measured_on_the_bets():
    right_market = fit_trust(_bets(4000, "market"))["ml"]
    assert abs(right_market["weight"]) < 0.25 and right_market["trust"] < 0.25
    right_model = fit_trust(_bets(4000, "model"))["ml"]
    assert right_model["weight"] > 0.75 and right_model["trust"] > 0.75
    wide = [b for b in right_market["buckets"] if b["lo"] >= 0.10]
    assert wide and all(
        abs(b["actual"] - b["market"]) < abs(b["actual"] - b["model"]) for b in wide
    )
    # Too few bets to judge: the market gets the benefit of the doubt.
    few = fit_trust(_bets(MIN_BETS - 1, "model"))
    assert few["ml"]["trust"] == 0.0 and few["ml"]["weight"] is None
    assert few["spread"]["n"] == 0 and few["spread"]["trust"] == 0.0
    assert fit_trust(pd.DataFrame())["total"]["trust"] == 0.0
