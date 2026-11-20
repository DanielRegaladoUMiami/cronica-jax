"""The soccer goal models and the walk-forward backtest, on a synthetic league."""

from dataclasses import replace

import numpy as np
import pytest
from soccer_league import synthetic_league
from src.soccer.backtest import calibration, metrics, walk_forward
from src.soccer.model import (
    AverageModel,
    EloModel,
    PoissonModel,
    PoissonParams,
    handicap_probs,
    make_leagues,
    most_likely_score,
    outcome_probs,
    poisson_pmf,
    score_matrix,
    total_probs,
)


@pytest.fixture(scope="module")
def league():
    matches, attack, defense = synthetic_league()
    return make_leagues(matches)["syn.1"], attack, defense


def with_home_goals(la, where, extra):
    """A copy of the league with `extra` goals added to the home side of the selected matches."""
    hg = la.hg.copy()
    hg[where] += extra
    return replace(la, hg=hg)


# ---------------------------------------------------------------- scorelines
def test_poisson_pmf_is_a_distribution():
    pmf = poisson_pmf(1.4)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-6)
    assert pmf[0] == pytest.approx(np.exp(-1.4))


def test_score_matrix_sums_to_one_and_rho_adds_draws():
    plain = score_matrix(1.5, 1.1)
    nudged = score_matrix(1.5, 1.1, rho=-0.09)
    assert plain.sum() == pytest.approx(1.0)
    assert nudged.sum() == pytest.approx(1.0)
    assert plain[2, 1] == pytest.approx(poisson_pmf(1.5)[2] * poisson_pmf(1.1)[1] / 1.0, rel=1e-6)
    assert nudged[0, 0] > plain[0, 0] and nudged[1, 1] > plain[1, 1]
    assert outcome_probs(nudged)[1] > outcome_probs(plain)[1]


def test_outcome_probs():
    home, draw, away = outcome_probs(score_matrix(1.3, 1.3))
    assert home + draw + away == pytest.approx(1.0)
    assert home == pytest.approx(away)
    strong = outcome_probs(score_matrix(2.4, 0.7))
    assert strong[0] > 0.7 > strong[1] > strong[2]


def test_totals_and_pushes():
    grid = score_matrix(1.6, 1.2)
    over, push, under = total_probs(grid, 2.5)
    assert push == 0 and over + under == pytest.approx(1.0)
    over3, push3, under3 = total_probs(grid, 3.0)
    assert push3 == pytest.approx(float(sum(grid[i, 3 - i] for i in range(4))))
    assert over3 + push3 + under3 == pytest.approx(1.0)
    assert over3 < over and under3 + push3 > under  # a higher line is harder to go over


def test_handicaps_match_the_three_results():
    grid = score_matrix(1.6, 1.2)
    home, draw, away = outcome_probs(grid)
    assert handicap_probs(grid, -0.5) == pytest.approx((home, 0.0, draw + away))
    assert handicap_probs(grid, 0.5) == pytest.approx((home + draw, 0.0, away))
    assert handicap_probs(grid, 0.0) == pytest.approx((home, draw, away))  # draw no bet
    cover, push, miss = handicap_probs(grid, -1.0)
    assert cover < home and push > 0 and cover + push + miss == pytest.approx(1.0)


def test_most_likely_score():
    assert most_likely_score(score_matrix(1.6, 0.4)) == (1, 0)
    assert most_likely_score(score_matrix(0.3, 2.9)) in {(0, 2), (0, 3)}


# -------------------------------------------------------------------- ratings
def test_poisson_recovers_team_strength(league):
    la, attack, defense = league
    model = PoissonModel(PoissonParams(half_life=365, k=5, rho=0.0))
    state = model.fit(la, np.ones(len(la.h), dtype=bool), float(la.day.max()) + 1)
    assert np.corrcoef(state.attack, attack)[0, 1] > 0.85
    assert np.corrcoef(state.defense, defense)[0, 1] > 0.8
    assert 0.1 < state.home_edge < 0.4  # the simulated edge is 0.25
    best_h, best_a = model.expected_goals(state, [0], [la.n - 1])
    worst_h, worst_a = model.expected_goals(state, [la.n - 1], [0])
    assert best_h[0] > 2 * best_a[0] and worst_a[0] > worst_h[0]
    assert state.games.min() > 0


def test_fit_does_not_look_ahead(league):
    la, _, _ = league
    model = PoissonModel()
    everything = np.ones(len(la.h), dtype=bool)
    cutoff = float(np.median(la.day))
    before = model.fit(la, everything, cutoff)
    changed = with_home_goals(la, la.day >= cutoff, 9)  # change every later result
    after = model.fit(changed, everything, cutoff)
    assert np.allclose(before.theta, after.theta)


def test_mask_removes_matches(league):
    la, _, _ = league
    model = PoissonModel()
    everything = np.ones(len(la.h), dtype=bool)
    cutoff = float(la.day.max()) + 1
    full = model.fit(la, everything, cutoff)
    mask = everything.copy()
    mask[np.flatnonzero((la.h == 0) | (la.a == 0))[-6:]] = False
    partial = model.fit(la, mask, cutoff)
    assert partial.games[0] == full.games[0] - 6
    assert not np.allclose(full.theta, partial.theta)


def test_promoted_team_starts_below_average():
    matches, _, _ = synthetic_league(newcomer=(2025, 3))
    la = make_leagues(matches)["syn.1"]
    newcomer = la.index["103"]
    start_2025 = float(la.day[la.season == 2025].min())
    model = PoissonModel(PoissonParams(promoted=-0.2))
    state = model.fit(la, np.ones(len(la.h), dtype=bool), start_2025)
    assert state.games[newcomer] == 0
    assert state.attack[newcomer] == pytest.approx(-0.2, abs=1e-6)
    assert state.defense[newcomer] == pytest.approx(-0.2, abs=1e-6)
    # Two seasons of data for everyone else: their ratings are not the prior.
    others = np.delete(np.arange(la.n), newcomer)
    assert np.abs(state.attack[others]).max() > 0.2


def test_recent_matches_weigh_more(league):
    la, _, _ = league
    everything = np.ones(len(la.h), dtype=bool)
    cutoff = float(la.day.max()) + 1
    # A scoring burst in one team's last four home matches.
    burst = with_home_goals(la, np.flatnonzero(la.h == 5)[-4:], 4)
    fast = PoissonModel(PoissonParams(half_life=60)).fit(burst, everything, cutoff)
    slow = PoissonModel(PoissonParams(half_life=2000)).fit(burst, everything, cutoff)
    assert fast.attack[5] > slow.attack[5]


def test_simple_models_give_valid_forecasts(league):
    la, _, _ = league
    everything = np.ones(len(la.h), dtype=bool)
    cutoff = float(la.day.max()) + 1
    elo = EloModel()
    state = elo.fit(la, everything, cutoff)
    home, draw, away = elo.probs(elo.diff(state, [0, la.n - 1], [la.n - 1, 0]))
    assert np.allclose(home + draw + away, 1.0) and (draw > 0).all()
    assert home[0] > away[0] and state.rating[0] > state.rating[la.n - 1]
    avg = AverageModel()
    lam_h, lam_a = avg.expected_goals(avg.fit(la, everything, cutoff), [0], [la.n - 1])
    assert lam_h[0] > lam_a[0] > 0


# ------------------------------------------------------------------- backtest
def test_walk_forward_only_uses_the_past(league):
    la, _, _ = league
    model = PoissonModel()
    first = walk_forward(model, {"syn.1": la}, [2024, 2025])
    assert set(first["season"]) == {2024, 2025}
    assert len(first) == int((la.completed & np.isin(la.season, [2024, 2025])).sum())
    probs = first[["p_home", "p_draw", "p_away"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)

    last_week = la.week.max()
    changed = with_home_goals(la, la.week == last_week, 5)
    second = walk_forward(model, {"syn.1": changed}, [2024, 2025])
    earlier = first["week"] < last_week
    assert np.allclose(first.loc[earlier, "lam_home"], second.loc[earlier, "lam_home"])
    # Matches of the changed week were predicted before they were played: same forecast.
    same_week = first["week"] == last_week
    assert np.allclose(first.loc[same_week, "lam_home"], second.loc[same_week, "lam_home"])


def test_metrics_reward_a_better_forecast(league):
    la, _, _ = league
    good = metrics(walk_forward(PoissonModel(), {"syn.1": la}, [2025]))
    assert good["n"] > 0 and 0.35 < good["accuracy"] < 0.9
    assert 0 < good["rps"] < 0.25 and good["log_loss"] < np.log(3)  # beats a coin with three sides

    frame = walk_forward(PoissonModel(), {"syn.1": la}, [2025])
    gd = frame["hg"] - frame["ag"]
    perfect = frame.assign(
        p_home=(gd > 0).astype(float), p_draw=(gd == 0).astype(float), p_away=(gd < 0).astype(float)
    )
    assert metrics(perfect)["rps"] == pytest.approx(0.0, abs=1e-9)
    assert metrics(perfect)["accuracy"] == 1.0
    assert metrics(frame.iloc[:0]) == {"n": 0}


def test_calibration_bins(league):
    la, _, _ = league
    bins = calibration(walk_forward(PoissonModel(), {"syn.1": la}, [2024, 2025]))
    assert bins and all(b["n"] >= 20 for b in bins)
    assert all(b["from"] <= b["predicted"] <= b["to"] + 1e-9 for b in bins)
    assert all(0 <= b["observed"] <= 1 for b in bins)
