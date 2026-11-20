"""Models and backtest on a synthetic league with known ratings."""

import numpy as np
import pandas as pd
import pytest
from src.model.backtest import calibration, metrics, walk_forward
from src.model.baseline import BaselineModel
from src.model.data import FCS, make_season, make_seasons
from src.model.elo import EloModel
from src.model.simulate import drop_last_games, record, settle
from src.model.srs import SrsModel, SrsParams, srs_iterative

TRUE = np.linspace(-15, 15, 12)  # each team's true rating
HFA = 3.0


def league(season=2030, weeks=10, noise=0.0, seed=1, flip_from_week=None) -> pd.DataFrame:
    """12 teams, everyone plays every week.
    margin = home rating - away rating + home-field advantage (+ noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, weeks + 1):
        order = rng.permutation(len(TRUE))
        for i in range(0, len(TRUE), 2):
            h, a = int(order[i]), int(order[i + 1])
            margin = TRUE[h] - TRUE[a] + HFA + rng.normal(0, noise)
            total = 50 + 0.4 * (TRUE[h] + TRUE[a])
            if flip_from_week is not None and week >= flip_from_week:
                margin = -margin  # same schedule, different results from that week on
            start = pd.Timestamp(f"{season}-09-01", tz="UTC")
            rows.append(
                {
                    "game_id": f"{season}-{week}-{i}",
                    "season": season,
                    "week_index": week,
                    "date": start + pd.Timedelta(days=7 * week, minutes=i),
                    "home_id": str(100 + h),
                    "home_team": f"T{h}",
                    "away_id": str(100 + a),
                    "away_team": f"T{a}",
                    "home_points": (total + margin) / 2,
                    "away_points": (total - margin) / 2,
                    "neutral_site": False,
                    "completed": True,
                    "fcs_game": False,
                    "home_fbs": True,
                    "away_fbs": True,
                }
            )
    return pd.DataFrame(rows)


LOOSE = SrsParams(cap=99, k=1e-6, k_total=1e-6, hfa_k=1e-6, base_total_k=1e-6)


def _all(sa):
    return np.ones(len(sa.week), dtype=bool)


def test_season_arrays_shape():
    sa = make_season(league(), 2030)
    assert sa.n == 13 and sa.team_ids[-1] == FCS  # 12 teams + the FCS pseudo-team
    assert sa.completed.all() and not sa.fcs.any()
    assert sa.idx("does-not-exist") == sa.fcs_index  # any non-FBS team falls into FCS


def test_srs_recovers_known_ratings():
    sa = make_season(league(), 2030)
    model = SrsModel(LOOSE)
    state = model.fit(sa, _all(sa), model.prior(None, None, sa))
    est = np.array([state.rating[sa.idx(str(100 + i))] for i in range(len(TRUE))])
    assert est - est.mean() == pytest.approx(TRUE - TRUE.mean(), abs=1e-3)
    assert state.hfa == pytest.approx(HFA, abs=1e-3)
    h, a = sa.idx("111"), sa.idx("100")
    margin, total = model.predict(state, [h], [a], [False])
    assert margin[0] == pytest.approx(TRUE[11] - TRUE[0] + HFA, abs=1e-3)
    assert total[0] == pytest.approx(50 + 0.4 * (TRUE[11] + TRUE[0]), abs=1e-3)
    neutral, _ = model.predict(state, [h], [a], [True])
    assert neutral[0] == pytest.approx(margin[0] - HFA, abs=1e-3)  # no home edge on neutral turf


def test_iterative_srs_converges_to_the_linear_solution():
    sa = make_season(league(noise=12.0, seed=3), 2030)
    params = SrsParams(cap=28, k=3.0, hfa_k=40.0)
    model = SrsModel(params)
    prior = model.prior(None, None, sa)
    state = model.fit(sa, _all(sa), prior)
    iterated = srs_iterative(
        sa, _all(sa), hfa=state.hfa, cap=params.cap, k=params.k, prior=prior["rating"]
    )
    assert iterated == pytest.approx(state.rating, abs=1e-6)


def test_shrinkage_pulls_ratings_toward_the_prior():
    sa = make_season(league(weeks=2, noise=10.0, seed=5), 2030)
    loose, tight = SrsModel(SrsParams(k=0.01)), SrsModel(SrsParams(k=50.0))
    spread_loose = np.ptp(loose.fit(sa, _all(sa), loose.prior(None, None, sa)).rating[:-1])
    spread_tight = np.ptp(tight.fit(sa, _all(sa), tight.prior(None, None, sa)).rating[:-1])
    assert spread_tight < 0.2 * spread_loose


def test_prior_carries_last_season_and_handles_newcomers():
    prev = make_season(league(season=2029), 2029)
    # one team is new to FBS this season
    games = league(season=2030).replace({"home_id": {"100": "999"}, "away_id": {"100": "999"}})
    cur = make_season(games, 2030)
    model = SrsModel(SrsParams(rho=0.5, newcomer=-8.0))
    state = model.fit(prev, _all(prev), model.prior(None, None, prev))
    prior = model.prior(state, prev, cur)
    assert prior["rating"][cur.idx("105")] == pytest.approx(0.5 * state.rating[prev.idx("105")])
    assert prior["rating"][cur.idx("999")] == -8.0
    assert prior["rating"][cur.fcs_index] == pytest.approx(state.rating[prev.fcs_index])


MODELS = [SrsModel(), EloModel(), BaselineModel()]
MODEL_IDS = ["srs", "elo", "baseline"]


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_walk_forward_never_sees_the_future(model):
    base = walk_forward(model, make_seasons(league(noise=8.0)))
    changed = walk_forward(model, make_seasons(league(noise=8.0, flip_from_week=5)))
    upto5 = base["week"] <= 5
    # Predictions through week 5 do not depend on what happens from week 5 on...
    assert changed.loc[upto5, "pred_margin"].to_numpy() == pytest.approx(
        base.loc[upto5, "pred_margin"].to_numpy()
    )
    # ...and week 6's do change, because they already include week 5.
    week6 = base["week"] == 6
    assert not np.allclose(changed.loc[week6, "pred_margin"], base.loc[week6, "pred_margin"])
    assert len(base) == 60 and set(base["week"]) == set(range(1, 11))


@pytest.mark.parametrize("model", MODELS, ids=MODEL_IDS)
def test_models_learn_who_is_better(model):
    wf = walk_forward(model, make_seasons(league(noise=6.0)))
    late = wf[wf["week"] >= 6]
    assert metrics(late)["accuracy"] > 0.75
    assert np.isfinite(late["pred_margin"]).all()


def test_elo_is_zero_sum_between_fbs_teams():
    sa = make_season(league(noise=10.0), 2030)
    model = EloModel()
    prior = model.prior(None, None, sa)
    state = model.fit(sa, _all(sa), prior)
    assert state.rating[:-1].sum() == pytest.approx(prior["rating"][:-1].sum())
    assert state.rating[sa.idx("111")] > state.rating[sa.idx("100")]


def test_metrics_and_calibration():
    df = pd.DataFrame(
        {
            "pred_margin": [10.0, -4.0, 20.0, 3.0],
            "margin": [7.0, 3.0, 28.0, 10.0],
            "pred_total": [50.0, 40.0, 60.0, np.nan],
            "total": [45.0, 47.0, 60.0, 30.0],
        }
    )
    m = metrics(df, sigma_margin=16.0)
    assert m["n"] == 4 and m["accuracy"] == 0.75
    assert m["mae_margin"] == pytest.approx((3 + 7 + 8 + 7) / 4)
    assert m["mae_total"] == pytest.approx((5 + 7 + 0) / 3)  # no projected total, not counted
    assert m["accuracy_14"] == 1.0 and m["n_14"] == 1
    assert 0 < m["brier"] < 0.25
    bins = calibration(df, sigma=16.0)
    assert sum(b["n"] for b in bins) == 4
    assert all(0.5 <= b["predicted"] <= 1 for b in bins)
    assert metrics(df.iloc[0:0]) == {"n": 0}


def test_drop_last_games_removes_two_per_team():
    sa = make_season(league(), 2030)
    mask = sa.week < 8
    h, a = sa.idx("100"), sa.idx("111")
    reduced = drop_last_games(sa, mask, (h, a))
    removed = np.flatnonzero(mask & ~reduced)
    assert 3 <= len(removed) <= 4  # 2 per team (one could be the same game)
    for team in (h, a):
        played = np.flatnonzero(mask & ((sa.h == team) | (sa.a == team)))
        assert set(played[-2:]).issubset(set(removed))
    assert not reduced[~mask].any()  # it never adds future games


@pytest.mark.parametrize(
    "market, side, line, margin, total, expected",
    [
        ("ml", "home", None, 3, 50, 1),
        ("ml", "away", None, 3, 50, -1),
        ("spread", "home", -7.5, 10, 50, 1),
        ("spread", "home", -7.5, 7, 50, -1),
        ("spread", "away", -7.5, 7, 50, 1),
        ("spread", "home", -7.0, 7, 50, 0),
        ("total", "over", 45.5, 0, 50, 1),
        ("total", "under", 45.5, 0, 50, -1),
        ("total", "over", 50.0, 0, 50, 0),
    ],
)
def test_settle(market, side, line, margin, total, expected):
    assert settle(market, side, line, margin, total) == expected


def test_record_summarizes_flat_bets():
    bets = pd.DataFrame({"result": [1, 1, -1, 0], "profit": [0.909, 1.5, -1.0, 0.0]})
    rec = record(bets)
    assert (rec["n"], rec["won"], rec["lost"], rec["push"]) == (4, 2, 1, 1)
    assert rec["win_pct"] == pytest.approx(2 / 3)
    assert rec["units"] == pytest.approx(1.409) and rec["roi"] == pytest.approx(1.409 / 4)
    assert record(bets.iloc[0:0])["n"] == 0
