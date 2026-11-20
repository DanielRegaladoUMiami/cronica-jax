"""Modelos y backtest sobre una liga sintética con ratings conocidos."""

import numpy as np
import pandas as pd
import pytest
from src.model.backtest import calibration, metrics, walk_forward
from src.model.baseline import BaselineModel
from src.model.data import FCS, make_season, make_seasons
from src.model.elo import EloModel
from src.model.simulate import drop_last_games, settle
from src.model.srs import SrsModel, SrsParams, srs_iterative

TRUE = np.linspace(-15, 15, 12)  # rating verdadero de cada equipo
HFA = 3.0


def league(season=2030, weeks=10, noise=0.0, seed=1, flip_from_week=None) -> pd.DataFrame:
    """12 equipos, todos juegan cada semana. margen = rating local - rating visita + ventaja de local (+ ruido)."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, weeks + 1):
        order = rng.permutation(len(TRUE))
        for i in range(0, len(TRUE), 2):
            h, a = int(order[i]), int(order[i + 1])
            margin = TRUE[h] - TRUE[a] + HFA + rng.normal(0, noise)
            total = 50 + 0.4 * (TRUE[h] + TRUE[a])
            if flip_from_week is not None and week >= flip_from_week:
                margin = -margin  # mismo calendario, resultados distintos desde esa semana
            rows.append(
                {
                    "game_id": f"{season}-{week}-{i}",
                    "season": season,
                    "week_index": week,
                    "date": pd.Timestamp(f"{season}-09-01", tz="UTC")
                    + pd.Timedelta(days=7 * week, minutes=i),
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


def test_season_arrays_shape():
    sa = make_season(league(), 2030)
    assert sa.n == 13 and sa.team_ids[-1] == FCS  # 12 equipos + el pseudo-equipo FCS
    assert sa.completed.all() and not sa.fcs.any()
    assert sa.idx("no-existe") == sa.fcs_index  # cualquier equipo que no es FBS cae en FCS


def test_srs_recovers_known_ratings():
    sa = make_season(league(), 2030)
    model = SrsModel(LOOSE)
    state = model.fit(sa, np.ones(len(sa.week), dtype=bool), model.prior(None, None, sa))
    est = np.array([state.rating[sa.idx(str(100 + i))] for i in range(len(TRUE))])
    assert est - est.mean() == pytest.approx(TRUE - TRUE.mean(), abs=1e-3)
    assert state.hfa == pytest.approx(HFA, abs=1e-3)
    h, a = sa.idx("111"), sa.idx("100")
    margin, total = model.predict(state, [h], [a], [False])
    assert margin[0] == pytest.approx(TRUE[11] - TRUE[0] + HFA, abs=1e-3)
    assert total[0] == pytest.approx(50 + 0.4 * (TRUE[11] + TRUE[0]), abs=1e-3)
    neutral, _ = model.predict(state, [h], [a], [True])
    assert neutral[0] == pytest.approx(
        margin[0] - HFA, abs=1e-3
    )  # en sitio neutral no hay ventaja de local


def test_iterative_srs_converges_to_the_linear_solution():
    sa = make_season(league(noise=12.0, seed=3), 2030)
    params = SrsParams(cap=28, k=3.0, hfa_k=40.0)
    model = SrsModel(params)
    prior = model.prior(None, None, sa)
    mask = np.ones(len(sa.week), dtype=bool)
    state = model.fit(sa, mask, prior)
    iterated = srs_iterative(
        sa, mask, hfa=state.hfa, cap=params.cap, k=params.k, prior=prior["rating"]
    )
    assert iterated == pytest.approx(state.rating, abs=1e-6)


def test_shrinkage_pulls_ratings_toward_the_prior():
    sa = make_season(league(weeks=2, noise=10.0, seed=5), 2030)
    mask = np.ones(len(sa.week), dtype=bool)
    loose, tight = SrsModel(SrsParams(k=0.01)), SrsModel(SrsParams(k=50.0))
    spread_loose = np.ptp(loose.fit(sa, mask, loose.prior(None, None, sa)).rating[:-1])
    spread_tight = np.ptp(tight.fit(sa, mask, tight.prior(None, None, sa)).rating[:-1])
    assert spread_tight < 0.2 * spread_loose


def test_prior_carries_last_season_and_handles_newcomers():
    prev = make_season(league(season=2029), 2029)
    games = league(season=2030).replace(
        {"home_id": {"100": "999"}, "away_id": {"100": "999"}}
    )  # un equipo nuevo en FBS
    cur = make_season(games, 2030)
    model = SrsModel(SrsParams(rho=0.5, newcomer=-8.0))
    state = model.fit(prev, np.ones(len(prev.week), dtype=bool), model.prior(None, None, prev))
    prior = model.prior(state, prev, cur)
    assert prior["rating"][cur.idx("105")] == pytest.approx(0.5 * state.rating[prev.idx("105")])
    assert prior["rating"][cur.idx("999")] == -8.0
    assert prior["rating"][cur.fcs_index] == pytest.approx(state.rating[prev.fcs_index])


@pytest.mark.parametrize(
    "model", [SrsModel(), EloModel(), BaselineModel()], ids=["srs", "elo", "baseline"]
)
def test_walk_forward_never_sees_the_future(model):
    base = walk_forward(model, make_seasons(league(noise=8.0)))
    changed = walk_forward(model, make_seasons(league(noise=8.0, flip_from_week=5)))
    upto5 = base["week"] <= 5
    # Las predicciones hasta la semana 5 no dependen de lo que pase de la semana 5 en adelante...
    assert changed.loc[upto5, "pred_margin"].to_numpy() == pytest.approx(
        base.loc[upto5, "pred_margin"].to_numpy()
    )
    # ...y las de la semana 6 sí cambian, porque ya incluyen la semana 5.
    week6 = base["week"] == 6
    assert not np.allclose(changed.loc[week6, "pred_margin"], base.loc[week6, "pred_margin"])
    assert len(base) == 60 and set(base["week"]) == set(range(1, 11))


@pytest.mark.parametrize(
    "model", [SrsModel(), EloModel(), BaselineModel()], ids=["srs", "elo", "baseline"]
)
def test_models_learn_who_is_better(model):
    wf = walk_forward(model, make_seasons(league(noise=6.0)))
    late = wf[wf["week"] >= 6]
    assert metrics(late)["acierto"] > 0.75
    assert np.isfinite(late["pred_margin"]).all()


def test_elo_is_zero_sum_between_fbs_teams():
    sa = make_season(league(noise=10.0), 2030)
    model = EloModel()
    prior = model.prior(None, None, sa)
    state = model.fit(sa, np.ones(len(sa.week), dtype=bool), prior)
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
    assert m["n"] == 4 and m["acierto"] == 0.75
    assert m["mae_margen"] == pytest.approx((3 + 7 + 8 + 7) / 4)
    assert m["mae_total"] == pytest.approx(
        (5 + 7 + 0) / 3
    )  # el juego sin total proyectado no cuenta
    assert m["acierto_14"] == 1.0 and m["n_14"] == 1
    assert 0 < m["brier"] < 0.25
    bins = calibration(df, sigma=16.0)
    assert sum(b["n"] for b in bins) == 4
    assert all(0.5 <= b["predicho"] <= 1 for b in bins)


def test_drop_last_games_removes_two_per_team():
    sa = make_season(league(), 2030)
    mask = sa.week < 8
    h, a = sa.idx("100"), sa.idx("111")
    reduced = drop_last_games(sa, mask, (h, a))
    removed = np.flatnonzero(mask & ~reduced)
    assert 3 <= len(removed) <= 4  # 2 por equipo (uno puede ser el mismo juego)
    for team in (h, a):
        played = np.flatnonzero(mask & ((sa.h == team) | (sa.a == team)))
        assert set(played[-2:]).issubset(set(removed))
    assert not reduced[~mask].any()  # nunca añade juegos futuros


@pytest.mark.parametrize(
    "tipo, lado, linea, margin, total, expected",
    [
        ("ml", "local", None, 3, 50, 1),
        ("ml", "visita", None, 3, 50, -1),
        ("spread", "local", -7.5, 10, 50, 1),
        ("spread", "local", -7.5, 7, 50, -1),
        ("spread", "visita", -7.5, 7, 50, 1),
        ("spread", "local", -7.0, 7, 50, 0),
        ("total", "over", 45.5, 0, 50, 1),
        ("total", "under", 45.5, 0, 50, -1),
        ("total", "over", 50.0, 0, 50, 0),
    ],
)
def test_settle(tipo, lado, linea, margin, total, expected):
    assert settle(tipo, lado, linea, margin, total) == expected
