"""Líneas manuales: nombres de equipos, combinación con ESPN y reglas de etiquetado."""

import pandas as pd
import pytest
from src.data.odds import (
    build_markets,
    compare_sources,
    load_manual_odds,
    match_manual,
    normalize_team,
)
from src.model.prob import american_to_prob
from src.model.rules import Rules, evaluate_game

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
            {
                "team_id": "259",
                "team": "Virginia Tech",
                "abbr": "VT",
                "display_name": "Virginia Tech Hokies",
            },
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
    assert (
        "invertidos" in warnings[0]
        and "no reconozco" in warnings[1]
        and "no está en los juegos" in warnings[2]
    )


def test_load_manual_odds_reads_blanks_as_none(tmp_path):
    path = tmp_path / "odds.csv"
    path.write_text(
        "season,week,away,home,ml_away,ml_home,spread_away,spread_away_odds,spread_home_odds,total,over_odds,under_odds,book,captured_at\n"
        "2026,12,Clemson,Duke,,-475,,,,,,,mi casa,2026-11-20\n"
        "2026,12,LSU,Tennessee,-110,-110,+1.5,-120,,45.5,,,mi casa,2026-11-20\n"
        "2026,11,Old,Game,-110,-110,,,,,,,mi casa,2026-11-13\n"
    )
    rows = load_manual_odds(path, season=2026, week=12)
    assert len(rows) == 2
    assert rows[0]["ml_home"] == -475 and rows[0]["ml_away"] is None and rows[0]["total"] is None
    assert (
        rows[1]["spread_away"] == 1.5
        and rows[1]["spread_away_odds"] == -120
        and rows[1]["over_odds"] is None
    )


def test_manual_price_with_market_probability_from_two_sided_quote():
    manual = {"ml_home": -475, "book": "mi casa"}  # solo capturaste un lado
    markets = build_markets(manual, ESPN)
    ml = markets["ml"]
    assert ml["local"] == {"momio": -475, "fuente": "mi casa"}  # se apuesta a TU precio
    assert ml["visita"] == {"momio": 400, "fuente": "espn"}  # el lado que falta sale de ESPN
    fair = american_to_prob(-535) / (american_to_prob(-535) + american_to_prob(400))
    assert ml["p_local"] == pytest.approx(fair)  # sin vig con los dos lados de una misma casa
    assert ml["p_local"] + ml["p_visita"] == pytest.approx(1.0)
    assert markets["spread"]["linea_local"] == -12.5 and markets["spread"]["fuente_linea"] == "espn"


def test_manual_spread_uses_your_line_and_flags_assumed_price():
    manual = {"spread_away": 1.5, "spread_away_odds": -120, "total": 45.5, "book": "mi casa"}
    espn = dict(
        ESPN, spread_home=1.5, total=45.5
    )  # ESPN tiene OTRA línea de spread (+1.5 local) y el mismo total
    markets = build_markets(manual, espn)
    sp = markets["spread"]
    assert (
        sp["linea_local"] == -1.5 and sp["fuente_linea"] == "mi casa"
    )  # +1.5 visitante = -1.5 local
    assert sp["visita"] == {"momio": -120, "fuente": "mi casa"}
    assert sp["local"]["fuente"] == "asumido"  # ese lado no lo capturaste
    assert sp["p_visita"] == pytest.approx(american_to_prob(-120) / 1.045)
    tot = markets["total"]
    assert (
        tot["linea"] == 45.5 and tot["over"]["fuente"] == "espn"
    )  # misma línea: se usan los precios de ESPN
    assert tot["p_over"] + tot["p_under"] == pytest.approx(1.0)


def test_no_odds_at_all_leaves_fields_empty():
    markets = build_markets(None, None)
    assert markets["ml"]["local"] is None and markets["ml"]["p_local"] is None
    assert markets["spread"]["linea_local"] is None and markets["total"]["linea"] is None


def test_compare_sources_flags_price_gaps():
    notes = compare_sources(
        {"ml_away": -1300, "spread_away": 1.5, "total": 37.5},
        {"ml_away": -950, "spread_home": 1.5, "total": 38.5},
    )
    assert any("DraftKings paga mejor" in n for n in notes)
    assert any(n.startswith("Spread del local") for n in notes)
    assert any(n.startswith("Total") for n in notes)
    assert compare_sources({"ml_home": -110}, {"ml_home": -112}) == []


# ----------------------------------------------------------------- etiquetas
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
    out = _evaluate(12.5, 12.5)  # el modelo coincide con el mercado
    assert (
        out["spread"]["local"]["etiqueta"] == "PASAR"
        and out["spread"]["visita"]["etiqueta"] == "PASAR"
    )
    assert out["spread"]["local"]["stake"] == 0.0


def test_stable_edge_without_alerts_is_a_play():
    espn = dict(ESPN, spread_home_open=-12.5, total_open=41.5)  # la línea no se ha movido
    out = _evaluate(16.5, 16.0, espn=espn)  # 4 pts mejor que el mercado, estable
    side = out["spread"]["local"]
    assert side["etiqueta"] == "JUGADA" and side["estable"] is True
    assert side["edge"] >= 0.05 and 0 < side["stake"] <= 0.02
    assert out["spread"]["pick"] == "local"


def test_unstable_pick_is_avoided():
    espn = dict(ESPN, spread_home_open=-12.5, total_open=41.5)
    out = _evaluate(16.5, 10.0, espn=espn)  # sin los últimos 2 partidos el pick se voltea
    side = out["spread"]["local"]
    assert side["etiqueta"] == "EVITAR" and side["estable"] is False and side["stake"] == 0.0
    assert "inestable" in side["motivos"][0]


def test_big_disagreement_with_market_is_avoided():
    espn = dict(ESPN, spread_home_open=-12.5, total_open=41.5)
    out = _evaluate(20.5, 20.5, espn=espn)  # 8 pts de diferencia: el mercado sabe algo
    assert out["spread"]["local"]["etiqueta"] == "EVITAR"
    assert out["desacuerdo_margen"] == pytest.approx(8.0)


def test_line_moving_against_the_pick_is_avoided():
    out = _evaluate(
        8.5, 8.5
    )  # el modelo prefiere al visitante, pero la línea se movió 4 pts hacia el local
    side = out["spread"]["visita"]
    assert side["etiqueta"] == "EVITAR" and any("se movió" in m for m in side["motivos"])


def test_few_games_blocks_plays():
    espn = dict(ESPN, spread_home_open=-12.5, total_open=41.5)
    out = _evaluate(16.5, 16.0, espn=espn, min_games_played=1)
    assert out["spread"]["local"]["etiqueta"] == "EVITAR"


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
    assert out["ml"]["local"]["etiqueta"] == "SIN LÍNEA" and out["ml"]["local"]["edge"] is None
    assert out["ml"]["pick"] == "local" and out["spread"]["pick"] is None
