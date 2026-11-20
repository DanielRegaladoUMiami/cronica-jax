"""Picks de la semana.

Uso:
    python -m src.picks --week 12        # una semana concreta
    python -m src.picks                  # la semana en curso
    python -m src.picks --skip-build     # sin volver a bajar games.parquet

Qué hace:
1. Actualiza games.parquet y las líneas de cierre de juegos ya jugados.
2. Ajusta el modelo seleccionado con todo lo jugado hasta hoy.
3. Para cada juego: marcador proyectado, probabilidades, edge contra tus líneas
   (data/odds_manual.csv) o las de ESPN, chequeo de estabilidad y etiqueta.
4. Exporta site/data/picks.json y site/data/backtest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .data import build as build_mod
from .data.espn import EspnClient, EspnError
from .data.odds import (
    build_markets,
    compare_sources,
    load_manual_odds,
    match_manual,
    normalize_team,
    team_lookup,
)
from .data.odds_history import fetch_odds_history, load_odds_history
from .data.parse import parse_calendar, parse_event, parse_event_odds
from .model.backtest import by_week, calibration, fbs_only, metrics, residual_sigma, walk_forward
from .model.data import make_seasons
from .model.prob import american_to_decimal, norm_cdf, prob_to_american
from .model.registry import build_models, load_params, selected_name
from .model.rules import Rules, evaluate_game
from .model.simulate import drop_last_games, record, simulate_bets

PROTOTYPE_REFERENCE = {"acierto": 0.73, "acierto_14": 0.92, "mae_margen": 15.5, "mae_total": 20.0}
MIN_POINTS = 3.0  # piso para el marcador proyectado de un equipo


# ------------------------------------------------------------------ utilidades
def _clean(obj):
    """Deja el objeto listo para JSON: sin NaN ni tipos de numpy, con decimales razonables."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    if obj is pd.NA or obj is None:
        return None
    return obj


def week_for(calendar: dict, now: datetime) -> int:
    """Semana en curso como week_index (postemporada = última semana regular + 1)."""
    regular = sorted(calendar["regular"], key=lambda w: w["week"])
    if not regular or now < regular[0]["start"]:
        return 1
    for entry in regular:
        if entry["start"] <= now < entry["end"]:
            return entry["week"]
    return regular[-1]["week"] + 1


def load_notes(season: int, week: int) -> list[dict]:
    """data/notas.csv: ajuste manual en puntos y notas por juego (lesiones, clima, motivación)."""
    if not config.NOTES_PATH.exists():
        return []
    rows = []
    with open(config.NOTES_PATH, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                if int(raw["season"]) != season or int(raw["week"]) != week:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "away": (raw.get("away") or "").strip(),
                    "home": (raw.get("home") or "").strip(),
                    "ajuste_margen_local": _points(raw.get("ajuste_margen_local")),
                    "ajuste_total": _points(raw.get("ajuste_total")),
                    "notas": (raw.get("notas") or "").strip(),
                }
            )
    return rows


def _points(value) -> float:
    """Ajuste en puntos escrito a mano; vacío o ilegible cuenta como 0."""
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def load_parlays(season: int, week: int) -> dict[str, list[str]]:
    """data/parlays.csv: parlays guardados (una fila por pierna de moneyline)."""
    path = config.DATA_DIR / "parlays.csv"
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                if int(raw["season"]) != season or int(raw["week"]) != week:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            if (raw.get("equipo") or "").strip():
                out.setdefault((raw.get("nombre") or "Parlay").strip(), []).append(
                    raw["equipo"].strip()
                )
    return out


def _records(games: pd.DataFrame, season: int) -> dict[str, str]:
    done = games[(games["season"] == season) & games["completed"]]
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for row in done.itertuples():
        home_won = row.home_points > row.away_points
        for tid, won in ((row.home_id, home_won), (row.away_id, not home_won)):
            (wins if won else losses)[tid] = (wins if won else losses).get(tid, 0) + 1
    return {tid: f"{wins.get(tid, 0)}-{losses.get(tid, 0)}" for tid in set(wins) | set(losses)}


# ---------------------------------------------------------------- backtest.json
def build_backtest(games: pd.DataFrame, season: int, rules: Rules, log=print) -> tuple[dict, dict]:
    """Corre el walk-forward de todos los modelos y el backtest de apuestas del seleccionado.
    Devuelve (backtest.json, contexto para los picks)."""
    params = load_params()
    seasons = make_seasons(games)
    years = sorted(seasons)
    eval_years = years[1:] if len(years) > 1 else years  # la primera temporada solo deja el prior
    history_years = [y for y in eval_years if y < season]
    models = build_models(params)
    chosen = selected_name(params)
    if chosen not in models:
        chosen = "srs"

    comparison = []
    frames = {}
    for name, model in models.items():
        wf = walk_forward(model, seasons, eval_years)
        frames[name] = wf
        hist = fbs_only(wf[wf["season"] < season])
        cur = wf[wf["season"] == season]
        sigma_hist = (
            residual_sigma(hist)[0] if len(hist) else None
        )  # sigma conocida ANTES de la temporada
        comparison.append(
            {
                "modelo": name,
                "label": model.label,
                "seleccionado": name == chosen,
                "actual": metrics(fbs_only(cur), sigma_hist),
                "actual_con_fcs": metrics(cur, sigma_hist),
                "historia": metrics(hist) if len(hist) else {"n": 0},
            }
        )

    wf = frames[chosen]
    pool = fbs_only(wf[wf["week"] >= 2])  # la semana 1 no tiene juegos del año: más error
    sigma_m, sigma_t = residual_sigma(pool)
    cur = fbs_only(wf[wf["season"] == season])
    hist = fbs_only(wf[wf["season"] < season])
    cur_metrics = metrics(cur, sigma_m)
    cal_cur, cal_hist = calibration(cur, sigma_m), calibration(hist, sigma_m)
    cal_error = (
        (
            sum(abs(b["predicho"] - b["observado"]) * b["n"] for b in cal_hist)
            / max(sum(b["n"] for b in cal_hist), 1)
        )
        if cal_hist
        else None
    )

    # ---- apuestas simuladas contra líneas de cierre
    odds = load_odds_history()
    odds_seasons = sorted(int(s) for s in odds.loc[odds["has_odds"], "season"].unique())
    bets = (
        simulate_bets(models[chosen], seasons, odds, sigma_m, sigma_t, rules, odds_seasons)
        if odds_seasons
        else pd.DataFrame()
    )

    def summary(frame: pd.DataFrame) -> dict:
        out = {}
        if frame.empty:
            return out
        for tipo in ("ml", "spread", "total"):
            out[tipo] = {
                label: record(frame[(frame["tipo"] == tipo) & (frame["etiqueta"] == label)])
                for label in ("JUGADA", "EVITAR", "PASAR")
            }
        out["todas"] = {
            label: record(frame[frame["etiqueta"] == label])
            for label in ("JUGADA", "EVITAR", "PASAR")
        }
        return out

    cur_bets = bets[bets["season"] == season] if len(bets) else bets
    weekly = []
    cumulative = 0.0
    for row in by_week(cur, sigma_m):
        wk = row["semana"]
        plays = (
            cur_bets[(cur_bets["week"] == wk) & (cur_bets["etiqueta"] == "JUGADA")]
            if len(cur_bets)
            else cur_bets
        )
        rec = (
            record(plays)
            if len(cur_bets)
            else record(pd.DataFrame(columns=["resultado", "profit"]))
        )
        cumulative += rec["unidades"]
        weekly.append(
            {
                "semana": wk,
                "n": row["n"],
                "acierto": row["acierto"],
                "mae_margen": row["mae_margen"],
                "mae_total": row["mae_total"],
                "jugadas": rec["n"],
                "ganadas": rec["ganadas"],
                "perdidas": rec["perdidas"],
                "unidades": rec["unidades"],
                "acumulado": cumulative,
            }
        )

    market = None
    if len(cur_bets):
        with_line = cur.merge(
            odds[["game_id", "spread_home", "total"]].rename(columns={"total": "linea_total"}),
            on="game_id",
        )
        with_line = with_line.dropna(subset=["spread_home"])
        if len(with_line):
            mkt_margin = -with_line["spread_home"]
            decided = mkt_margin != 0
            market = {
                "n": len(with_line),
                "mae_margen": float((mkt_margin - with_line["margin"]).abs().mean()),
                "acierto": float(((mkt_margin > 0) == (with_line["margin"] > 0))[decided].mean()),
                "mae_total": float((with_line["linea_total"] - with_line["total"]).abs().mean()),
                "modelo_mae_margen": float(
                    (with_line["pred_margin"] - with_line["margin"]).abs().mean()
                ),
                "modelo_mae_total": float(
                    (with_line["pred_total"] - with_line["total"]).abs().mean()
                ),
            }

    backtest = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temporada": season,
        "seleccionado": chosen,
        "seleccionado_label": models[chosen].label,
        "calibrado_con": params.get("calibrado_con"),
        "temporadas_historia": history_years,
        "comparacion": comparison,
        "mercado": market,
        "semanas": weekly,
        "calibracion": {"actual": cal_cur, "historia": cal_hist, "error_medio": cal_error},
        "sigma": {"margen": sigma_m, "total": sigma_t, "n": len(pool)},
        "apuestas": {
            "actual": summary(cur_bets),
            "historia": summary(bets),
            "temporadas_con_lineas": odds_seasons,
            "nota": "Unidad plana (1u por jugada) contra la línea de cierre de DraftKings/consenso vía ESPN.",
        },
        "reglas": rules.as_dict(),
        "referencia_prototipo": PROTOTYPE_REFERENCE,
    }
    context = {
        "model": models[chosen],
        "seasons": seasons,
        "sigma_m": sigma_m,
        "sigma_t": sigma_t,
        "metrics": cur_metrics,
        "cal_error": cal_error,
        "evidence": summary(bets),
        "odds_seasons": odds_seasons,
    }
    log(f"Modelo: {models[chosen].label} · sigma margen {sigma_m:.1f} · sigma total {sigma_t:.1f}")
    return backtest, context


# ------------------------------------------------------------------ picks.json
def build_picks(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    season: int,
    week: int,
    context: dict,
    rules: Rules,
    client: EspnClient,
    log=print,
) -> dict:
    model, seasons = context["model"], context["seasons"]
    sigma_m, sigma_t = context["sigma_m"], context["sigma_t"]
    sa = seasons[season]
    warnings: list[str] = []

    # Estado del modelo con TODO lo jugado hasta hoy (incluye juegos entre semana de esta misma semana).
    prior = _season_prior(model, seasons, season)
    train = sa.completed.copy()
    state = model.fit(sa, train, prior, now_week=week)
    rating = getattr(state, "rating", None)
    if rating is None and hasattr(state, "srs"):
        rating = state.srs.rating
    rank_of = {}
    if rating is not None:
        order = np.argsort(-rating[: sa.n - 1])
        rank_of = {int(i): pos + 1 for pos, i in enumerate(order)}

    # Odds de ESPN en vivo para la semana.
    n_regular = int(
        games.loc[(games["season"] == season) & (games["season_type"] == 2), "week"].max()
    )
    espn_odds: dict[str, dict] = {}
    live_status: dict[str, str] = {}
    try:
        if week <= n_regular:
            payloads = [client.scoreboard_week(season, week, max_age=300)]
        else:
            first = client.scoreboard_week(season, 1)
            payloads = [
                client.scoreboard_week(season, e["week"], seasontype=3, max_age=300)
                for e in parse_calendar(first)["postseason"]
            ]
        for payload in payloads:
            for event in payload.get("events") or []:
                row = parse_event(event)
                if row is None:
                    continue
                live_status[row["game_id"]] = row["status"]
                odds = parse_event_odds(event)
                if odds:
                    espn_odds[row["game_id"]] = odds
    except EspnError as exc:
        warnings.append(
            f"ESPN no devolvió las odds de la semana ({exc}); se usan solo las líneas manuales"
        )

    week_games = games[(games["season"] == season) & (games["week_index"] == week)].sort_values(
        ["date", "game_id"]
    )
    manual_rows = load_manual_odds(season=season, week=week)
    manual_by_game, manual_warnings = match_manual(manual_rows, week_games, teams)
    warnings.extend(manual_warnings)

    lookup = team_lookup(teams)
    notes_by_game: dict[str, dict] = {}
    pair_to_game = {(g.away_id, g.home_id): g.game_id for g in week_games.itertuples()}
    for note in load_notes(season, week):
        gid = pair_to_game.get(
            (lookup.get(normalize_team(note["away"])), lookup.get(normalize_team(note["home"])))
        )
        if gid is None:
            warnings.append(
                f"notas.csv: no encuentro {note['away']} @ {note['home']} en la semana {week}"
            )
        else:
            notes_by_game[gid] = note

    team_info = teams.set_index("team_id").to_dict("index")
    records = _records(games, season)

    def team_block(tid: str, name: str, abbr: str, rank, idx: int) -> dict:
        info = team_info.get(tid, {})
        is_fbs = idx != sa.fcs_index
        return {
            "id": tid,
            "nombre": name,
            "abbr": abbr,
            "color": info.get("color") or "",
            "alt_color": info.get("alt_color") or "",
            "rank": None if pd.isna(rank) else int(rank),
            "record": records.get(tid),
            "fbs": is_fbs,
            "rating": float(rating[idx]) if rating is not None and is_fbs else None,
            "rating_rank": rank_of.get(idx) if is_fbs else None,
            "juegos": int(state.games[idx]) if is_fbs else None,
        }

    juegos = []
    for g in week_games.itertuples():
        status = live_status.get(g.game_id, g.status)
        if status == "final":
            continue  # ya jugado: aparecerá en el track record
        h, a = sa.idx(g.home_id), sa.idx(g.away_id)
        manual = manual_by_game.get(g.game_id)
        espn = espn_odds.get(g.game_id)
        note = notes_by_game.get(g.game_id, {})
        base = {
            "id": g.game_id,
            "fecha": g.date.isoformat().replace("+00:00", "Z"),
            "estado": status,
            "neutral": bool(g.neutral_site),
            "en_mi_lista": manual is not None,
            "visita": team_block(g.away_id, g.away_team, g.away_abbr, g.away_rank, a),
            "local": team_block(g.home_id, g.home_team, g.home_abbr, g.home_rank, h),
            "notas": note.get("notas", ""),
        }
        if status not in ("scheduled",):
            juegos.append(
                {
                    **base,
                    "sin_datos": True,
                    "motivo_sin_datos": {
                        "in_progress": "ya empezó",
                        "postponed": "pospuesto",
                        "canceled": "cancelado",
                    }.get(status, "sin datos del API"),
                }
            )
            continue

        margin, total = model.predict(state, [h], [a], [bool(g.neutral_site)])
        alt_state = model.fit(sa, drop_last_games(sa, train, (h, a)), prior, now_week=week)
        margin_alt, total_alt = model.predict(alt_state, [h], [a], [bool(g.neutral_site)])
        adj_m, adj_t = (
            float(note.get("ajuste_margen_local", 0.0)),
            float(note.get("ajuste_total", 0.0)),
        )
        margin_f, margin_alt_f = float(margin[0]) + adj_m, float(margin_alt[0]) + adj_m
        has_total = not math.isnan(float(total[0]))
        total_f = float(total[0]) + adj_t if has_total else None
        total_alt_f = float(total_alt[0]) + adj_t if has_total else None

        markets = build_markets(manual, espn)
        fcs_game = bool(g.fcs_game)
        ev = evaluate_game(
            margin=margin_f,
            total=total_f,
            margin_alt=margin_alt_f,
            total_alt=total_alt_f,
            markets=markets,
            sigma_m=sigma_m,
            sigma_t=sigma_t,
            rules=rules,
            min_games_played=int(min(state.games[h], state.games[a])) if not fcs_game else 99,
        )
        alerts: list[str] = []
        if fcs_game:
            alerts.append("rival FCS: el modelo usa el promedio de FCS, proyección poco fiable")
            for tipo in ("ml", "spread", "total"):
                for side in ev[tipo].values():
                    if isinstance(side, dict) and side.get("etiqueta") == "JUGADA":
                        side.update(
                            {
                                "etiqueta": "EVITAR",
                                "stake": 0.0,
                                "kelly": 0.0,
                                "motivos": ["rival FCS: proyección poco fiable", *side["motivos"]],
                            }
                        )
        if abs(margin_f - margin_alt_f) >= 4.0:
            alerts.append(
                f"modelo inestable: la proyección cambia {abs(margin_f - margin_alt_f):.1f} pts al quitar los últimos 2 partidos de cada equipo"
            )
        if ev["mov_spread"] is not None and abs(ev["mov_spread"]) >= rules.move_spread:
            toward = g.home_team if ev["mov_spread"] > 0 else g.away_team
            alerts.append(
                f"la línea se movió {abs(ev['mov_spread']):.1f} pts hacia {toward} desde la apertura"
            )
        if ev["mov_total"] is not None and abs(ev["mov_total"]) >= rules.move_total:
            alerts.append(
                f"el total {'subió' if ev['mov_total'] > 0 else 'bajó'} {abs(ev['mov_total']):.1f} pts desde la apertura (¿clima, lesiones?)"
            )
        if ev["desacuerdo_margen"] is not None and abs(ev["desacuerdo_margen"]) >= rules.gap_margin:
            alerts.append(
                f"el modelo difiere {abs(ev['desacuerdo_margen']):.1f} pts del spread del mercado"
            )

        home_pts, away_pts = _scoreline(margin_f, total_f)
        p_home = float(norm_cdf(margin_f / sigma_m))
        espn_ref = None
        if espn:
            espn_ref = {
                "proveedor": espn.get("provider"),
                "spread_local": espn.get("spread_home"),
                "total": espn.get("total"),
                "ml_local": espn.get("ml_home"),
                "ml_visita": espn.get("ml_away"),
                "spread_apertura": espn.get("spread_home_open"),
                "total_apertura": espn.get("total_open"),
            }
        juegos.append(
            {
                **base,
                "sin_datos": False,
                "fcs": fcs_game,
                "proyeccion": {
                    "local": home_pts,
                    "visita": away_pts,
                    "margen": margin_f,
                    "total": total_f,
                    "p_local": p_home,
                    "p_visita": 1 - p_home,
                    "ganador": "local" if margin_f >= 0 else "visita",
                    "momio_justo_local": prob_to_american(p_home),
                    "momio_justo_visita": prob_to_american(1 - p_home),
                    "margen_sin_ultimos2": margin_alt_f,
                    "total_sin_ultimos2": total_alt_f,
                    "ajuste_margen_local": adj_m,
                    "ajuste_total": adj_t,
                },
                "mercado": {
                    "margen": ev["margen_mercado"],
                    "desacuerdo_margen": ev["desacuerdo_margen"],
                    "desacuerdo_total": ev["desacuerdo_total"],
                    "mov_spread": ev["mov_spread"],
                    "mov_total": ev["mov_total"],
                    "espn": espn_ref,
                },
                "ml": {**ev["ml"], "sin_vig": markets["ml"]["sin_vig"]},
                "spread": {
                    **ev["spread"],
                    "linea_local": markets["spread"]["linea_local"],
                    "fuente_linea": markets["spread"]["fuente_linea"],
                    "sin_vig": markets["spread"]["sin_vig"],
                },
                "total": {
                    **ev["total"],
                    "linea": markets["total"]["linea"],
                    "fuente_linea": markets["total"]["fuente_linea"],
                    "sin_vig": markets["total"]["sin_vig"],
                },
                "comparacion": compare_sources(manual, espn),
                "alertas": alerts,
            }
        )

    parlays = []
    for name, legs in load_parlays(season, week).items():
        parlay, parlay_warnings = evaluate_parlay(name, legs, juegos, lookup)
        warnings.extend(parlay_warnings)
        parlays.append(parlay)

    met = context["metrics"]
    return {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "temporada": season,
        "semana": week,
        "semana_label": f"Semana {week}" if week <= n_regular else "Postemporada",
        "modelo": {
            "nombre": model.name,
            "label": model.label,
            "sigma_margen": sigma_m,
            "sigma_total": sigma_t,
            "acierto": met.get("acierto"),
            "acierto_14": met.get("acierto_14"),
            "mae_margen": met.get("mae_margen"),
            "mae_total": met.get("mae_total"),
            "n_backtest": met.get("n"),
            "error_calibracion": context["cal_error"],
            "juegos_entrenamiento": int(train.sum()),
        },
        "reglas": rules.as_dict(),
        "evidencia": {"temporadas": context["odds_seasons"], **context["evidence"]},
        "avisos": warnings,
        "juegos": juegos,
        "parlays": parlays,
    }


def _season_prior(model, seasons: dict, season: int) -> dict:
    """Recorre las temporadas anteriores para dejar el prior de la actual."""
    prev_state = prev_sa = None
    for year in sorted(y for y in seasons if y < season):
        sa = seasons[year]
        prior = model.prior(prev_state, prev_sa, sa)
        if sa.completed.any():
            prev_state = model.fit(
                sa, np.ones(len(sa.week), dtype=bool), prior, now_week=int(sa.week.max()) + 1
            )
            prev_sa = sa
    return model.prior(prev_state, prev_sa, seasons[season])


def _scoreline(margin: float, total: float | None) -> tuple[float | None, float | None]:
    """Marcador proyectado a partir de margen y total, sin dejar a nadie por debajo de un piso."""
    if total is None:
        return None, None
    home, away = (total + margin) / 2.0, (total - margin) / 2.0
    if away < MIN_POINTS:
        away, home = MIN_POINTS, MIN_POINTS + margin
    elif home < MIN_POINTS:
        home, away = MIN_POINTS, MIN_POINTS - margin
    return home, away


def evaluate_parlay(
    name: str, legs: list[str], juegos: list[dict], lookup: dict[str, str]
) -> tuple[dict, list[str]]:
    """Parlay de moneylines: probabilidad del modelo, del mercado sin vig y lo que paga la casa."""
    warnings: list[str] = []
    by_team: dict[str, tuple[dict, str]] = {}
    for game in juegos:
        if game.get("sin_datos"):
            continue
        by_team[game["local"]["id"]] = (game, "local")
        by_team[game["visita"]["id"]] = (game, "visita")
    piernas = []
    for team in legs:
        tid = lookup.get(normalize_team(team))
        if tid is None or tid not in by_team:
            warnings.append(
                f"parlay '{name}': no encuentro un juego pendiente de '{team}' esta semana"
            )
            piernas.append({"equipo": team, "sin_datos": True})
            continue
        game, lado = by_team[tid]
        side = game["ml"][lado]
        rival = game["visita" if lado == "local" else "local"]["nombre"]
        piernas.append(
            {
                "equipo": game[lado]["nombre"],
                "juego_id": game["id"],
                "lado": lado,
                "rival": rival,
                "sede": "local" if lado == "local" else "visita",
                "p_modelo": side["p_modelo"],
                "p_mercado": side["p_mercado"],
                "momio": side["momio"],
                "fuente": side["fuente"],
                "edge": side["edge"],
                "etiqueta": side["etiqueta"],
                "sin_datos": False,
            }
        )
    ok = [p for p in piernas if not p.get("sin_datos")]
    complete = len(ok) == len(piernas) and len(ok) > 0
    out = {"nombre": name, "piernas": piernas, "completo": complete}
    if not ok:
        return out, warnings
    p_model = float(np.prod([p["p_modelo"] for p in ok]))
    out["p_modelo"] = p_model
    if all(p["p_mercado"] is not None for p in ok):
        out["p_mercado"] = float(np.prod([p["p_mercado"] for p in ok]))
    if all(p["momio"] is not None for p in ok):
        decimal = float(np.prod([american_to_decimal(p["momio"]) for p in ok]))
        out.update(
            {
                "decimal": decimal,
                "prob_casa": 1.0 / decimal,
                "ev_modelo": p_model * decimal - 1.0,
                "ev_mercado": out["p_mercado"] * decimal - 1.0 if "p_mercado" in out else None,
            }
        )
    weakest = min(ok, key=lambda p: p["p_modelo"])
    out["mas_debil"] = weakest["equipo"]
    with_edge = [p for p in ok if p["edge"] is not None]
    if with_edge:
        out["peor_valor"] = min(with_edge, key=lambda p: p["edge"])["equipo"]
    return out, warnings


# ------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Genera los picks de la semana y exporta los JSON del sitio"
    )
    parser.add_argument(
        "--week", type=int, default=None, help="semana (por defecto, la que está en curso)"
    )
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="no volver a descargar juegos ni líneas históricas",
    )
    parser.add_argument("--offline", action="store_true", help="usar solo la caché")
    args = parser.parse_args(argv)

    rules = Rules()
    client = EspnClient(offline=args.offline)
    try:
        if not args.skip_build and not args.offline:
            build_mod.build(args.season, [y for y in config.HISTORY_SEASONS if y < args.season])
            fetch_odds_history(args.season)
        games = build_mod.load_games()
        teams = build_mod.load_teams()
        week = args.week
        if week is None:
            calendar = parse_calendar(client.scoreboard_week(args.season, 1))
            week = week_for(calendar, datetime.now(timezone.utc))
        backtest, context = build_backtest(games, args.season, rules)
        picks = build_picks(games, teams, args.season, week, context, rules, client)
    except EspnError as exc:
        print(f"ERROR: el API de ESPN falló y no se inventan datos: {exc}", file=sys.stderr)
        return 1

    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "picks.json").write_text(
        json.dumps(_clean(picks), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (config.SITE_DATA_DIR / "backtest.json").write_text(
        json.dumps(_clean(backtest), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print_summary(picks)
    return 0


def print_summary(picks: dict) -> None:
    juegos = [g for g in picks["juegos"] if not g.get("sin_datos")]
    print(
        f"\n{picks['semana_label']} {picks['temporada']}: {len(juegos)} juegos con proyección, "
        f"{sum(g['en_mi_lista'] for g in juegos)} con tus líneas"
    )
    plays = []
    for g in juegos:
        for tipo, keys in (
            ("ml", ("local", "visita")),
            ("spread", ("local", "visita")),
            ("total", ("over", "under")),
        ):
            for key in keys:
                side = g[tipo].get(key)
                if side and side["etiqueta"] in ("JUGADA", "EVITAR"):
                    plays.append((side["etiqueta"], side["edge"], tipo, key, g))
    for label in ("JUGADA", "EVITAR"):
        rows = sorted((p for p in plays if p[0] == label), key=lambda p: -p[1])
        print(f"\n{label} ({len(rows)}):")
        for _, edge, tipo, key, g in rows[:25]:
            side = g[tipo][key]
            name = {
                "local": g["local"]["nombre"],
                "visita": g["visita"]["nombre"],
                "over": "Over",
                "under": "Under",
            }[key]
            line = ""
            if tipo == "spread":
                home_line = g["spread"]["linea_local"]
                line = f" {(home_line if key == 'local' else -home_line):+g}"
            elif tipo == "total":
                line = f" {g['total']['linea']:g}"
            print(
                f"  {tipo:6s} {name}{line} ({side['momio']:+d}) · modelo {side['p_modelo']:.0%} vs mercado {side['p_mercado']:.0%} "
                f"· edge {edge:+.1%} · {g['visita']['abbr']} @ {g['local']['abbr']}"
                + (f" · {side['motivos'][0]}" if side["motivos"] else "")
            )
    for parlay in picks["parlays"]:
        if "p_modelo" in parlay:
            print(
                f"\nParlay '{parlay['nombre']}': modelo {parlay['p_modelo']:.1%}"
                + (
                    f" · mercado sin vig {parlay['p_mercado']:.1%}"
                    if parlay.get("p_mercado")
                    else ""
                )
                + (
                    f" · la casa paga como {parlay['prob_casa']:.1%}"
                    if parlay.get("prob_casa")
                    else ""
                )
                + f" · pierna más débil: {parlay['mas_debil']}"
            )
    for warning in picks["avisos"]:
        print(f"AVISO: {warning}")
    print(
        f"\nExportado: {(config.SITE_DATA_DIR / 'picks.json').relative_to(config.ROOT)} y backtest.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
