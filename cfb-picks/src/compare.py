"""Tabla comparativa de modelos (backtest walk-forward) y selección del mejor.

Uso:
    python -m src.compare             # imprime la tabla
    python -m src.compare --select    # además guarda el de menor error en params.json

La selección usa el error absoluto medio del margen en la temporada actual
(FBS vs FBS), que los modelos nunca vieron al calibrarse.
"""

from __future__ import annotations

import argparse
import json

from . import config
from .data.build import load_games
from .model.registry import load_params
from .model.rules import Rules
from .picks import build_backtest


def _fmt(value, pct=False, nd=2) -> str:
    if value is None:
        return "  -  "
    return f"{value:.1%}" if pct else f"{value:.{nd}f}"


def print_table(backtest: dict) -> None:
    season = backtest["temporada"]
    for key, title in (
        ("actual", f"{season} · FBS vs FBS (fuera de muestra)"),
        ("actual_con_fcs", f"{season} · todos los juegos (incluye FCS)"),
        ("historia", f"{backtest.get('calibrado_con')} · FBS vs FBS (usado para calibrar)"),
    ):
        print(f"\n{title}")
        print(
            f"{'modelo':46s} {'n':>5s} {'acierto':>8s} {'14+ pts':>8s} {'MAE margen':>11s} {'MAE total':>10s} {'Brier':>7s}"
        )
        for row in backtest["comparacion"]:
            m = row[key]
            if not m.get("n"):
                continue
            mark = " <-" if row["seleccionado"] else ""
            print(
                f"{row['label'][:46]:46s} {m['n']:5d} {_fmt(m['acierto'], True):>8s} {_fmt(m['acierto_14'], True):>8s} "
                f"{_fmt(m['mae_margen']):>11s} {_fmt(m['mae_total']):>10s} {_fmt(m['brier'], nd=4):>7s}{mark}"
            )
    mkt = backtest.get("mercado")
    if mkt:
        print(
            f"\nReferencia · línea de cierre del mercado ({mkt['n']} juegos de {season}): acierto {mkt['acierto']:.1%}, "
            f"MAE margen {mkt['mae_margen']:.2f} (modelo {mkt['modelo_mae_margen']:.2f}), MAE total {mkt['mae_total']:.2f} (modelo {mkt['modelo_mae_total']:.2f})"
        )
    ref = backtest["referencia_prototipo"]
    print(
        f"Referencia · prototipo: acierto {ref['acierto']:.0%}, {ref['acierto_14']:.0%} con 14+, MAE margen ~{ref['mae_margen']}, MAE total ~{ref['mae_total']}"
    )
    bets = backtest["apuestas"].get("historia") or {}
    if bets:
        print(
            f"\nRegla JUGADA contra líneas de cierre, temporadas {backtest['apuestas']['temporadas_con_lineas']}:"
        )
        for tipo in ("ml", "spread", "total", "todas"):
            for label in ("JUGADA", "EVITAR"):
                r = bets[tipo][label]
                if r["n"]:
                    print(
                        f"  {tipo:6s} {label:7s} {r['ganadas']:4d}-{r['perdidas']:<4d} ({r['pct']:.1%})  {r['unidades']:+7.1f}u  ROI {r['roi']:+.1%} ± {r['roi_error']:.1%}"
                    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara los modelos en el backtest walk-forward")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument(
        "--select", action="store_true", help="guardar el modelo de menor error en params.json"
    )
    args = parser.parse_args(argv)

    backtest, _ = build_backtest(load_games(), args.season, Rules(), log=lambda *_: None)
    if args.select:
        best = min(
            (r for r in backtest["comparacion"] if r["actual"].get("n")),
            key=lambda r: r["actual"]["mae_margen"],
        )
        params = load_params()
        params["seleccionado"] = best["modelo"]
        params["seleccionado_por"] = (
            f"menor MAE de margen en el walk-forward de {args.season} (FBS vs FBS)"
        )
        config.PARAMS_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")
        for row in backtest["comparacion"]:
            row["seleccionado"] = row["modelo"] == best["modelo"]
        print(f"Seleccionado: {best['label']}")
    print_table(backtest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
