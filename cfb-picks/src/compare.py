"""Model comparison table (walk-forward backtest) and selection of the best one.

Usage:
    python -m src.compare             # print the table
    python -m src.compare --select    # also save the lowest-error model to params.json

Selection uses the mean absolute error of the margin in the current season
(FBS vs FBS), which the models never saw while being tuned.
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
    season = backtest["season"]
    sections = (
        ("current", f"{season} · FBS vs FBS (out of sample)"),
        ("current_with_fcs", f"{season} · all games (FCS included)"),
        ("history", f"{backtest.get('tuned_on')} · FBS vs FBS (used for tuning)"),
    )
    header = (
        f"{'model':40s} {'n':>5s} {'winner':>8s} {'14+ pts':>8s}"
        f" {'margin MAE':>11s} {'total MAE':>10s} {'Brier':>7s}"
    )
    for key, title in sections:
        print(f"\n{title}\n{header}")
        for row in backtest["comparison"]:
            m = row[key]
            if not m.get("n"):
                continue
            mark = " <-" if row["selected"] else ""
            print(
                f"{row['label'][:40]:40s} {m['n']:5d} {_fmt(m['accuracy'], True):>8s}"
                f" {_fmt(m['accuracy_14'], True):>8s} {_fmt(m['mae_margin']):>11s}"
                f" {_fmt(m['mae_total']):>10s} {_fmt(m['brier'], nd=4):>7s}{mark}"
            )
    mkt = backtest.get("market")
    if mkt:
        print(
            f"\nReference · market closing line ({mkt['n']} games in {season}):"
            f" winner {mkt['accuracy']:.1%}, margin MAE {mkt['mae_margin']:.2f}"
            f" (model {mkt['model_mae_margin']:.2f}), total MAE {mkt['mae_total']:.2f}"
            f" (model {mkt['model_mae_total']:.2f})"
        )
    ref = backtest["prototype_reference"]
    print(
        f"Reference · original prototype: winner {ref['accuracy']:.0%},"
        f" {ref['accuracy_14']:.0%} at 14+, margin MAE ~{ref['mae_margin']},"
        f" total MAE ~{ref['mae_total']}"
    )
    bets = backtest["bets"].get("history") or {}
    if bets:
        print(f"\nPLAY rule vs closing lines, seasons {backtest['bets']['seasons_with_lines']}:")
        for market in ("ml", "spread", "total", "all"):
            for label in ("PLAY", "AVOID"):
                r = bets[market][label]
                if r["n"]:
                    print(
                        f"  {market:6s} {label:5s} {r['won']:4d}-{r['lost']:<4d}"
                        f" ({r['win_pct']:.1%})  {r['units']:+7.1f}u"
                        f"  ROI {r['roi']:+.1%} ± {r['roi_error']:.1%}"
                    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare the models in the walk-forward backtest")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument(
        "--select", action="store_true", help="save the lowest-error model to params.json"
    )
    args = parser.parse_args(argv)

    backtest, _ = build_backtest(load_games(), args.season, Rules(), log=lambda *_: None)
    if args.select:
        best = min(
            (r for r in backtest["comparison"] if r["current"].get("n")),
            key=lambda r: r["current"]["mae_margin"],
        )
        params = load_params()
        params["selected"] = best["model"]
        params["selected_by"] = (
            f"lowest margin MAE in the {args.season} walk-forward backtest (FBS vs FBS)"
        )
        config.PARAMS_PATH.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")
        for row in backtest["comparison"]:
            row["selected"] = row["model"] == best["model"]
        print(f"Selected: {best['label']}")
    print_table(backtest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
