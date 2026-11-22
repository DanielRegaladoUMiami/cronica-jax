# CFB Picks (and NFL)

A weekly football prediction system for two leagues. It downloads every game
from ESPN's public API, fits a ratings model, checks it with a walk-forward
backtest, and publishes the picks as a static site.

- **College football**: every FBS game. The rest of this page describes it.
- **NFL**: the same pipeline with `--league nfl`. See [NFL](#nfl).

**Live site:** https://danielregaladoumiami.github.io/cronica-jax/ (the switch
at the top changes league; `#nfl-picks` links straight to the NFL).

## College football: how good it is, and how good it is not

Walk-forward backtest of the 2026 season through week 12: 695 FBS vs FBS games
the model never saw while it was being tuned.

| Model | Winner | When favored by 14+ | Margin miss | Total miss |
|---|---|---|---|---|
| Baseline (points vs opponent win %) | 66.8% | 79.7% | 14.46 | 12.86 |
| **SRS (in use)** | **70.7%** | **90.4%** | **12.77** | **12.83** |
| Elo with margin of victory | 69.5% | 83.4% | 13.11 | n/a |
| Market closing line | 74.5% | n/a | 11.70 | 12.25 |
| Original prototype (reference) | 73% | 92% | ~15.5 | ~20 |

Counting games against FCS teams, as the prototype did, SRS picks 74.5% of
winners and 92.4% when it favors a team by 14 or more.

Two things to keep in mind:

- **The percentages are honest.** In 2022-2025 the chance the model gave its
  favorite was off by 2 points on average.
- **The model does not beat the market.** Applying the PLAY rule to 1,817 past
  bets against closing lines (2021, 2022 and 2026) went 832-974 for an ROI of
  -5.8% ± 2.6%, which is roughly the bookmaker's cut. Picks flagged AVOID did
  worse on spreads (-13.7%). Read a PLAY as "my number disagrees with the
  market", not as a bet with a proven edge.

These numbers are recomputed on every run and shown on the site's *Track
record* tab.

## Run it locally

```bash
cd cfb-picks
pip install -r requirements.txt

python -m src.picks --week 12     # download data, run the backtest, generate the picks
python -m src.picks --league nfl  # the same for the NFL, current week
python -m http.server -d site     # then open http://localhost:8000
```

`python -m src.picks` without `--week` uses the current week, or the next one
once every game of the current week is final. `--skip-build` skips the
downloads, which is handy after editing odds or notes. Every command below
takes `--league nfl` too.

Other commands:

| Command | What it does |
|---|---|
| `python -m src.data.build` | Only download games into `data/games.parquet` (`data/nfl_games.parquet`) |
| `python -m src.data.odds_history` | Only download closing lines for games already played (`--season` picks the year) |
| `python -m src.compare` | Model comparison table (`--select` saves the lowest-error model) |
| `python -m src.model.tune` | Re-tune hyperparameters on past seasons |
| `python -m pytest` | Tests |

## Updating odds by hand

Edit `data/odds_manual.csv`, one row per game. Leave blank whatever you do not
have.

```csv
season,week,away,home,ml_away,ml_home,spread_away,spread_away_odds,spread_home_odds,total,over_odds,under_odds,book,captured_at
2026,12,Indiana,Washington,-300,+240,-7.5,-110,,43.5,-115,-105,my book,2026-11-20
2026,12,Clemson,Duke,,-475,,,,,,,my book,2026-11-20
```

- `spread_away` is the **away** team's line (`-7.5` means the away team is favored).
- Write teams the way ESPN does (`Miami (FL)`, `Pitt` and `Hawaii` also work).
  If a name is not recognized, the row is ignored and a warning shows up on the
  site. Nothing is guessed.
- Your lines and prices come first. Anything missing is filled in from ESPN
  (DraftKings) and labeled with its source. A spread or total price that is not
  available anywhere is assumed to be -110 and marked as assumed.
- The no-vig market probability uses both sides of one book: yours if you
  captured both, otherwise ESPN's.

Then run `python -m src.picks --week 12 --skip-build`.

## Notes and manual adjustments

The model knows nothing about injuries, weather or motivation. In
`data/notes.csv` you can shift a projection and leave a note per game (this
example is made up):

```csv
season,week,away,home,home_margin_adj,total_adj,notes
2026,12,LSU,Tennessee,-3,-6,Home starting QB doubtful; 25 mph wind
```

`home_margin_adj` adds points to the home team's margin (negative favors the
away team). `total_adj` adds points to the total. Notes are published on the
site as written.

`data/parlays.csv` holds saved moneyline parlays (one row per leg) that the
site evaluates every week.

## How it works

1. **Data** (`src/data/`): ESPN's scoreboard, week by week, with a disk cache.
   Weeks already played are never requested again. FBS membership comes from
   each season's standings; every other team counts as FCS.
2. **Models** (`src/model/`): three candidates.
   - `baseline.py`: points scored and allowed as a function of the opponent's
     win percentage. It is a reconstruction of the prototype from its
     description.
   - `srs.py`: schedule-adjusted ratings (a team's rating is its average margin
     plus its opponents' average rating), with a cap on blowouts, shrinkage
     toward last season's rating and an estimated home-field advantage. The
     same system run on totals gives the projected score.
   - `elo.py`: Elo with a margin-of-victory multiplier, carried across seasons.
3. **Backtest** (`src/model/backtest.py`): to predict week W, only earlier
   weeks are used. Hyperparameters are tuned on 2022-2025 (`tune.py`) and 2026
   is scored without touching them.
4. **Probabilities** (`src/model/prob.py`): a normal distribution with the
   real standard deviation from the backtest (15.8 points for the margin, 16.1
   for the total).
5. **Rules** (`src/model/rules.py`):
   - **PLAY**: edge of at least 5%, positive expected value at the real price,
     and no red flags.
   - **AVOID**: there is an edge on paper, but the pick flips when each team's
     last 2 games are removed, the model is 5+ points away from the market
     spread (7+ on a total), or the line moved 3+ points against the pick since
     it opened (4+ on a total).
   - **PASS**: not enough edge.
   - Stake: quarter Kelly, capped at 2% of bankroll.
6. **Site** (`site/`): plain HTML and JavaScript that read
   `site/data/picks.json` and `site/data/backtest.json`. No backend.

## NFL

The same code runs the NFL: `src/config.py` describes each league (where ESPN
keeps it, how its files are named, how its postseason works) and everything
else takes `--league nfl`. Its files carry an `nfl_` prefix:
`data/nfl_games.parquet`, `data/nfl_odds_manual.csv`, `data/nfl_notes.csv`,
`data/nfl_parlays.csv`, `src/model/nfl_params.json` and
`site/data/nfl_picks.json`. The manual files use the same columns as the
college ones.

### How good it is, and how good it is not

Walk-forward backtest of the 2026 season through week 10 plus Thursday of week
11: 151 games the model never saw while it was being tuned (2022-2025).

| Model | Winner | When favored by 7+ | Margin miss | Total miss |
|---|---|---|---|---|
| Baseline (points vs opponent win %) | 61.3% | 69.0% | 10.56 | 11.96 |
| **SRS (in use)** | **63.3%** | **85.2%** | **9.90** | **11.17** |
| Elo with margin of victory | 63.3% | 88.5% | 9.93 | n/a |
| SRS + Elo blend | 62.7% | 87.0% | 9.91 | 11.17 |
| Market closing line | 68.2% | n/a | 9.55 | 10.40 |

- **The three real models are tied** (the gaps are far smaller than the noise
  of 151 games). SRS is used because it also projects totals.
- **The model does not beat the market, and here it loses more than the
  vig.** One unit on every PLAY against the closing line went 264-356, ROI
  -7.7% ± 4.7%, over 625 bets. Picks flagged AVOID returned -14.5% ± 3.8% over
  885. The NFL market is the sharpest there is, and the model does not know
  who is playing quarterback.
- ESPN only keeps closing lines for 2021, 2022, 2026 and parts of 2023 and
  2025, so that record covers about 845 games.

### What is different from college football

- **No lower division.** Every team with a regular-season game is in the
  league; there is no FCS pool.
- **Teams go by nickname** (`Bills`, `Giants`), because cities repeat. In the
  manual files the nickname, the full name, the abbreviation and a city that
  only one team has (`Kansas City`, not `New York`) all work.
- **Each playoff round is its own week**: 19 Wild Card, 20 Divisional, 21
  Conference Championships, 22 Super Bowl. The Pro Bowl is skipped.
- **Ties exist.** A tie is left out of the winner percentage, shows in a
  record as `5-3-1`, and refunds a moneyline bet.
- **Tighter red flags.** A pick is AVOID when the model is 4+ points from the
  market spread or total, or the line moved 1+ point against it since it
  opened (3+ on a total). They were set on the betting record above, so the
  gap between PLAY and AVOID is a best case.
- **"Clear favorite" means 7+ points**, not 14.

## Publishing

The site is served by GitHub Pages from the `gh-pages` branch, which holds
nothing but the contents of `site/`.

The workflow `.github/workflows/cfb-picks.yml` reruns both pipelines every
Thursday, Saturday and Sunday from August through February and replaces that
branch with the new build. It is scheduled twice on those days (1:17 and 8:17
AM Eastern in winter) because GitHub can start a scheduled run hours late; the
header of the site shows when the picks were last updated. College football is generated from
August to January and the NFL from September to February; out of season a
league keeps the picks already committed. GitHub only runs schedules from the
default branch, so the workflow has to be on `main` for the weekly updates to
happen. It can also be started by hand from **Actions → CFB Picks → Run
workflow**.

If the site ever stops being served, check **Settings → Pages** and make sure
the source is *Deploy from a branch*, branch `gh-pages`, folder `/ (root)`.
GitHub pauses scheduled workflows after 60 days without repository activity;
they can be re-enabled from the Actions tab. If ESPN's API fails, the workflow
fails and the site keeps its previous version: no data is made up.

## Known limits

- It only uses final scores. It does not see plays, yards, injuries or weather.
  In the NFL that includes the one thing that moves a line most: who is at
  quarterback.
- FCS opponents are treated as one average team.
- The simulated ROI uses the closing line, not the line you see on Thursday.
- ESPN's historical closing lines are only complete for 2021, 2022 and 2026,
  so the betting sample covers three seasons (plus parts of two more for the
  NFL).
