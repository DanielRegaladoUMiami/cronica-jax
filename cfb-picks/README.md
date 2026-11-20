# CFB Picks

A weekly college football prediction system. It downloads every FBS game from
ESPN's public API, fits a ratings model, checks it with a walk-forward
backtest, and publishes the picks as a static site.

## How good it is, and how good it is not

Walk-forward backtest of the 2026 season: 635 FBS vs FBS games the model never
saw while it was being tuned.

| Model | Winner | When favored by 14+ | Margin miss | Total miss |
|---|---|---|---|---|
| Baseline (points vs opponent win %) | 65.7% | 79.5% | 14.72 | 13.08 |
| **SRS (in use)** | **70.4%** | **90.6%** | **12.83** | **13.01** |
| Elo with margin of victory | 68.7% | 83.3% | 13.29 | n/a |
| Market closing line | 74.6% | n/a | 11.72 | 12.42 |
| Original prototype (reference) | 73% | 92% | ~15.5 | ~20 |

Counting games against FCS teams, as the prototype did, SRS picks 74.4% of
winners and 92.5% when it favors a team by 14 or more.

Two things to keep in mind:

- **The percentages are honest.** When the model says 70-75%, its favorite
  wins about 73% of the time.
- **The model does not beat the market.** Applying the PLAY rule to 1,764 past
  bets against closing lines (2021, 2022 and 2026) went 809-944 for an ROI of
  -5.4% ± 2.6%, which is roughly the bookmaker's cut. Picks flagged AVOID did
  worse (-13% on spreads). Read a PLAY as "my number disagrees with the
  market", not as a bet with a proven edge.

These numbers are recomputed on every run and shown on the site's *Track
record* tab.

## Run it locally

```bash
cd cfb-picks
pip install -r requirements.txt

python -m src.picks --week 12     # download data, run the backtest, generate the picks
python -m http.server -d site     # then open http://localhost:8000
```

`python -m src.picks` without `--week` uses the current week. `--skip-build`
skips the downloads, which is handy after editing odds or notes.

Other commands:

| Command | What it does |
|---|---|
| `python -m src.data.build` | Only download games into `data/games.parquet` |
| `python -m src.data.odds_history` | Only download closing lines for games already played |
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

## Publishing

The site is served by GitHub Pages from the `gh-pages` branch, which holds
nothing but the contents of `site/`.

The workflow `.github/workflows/cfb-picks.yml` reruns the pipeline every
Thursday and Saturday morning (August through January) and replaces that
branch with the new build. GitHub only runs schedules from the default branch,
so the workflow has to be on `main` for the weekly updates to happen. It can
also be started by hand from **Actions → CFB Picks → Run workflow**.

If the site ever stops being served, check **Settings → Pages** and make sure
the source is *Deploy from a branch*, branch `gh-pages`, folder `/ (root)`.
GitHub pauses scheduled workflows after 60 days without repository activity;
they can be re-enabled from the Actions tab. If ESPN's API fails, the workflow
fails and the site keeps its previous version: no data is made up.

## Known limits

- It only uses final scores. It does not see plays, yards, injuries or weather.
- FCS opponents are treated as one average team.
- The simulated ROI uses the closing line, not the line you see on Thursday.
- ESPN's historical closing lines are only complete for 2021, 2022 and 2026,
  so the betting sample covers three seasons.
