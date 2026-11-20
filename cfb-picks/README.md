# CFB Picks (and soccer)

A prediction system for two sports. It downloads every game from ESPN's public
API, fits a ratings model, checks it with a walk-forward backtest, and
publishes the picks as a static site.

- **College football**: every FBS game, week by week.
- **Soccer**: the Premier League, LaLiga, Serie A, the Bundesliga and Ligue 1,
  for the next 7 days. See [Soccer](#soccer).

**Live site:** https://danielregaladoumiami.github.io/cronica-jax/ (the switch
at the top changes sport; `#soccer-picks` links straight to soccer).

## College football: how good it is, and how good it is not

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
| `python -m pytest` | Tests (both sports) |

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
   `site/data/picks.json` and `site/data/backtest.json` (and their `soccer_`
   twins). No backend.

## Soccer

The same idea for the five big European leagues: results from ESPN, a goals
model, a walk-forward backtest, and three markets per match (the three-way
result, the goals total and the handicap).

```bash
python -m src.soccer.picks              # download, backtest, picks for the next 7 days
python -m src.soccer.picks --skip-build # reuse the data on disk
python -m src.soccer.data --odds        # only download matches and closing odds
python -m src.soccer.tune               # re-tune the models on past seasons
```

### How good it is, and how good it is not

Walk-forward backtest of the 2026-27 season so far (503 matches the model never
saw while it was being tuned). RPS is the error of the three percentages: 0 is
perfect, lower is better.

| Forecast | Called the result | RPS | Log loss |
|---|---|---|---|
| League averages for every match | 45.5% | 0.2323 | 1.0613 |
| Baseline (this season's scoring averages) | 51.9% | 0.2067 | 0.9887 |
| **Poisson ratings (in use)** | **56.5%** | **0.1997** | **0.9681** |
| Elo with goal difference | 53.9% | 0.1975 | 0.9588 |
| Market closing odds | 56.0% | 0.1933 | 0.9477 |

- **The percentages are honest**: pooled over home wins, draws and away wins,
  they were off by half a point on average in 2022-2025.
- **The model does not beat the market.** One unit on every PLAY against the
  closing odds, 2022-23 through 2026-27: 4,455 bets, 2,289-2,108, ROI
  -4.8% ± 1.4%, roughly the bookmaker's cut. Picks flagged AVOID returned
  -14.7% ± 2.1%, so the flags do find the worst bets. In 2026-27 alone, which
  the flags were not fitted to: PLAY -5.8% ± 5.2%, AVOID -17.2% ± 6.0%.
- Elo forecasts results slightly better. The Poisson model is used because it
  is the only one that gives a chance for every scoreline, which totals and
  handicaps need.

### How it works

1. **Data** (`src/soccer/data.py`): the monthly scoreboard of each league
   (11 requests per league and season, cached once a month is over) and, for
   matches already played, the closing odds from each match's summary (one
   request per match, cached forever).
2. **Model** (`src/soccer/model.py`): every team gets an attack and a defense
   rating, fitted together per league on about three seasons of matches.
   Recent matches count more (a match's weight halves every 12 months),
   ratings are shrunk toward the league average, and a newly promoted team
   starts below average. Expected goals become a probability for every
   scoreline (Poisson with a low-score correction), and from that grid come
   the result, the totals and the handicaps.
3. **Backtest** (`src/soccer/backtest.py`, `simulate.py`): to predict a week
   (Tuesday to Monday), only earlier matches are used. Settings were tuned on
   2022-2025 by log loss (`tune.py`); 2026-27 is scored without touching them.
4. **Rules** (`src/soccer/rules.py`):
   - **PLAY**: edge of at least 5%, positive expected value, no red flags.
   - **AVOID**: the edge is 12+ points (the market strongly disagrees), it is
     a result bet on a side the market gives under 45%, the pick flips without
     each team's last 2 matches, or a team has fewer than 6 league matches on
     record. Each flag marked picks that lost far more than the vig in
     2022-2025.
   - Line movement since the open is shown on the card but is not a flag: in
     the two seasons with opening prices it did not separate good picks from
     bad ones.

### Notes and manual adjustments

`data/soccer_notes.csv` holds a note per match and, optionally, goals added to
or taken from a team's expected goals (this example is made up):

```csv
home,away,home_goals_adj,away_goals_adj,notes
Arsenal,Chelsea,-0.3,,Top scorer suspended
```

Write teams the way ESPN does (`Man City`, `Manchester City` and `MNC` all
work). A row that matches no team is ignored and a warning shows up on the
site.

### Known limits

- Only final scores: no shots, no expected-goals data, no injuries, no
  suspensions, no rotation before a European night.
- Odds come from ESPN (DraftKings or ESPN BET). There is no manual odds file
  for soccer yet, so a match ESPN has no price for shows the model's number
  only.
- Quarter lines (2.25, -0.75) are not evaluated, and neither are alternate
  lines priced far from even.
- One season of archived goal lines (2023-24, the old DraftKings feed) is left
  out of the betting record: those lines kept updating during the matches, so
  they were not prices anyone could have bet before kickoff.

## Publishing

The site is served by GitHub Pages from the `gh-pages` branch, which holds
nothing but the contents of `site/`.

The workflow `.github/workflows/cfb-picks.yml` reruns both pipelines every
morning from August through May and replaces that branch with the new build.
College football is generated from August to January and soccer from August to
May; out of season a sport keeps the picks already committed. GitHub only runs
schedules from the default branch, so the workflow has to be on `main` for the
updates to happen. It can also be started by hand from **Actions → CFB Picks →
Run workflow**.

If the site ever stops being served, check **Settings → Pages** and make sure
the source is *Deploy from a branch*, branch `gh-pages`, folder `/ (root)`.
GitHub pauses scheduled workflows after 60 days without repository activity;
they can be re-enabled from the Actions tab. If ESPN's API fails, the workflow
fails and the site keeps its previous version: no data is made up.

## Known limits (college football)

- It only uses final scores. It does not see plays, yards, injuries or weather.
- FCS opponents are treated as one average team.
- The simulated ROI uses the closing line, not the line you see on Thursday.
- ESPN's historical closing lines are only complete for 2021, 2022 and 2026,
  so the betting sample covers three seasons.
