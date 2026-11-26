# CFB Picks (NFL, college football and European soccer)

A weekly betting-research site with three sections. It downloads every game
from ESPN's public API, fits its models, checks them with walk-forward
backtests, and publishes the result as a static site.

- **College football**: every FBS game. Most of this page describes it.
- **NFL**: the same team pipeline with `--league nfl` (see [NFL](#nfl)), plus
  projections for every player compared with DraftKings lines and Kalshi's
  prices (see [NFL players](#nfl-players)).
- **European soccer**: Champions League, Europa League and Conference League.
  No model: the sportsbook's odds next to what Kalshi charges (see
  [European soccer](#european-soccer)).

**Live site:** https://danielregaladoumiami.github.io/cronica-jax/ (the switch
at the top changes section; `#nfl-players` or `#soccer-picks` link straight to
a tab).

## College football: how good it is, and how good it is not

Walk-forward backtest of the 2026 season through week 12: 695 FBS vs FBS games
the model never saw while it was being tuned. Its pick won 491 of them and
lost 204; the betting favorite went 518-177.

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
- **The model does not beat the market.** Betting every disagreement of 5%+
  that had no red flag, 1,817 past bets against closing lines (2021, 2022 and
  2026), went 832-974 for an ROI of -5.8% ± 2.6%, which is roughly the
  bookmaker's cut. Picks flagged AVOID did worse on spreads (-13.7%).
- **When the two disagree, the market is right.** In 666 college games where
  the model gave a team 10+ points more than the market did, the model said
  51%, the market said 32% and the team won 31%. In the NFL (269
  games) it was 46%, 30% and 25%. The same holds for spreads and totals.
  So a gap between the model and the market is no longer treated as a play: see
  [the rules](#how-it-works).

These numbers are recomputed on every run and shown on the site's *Track
record* tab.

## Run it locally

```bash
cd cfb-picks
pip install -r requirements.txt

python -m src.picks --week 12     # download data, run the backtest, generate the picks
python -m src.picks --league nfl  # the same for the NFL, current week
python -m src.players.week        # NFL players (run it after the NFL picks)
python -m src.soccer.board        # European soccer: odds and Kalshi prices
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
| `python -m src.players.fetch --seasons 2025 2026` | Only download per-player box scores into `data/nfl_player_games.parquet` |
| `python -m src.players.backtest --ablation` | Player model: error by statistic, and what each ingredient adds |
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
5. **The model against the market** (`src/model/simulate.py`, `fit_trust`): the
   model's chance and the market's are two opinions. The betting backtest
   measures how much of their disagreement came true (1 = the model was right,
   0 = the market was), per market. So far it is 0 everywhere, so the chance
   the site shows for a team is the market's, with the model's own number
   beside it. If the model starts getting its disagreements right, that weight
   rises on its own and the model gets that much credit.
6. **Rules** (`src/model/rules.py`):
   - The **moneyline pick** is the team more likely to win, not the side where
     the model is furthest from the market. A 16% chance against a 12% price is
     a long shot either way.
   - **PLAY**: after that check, the chance still beats the price by 5%+, with
     positive expected value and no red flags. With the weight at 0 there are
     none.
   - **AVOID**: the model alone sees 5%+, but the pick flips when each team's
     last 2 games are removed, the model is 5+ points away from the market
     spread (7+ on a total), the line moved 3+ points against the pick since
     it opened (4+ on a total), or (NFL) a team starts a different quarterback.
   - **PASS**: no value: the model agrees with the market, or disagrees
     without a record to back it.
   - Stake: quarter Kelly, capped at 2% of bankroll.
7. **Site** (`site/`): plain HTML and JavaScript that read the JSON files in
   `site/data/`. No backend. The picks page opens on the most likely winners
   of the week with their price (and Kalshi's, for the NFL), then the model's
   disagreements with the market, kept apart and labelled as such. Unticking
   "Spreads and totals too" leaves only the moneylines.

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
11: 151 games the model never saw while it was being tuned (2022-2025). Its
pick won 95 of them and lost 55 (one game was a tie); the betting favorite
went 103-47.

| Model | Winner | When favored by 7+ | Margin miss | Total miss |
|---|---|---|---|---|
| Baseline (points vs opponent win %) | 61.3% | 69.0% | 10.56 | 11.96 |
| **SRS (in use)** | **63.3%** | **85.2%** | **9.90** | **11.17** |
| Elo with margin of victory | 63.3% | 88.5% | 9.93 | n/a |
| SRS + Elo blend | 62.7% | 87.0% | 9.91 | 11.17 |
| Market closing line | 68.7% | n/a | 9.55 | 10.40 |

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

## NFL players

`python -m src.players.week` projects every quarterback, runner and receiver of
the week (passing, rushing and receiving yards, attempts, completions,
receptions, touchdowns, interceptions) and puts the result next to what the
market says: the DraftKings line for each player, and Kalshi's ladder of
yes/no contracts ("250+ passing yards" at 65 cents). It writes
`site/data/nfl_players.json`, which the **Players** tab of the site reads.

### How a projection is built

Top-down, in the order things happen on the field (`src/players/model.py`):

1. **The team.** How many passes and runs the offense calls and how efficient
   it is (yards per pass, completion rate, touchdown and interception rates,
   yards per carry). Each team gets an offense effect and a defense effect per
   statistic, estimated together from all games so far, so the projection is
   adjusted for **the opponent**.
2. **The quarterback.** The passing numbers belong to whoever starts. A backup
   changes every receiver's projection, and the team throws less with him.
3. **The role.** Each player's share of the team's targets and carries in the
   games he played, weighted toward the recent ones.
4. **Who is available.** Rosters, injury statuses and depth charts come from
   ESPN. Shares are split among the players who will play; when a teammate is
   out, the man who replaces him takes most of the vacancy.
5. **His own efficiency.** Catch rate, yards per target and per carry, pulled
   toward the average of his position when he has had few chances.

A projection is an average, and a bet asks about a threshold. The probability
of "250+ passing yards" comes from how real results have spread around
projections of that size in the backtest (`src/players/dist.py`), so it carries
the model's real error.

### How good it is, and how good it is not

Walk-forward backtest over 2025 and 2026 (448 games): every week projected
with only what was known before it. The yardstick is the simplest thing a
bettor would do, the player's average per game so far. Average miss:

| Statistic | Model, both seasons | Season average | Model, 2026 | Season average, 2026 |
|---|---|---|---|---|
| Passing yards | 58.18 | 63.92 | 58.44 | 64.95 |
| Passing touchdowns | 0.93 | 1.02 | 0.91 | 1.01 |
| Interceptions thrown | 0.66 | 0.69 | 0.66 | 0.66 |
| Carries | 3.82 | 4.13 | 3.76 | 4.19 |
| Rushing yards | 23.31 | 24.49 | 22.44 | 24.03 |
| Receptions | 1.57 | 1.64 | 1.60 | 1.69 |
| Receiving yards | 21.34 | 22.71 | 21.85 | 23.65 |

- **Who is out matters most.** When a team is missing a main running back
  (13% of team games), the backs who are left carry 4 more times and gain
  16 more yards than their average says. Knowing who is out cuts the miss on
  carries from 5.4 to 4.0. When a main receiver is out (21% of team
  games) the others gain about 8 receiving yards each.
- **The opponent matters less than it seems.** Dropping the defense adjustment
  moves the miss on passing yards from 58.2 to 58.6: one game is too noisy
  for the defense to show.
- **The percentages hold up.** Fitted on 2025 and checked on 2026, what the model
  called 65% happened between 62% and 71% of the time (off by 1 to 4 points on
  average, depending on the statistic).
- **Against the market.** In week 12 of 2026 the projections sat about 5 yards
  from the DraftKings line on receiving yards, 8 on rushing and 13 on passing,
  closer than the season averages do. That says the model is sane, not that it
  beats the line.
- **There is no betting record yet.** ESPN does not keep past player lines and
  the history of Kalshi's prices was not downloaded, so nothing here has been
  tested against the prices you would have paid. Every flagged contract is
  written to `site/data/nfl_props_log.json` before kickoff and settled once
  the game is played; the site shows that record as it grows.

### The flags

A contract is a rung of Kalshi's ladder, bought as Yes or No.

Two things decide what gets flagged, both from evidence:

- **Only likely things.** Nothing under 30 cents, and nothing the model gives
  less than 55%, is ever flagged. A cheap contract is a lottery ticket: on
  Kalshi those win less often than their price implies
  ([Bürgi, Deng and Whelan](https://www.karlwhelan.com/Papers/Kalshi.pdf)), and
  a few points of model error is a third of what one is worth.
- **The model is not enough on its own.** In the team markets the model's
  disagreements with the market came to nothing, so a play needs a second
  market on its side.

- **PLAY**: a likely contract (55%+ by the model and by DraftKings) where both
  the model and the DraftKings line leave 4+ points after Kalshi's fee, the
  price is between 30 and 90 cents with bid and ask no more than 5 cents
  apart, at least 50 contracts have traded, and the player is not
  questionable, just back from injury or playing with a new quarterback.
- **LEAN**: a likely contract the model thinks is cheap, without a DraftKings
  line that fully backs it, or in a thin market.
- **CHECK**: the model is more than 12 points from the market, or something
  about the player changed. A gap that size usually means the market knows
  something: read the news first.

The same step also tells the team picks which teams start a different
quarterback than usual (the team model rates a team by its season and does not
know the starter is out), and adds Kalshi's price for each team to win.

Kalshi charges about 7% x price x (1 - price) per contract (1.7 cents at 50
cents, almost nothing near 0 or 100). Every edge on the site is after that fee.

## European soccer

`python -m src.soccer.board` builds the **Soccer** section: every Champions
League, Europa League and Conference League match of the next few days, with
no prediction of ours. Nothing in this project has been tested on European cup
matches, so the board shows the market instead:

1. The sportsbook's three-way odds (home, draw, away) from ESPN, with the
   bookmaker's margin removed by Shin's method (an even split overstates the
   long shot, where most of the margin sits).
2. Kalshi's price for each result, Yes and No. **"No" on a team wins if it
   draws or loses**, so it is the same as backing the other team and the draw.
3. What each contract leaves after Kalshi's fee: its market chance, minus the
   price, minus the fee.

A positive number means Kalshi charges less than the sportsbook implies. That
is two markets disagreeing, not a tested edge; most weeks nothing is positive.
The list of best prices leaves out contracts under 30 cents: a one-cent gap on
a long shot is noise.
"No" is not safer either: "No" on a 14% underdog costs about 86 cents to win
14, and comes in exactly as often as it needs to.

Matches are paired with Kalshi's by day and club names (`match_kalshi`); a
weak or ambiguous pairing is left without a price rather than guessed. Team
news can be added by hand in `data/soccer_notes.csv` (`date,team,note`).

## Publishing

The site is served by GitHub Pages from the `gh-pages` branch, which holds
nothing but the contents of `site/`.

The workflow `.github/workflows/cfb-picks.yml` reruns every pipeline (team
picks for both leagues, NFL players, the soccer board) every Thursday,
Saturday and Sunday from August through February, plus Tuesday and Wednesday
during the European season, and replaces that branch with the new build. It is scheduled twice on those days (1:17 and 8:17
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
fails and the site keeps its previous version: no data is made up. If only the
player projections or the soccer board fail (Kalshi not answering, say), the
run goes on with the previous ones and the page shows when they were made.

The record of flagged player contracts lives with the published site: before
generating, the workflow downloads `data/nfl_props_log.json` from the live
site so each run adds to it.

## Known limits

- The team models only use final scores. They do not see plays, yards,
  injuries or weather. In the NFL the player step covers the biggest gap, a
  change of quarterback, by flagging those games; college football has no
  such check.
- The player model knows who is out from ESPN's injury list. It does not know
  a snap count, a game plan or the weather, and it cannot tell a healthy
  scratch from a starter until he has played.
- Player props have no betting record yet (see [NFL players](#nfl-players)).
- The soccer board has no model at all.
- FCS opponents are treated as one average team.
- The simulated ROI uses the closing line, not the line you see on Thursday.
- ESPN's historical closing lines are only complete for 2021, 2022 and 2026,
  so the betting sample covers three seasons (plus parts of two more for the
  NFL).
