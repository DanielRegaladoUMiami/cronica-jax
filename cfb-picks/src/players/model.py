"""What each NFL player is expected to do in his next game.

The projection is built top-down, in the order the numbers actually happen:

1. The team. How many passes and runs the offense calls, and how efficient it is
   (yards per pass, completion rate, touchdown and interception rates, yards per
   carry), each adjusted for THE OPPONENT. Every team gets an offense effect and a
   defense effect per statistic, estimated together from all games so far (the same
   idea as the schedule-adjusted ratings of the team model): a defense that has
   allowed 7.5 yards per pass against average offenses is a +0.9 defense.
2. The quarterback. The passing numbers belong to whoever starts. His own yards per
   attempt (shrunk toward the team's level when he has few passes) replace the
   team's, so a backup starting changes every receiver's projection too.
3. The role. Each player's share of the team's targets and carries in the games he
   played, weighted toward the recent ones.
4. WHO IS AVAILABLE. Shares are split among the players who will actually play.
   When a teammate is out, his share is handed out: mostly to the players at his
   own position, the rest to everyone in proportion to their role.
5. The player's own efficiency (catch rate, yards per target, yards per carry,
   touchdown rates), shrunk toward the average of his position when he has few
   chances, and scaled so the players add up to the team total.

Everything uses only games played BEFORE the one being projected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Weeks between the same week of two consecutive seasons, for recency weights
# (22 game weeks plus the offseason counted as 12).
SEASON_SPAN = 34
POSITION_GROUPS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

# name: (numerator, denominator or None for a per-game count, kind of ridge prior)
TEAM_MODELS = {
    "pass_att": ("pass_att", None, "volume"),
    "rush_att": ("rush_att", None, "volume"),
    "ypa": ("pass_yds", "pass_att", "rate"),
    "cmp": ("pass_cmp", "pass_att", "rate"),
    "ptd": ("pass_td", "pass_att", "td"),
    "int": ("pass_int", "pass_att", "td"),
    "ypc": ("rush_yds", "rush_att", "rate"),
    "rtd": ("rush_td", "rush_att", "td"),
    "points": ("points", None, "volume"),
}

# What is projected for a player. The site and the backtest use the same names.
STATS = [
    "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int",
    "rush_att", "rush_yds", "rush_td",
    "targets", "rec", "rec_yds", "rec_td",
    "td",  # rushing plus receiving touchdowns: what an "anytime touchdown" bet pays on
]  # fmt: skip


@dataclass(frozen=True)
class Params:
    # Recency: how many weeks until a game counts half.
    hl_team: float = 12.0
    hl_share: float = 5.0
    hl_eff: float = 20.0
    # Ridge priors of the team effects, in games ("this many games of league average").
    ridge_volume: float = 4.0
    ridge_rate: float = 8.0
    ridge_td: float = 16.0
    # Shrinkage of player numbers, in the units of each denominator.
    k_targets: float = 2.0  # games of the newcomer's share, for the target share
    k_carries: float = 0.75  # the same for the carry share (roles in the run game settle faster)
    k_catch: float = 40.0  # targets
    k_ypt: float = 60.0
    k_rec_td: float = 80.0
    k_ypc: float = 90.0  # carries
    k_ypc_qb: float = 40.0  # quarterbacks differ more in how they run: trust theirs sooner
    k_rush_td: float = 80.0
    k_ypa: float = 200.0  # pass attempts
    k_cmp: float = 200.0
    k_ptd: float = 400.0
    k_int: float = 600.0
    qb_team_weight: float = 0.5  # how much of the team's passing level a new QB inherits
    # A missing teammate's share: this fraction goes to his own position group.
    same_pos_targets: float = 0.9
    same_pos_carries: float = 0.5
    qb_share: float = 0.985  # share of the team's passes thrown by the starter
    # A player with at least this share of targets or carries is a regular: when he is
    # active he gets the ball. Below it, he only does in some games (see `appear`).
    regular_tshare: float = 0.07
    regular_cshare: float = 0.10
    appear_floor: float = 0.15
    appear_new: float = 0.5  # a player with no game on record yet
    # Splitting a team's carries among everyone flattens the split a little: the lead
    # back ends up short. Shares are raised to this power before they are normalized.
    sharpen_carries: float = 1.1
    sharpen_targets: float = 1.1
    # How the same-position part of a missing teammate's share is split: 0 = in
    # proportion to each player's role, 1 = by head (the replacement takes the vacancy).
    even_targets: float = 1.0
    even_carries: float = 0.0
    # A different quarterback than usual: this much of the pass volume is his own habit.
    qb_volume_weight: float = 0.5
    # ...and the team throws less with him: in the backtest, replacement starters came
    # in 2 passes and 14 yards under what the usual volume implied.
    qb_change_volume: float = 0.94
    # The betting line as information. When the line expects a team to score 10% more
    # than its season says, how much more of each thing happens (measured in the
    # backtest, then rounded down a little).
    script_td: float = 1.2
    script_pass_att: float = 0.2
    script_ypa: float = 0.4
    script_rush_yds: float = 0.25
    # Switches for the ablations of the backtest.
    use_opponent: bool = True
    use_teammates: bool = True
    use_market: bool = True


def abs_week(season, week_index):
    """A single week counter across seasons, so 'weeks ago' is one subtraction."""
    return season * SEASON_SPAN + week_index


def prepare(players: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the week counter and each player's team totals for the game."""
    teams = teams.copy()
    teams["t"] = abs_week(teams["season"], teams["week_index"])
    players = players.copy()
    players["t"] = abs_week(players["season"], players["week_index"])
    players["td"] = players["rush_td"] + players["rec_td"]
    totals = teams[["game_id", "team_id", "pass_att", "rush_att", "targets"]].rename(
        columns={
            "pass_att": "team_pass_att",
            "rush_att": "team_rush_att",
            "targets": "team_targets",
        }
    )
    players = players.merge(totals, on=["game_id", "team_id"], how="left")
    players = players.sort_values(["t", "game_id", "team_id", "athlete_id"]).reset_index(drop=True)
    players["game_no"] = players.groupby("athlete_id").cumcount()  # 0 = first game in the data
    return players, teams


def infer_positions(players: pd.DataFrame, roster: pd.DataFrame | None = None) -> dict[str, str]:
    """athlete id -> QB / RB / WR / TE. The roster decides; the rest is read off the numbers."""
    known: dict[str, str] = {}
    if roster is not None and len(roster):
        for aid, pos in zip(roster["athlete_id"], roster["position"]):
            if pos in POSITION_GROUPS:
                known[str(aid)] = POSITION_GROUPS[pos]
    sums = players.groupby("athlete_id")[["pass_att", "rush_att", "targets"]].sum()
    for aid, row in sums.iterrows():
        if aid in known:
            continue
        if row["pass_att"] >= 15 and row["pass_att"] > row["rush_att"] + row["targets"]:
            known[aid] = "QB"
        elif row["rush_att"] > row["targets"]:
            known[aid] = "RB"
        else:
            known[aid] = "WR"
    return known


# ----------------------------------------------------------------- team effects
@dataclass
class TeamEffects:
    mu: dict[str, float]
    home: dict[str, float]
    off: dict[str, dict[str, float]]
    dfn: dict[str, dict[str, float]]

    def expect(self, stat: str, team: str, opp: str, home: bool, opponent: bool = True) -> float:
        value = self.mu[stat] + self.off[stat].get(team, 0.0)
        if opponent:
            value += self.dfn[stat].get(opp, 0.0)
            value += self.home[stat] * (0.5 if home else -0.5)
        return value

    def offense(self, stat: str, team: str) -> float:
        return self.off[stat].get(team, 0.0)

    def defense(self, stat: str, team: str) -> float:
        return self.dfn[stat].get(team, 0.0)


def fit_team_effects(teams: pd.DataFrame, now: int, params: Params) -> TeamEffects:
    """Offense and defense effects per statistic from the games before week `now`.

    For each statistic: value = league + offense[team] + defense[opponent] + home/2,
    by weighted ridge regression. Recent games weigh more; the ridge pulls teams
    with few games toward the league average.
    """
    past = teams[teams["t"] < now]
    ids = sorted(set(past["team_id"]) | set(past["opp_id"]))
    index = {tid: i for i, tid in enumerate(ids)}
    n = len(ids)
    out = TeamEffects({}, {}, {}, {})
    if past.empty:
        for stat in TEAM_MODELS:
            out.mu[stat], out.home[stat], out.off[stat], out.dfn[stat] = 0.0, 0.0, {}, {}
        return out
    recency = 0.5 ** ((now - past["t"].to_numpy(dtype=float)) / params.hl_team)
    team_idx = past["team_id"].map(index).to_numpy()
    opp_idx = past["opp_id"].map(index).to_numpy()
    home = np.where(past["home"].to_numpy(dtype=bool), 0.5, -0.5)
    rows = np.arange(len(past))
    design = np.zeros((len(past), 2 + 2 * n))
    design[:, 0] = 1.0
    design[:, 1] = home
    design[rows, 2 + team_idx] = 1.0
    design[rows, 2 + n + opp_idx] = 1.0
    ridge = {"volume": params.ridge_volume, "rate": params.ridge_rate, "td": params.ridge_td}
    for stat, (num, den, kind) in TEAM_MODELS.items():
        y = past[num].to_numpy(dtype=float)
        weight = recency.copy()
        if den is not None:
            d = past[den].to_numpy(dtype=float)
            ok = d > 0
            y = np.where(ok, y / np.where(ok, d, 1.0), 0.0)
            weight = weight * np.where(ok, d / d[ok].mean(), 0.0)
        good = np.isfinite(y)
        weight = np.where(good, weight, 0.0)
        y = np.where(good, y, 0.0)
        xtw = design.T * weight
        penalty = np.zeros(2 + 2 * n)
        penalty[2:] = ridge[kind]
        penalty[1] = 1e-6
        beta = np.linalg.solve(xtw @ design + np.diag(penalty) + 1e-9 * np.eye(2 + 2 * n), xtw @ y)
        out.mu[stat] = float(beta[0])
        out.home[stat] = float(beta[1])
        out.off[stat] = {tid: float(beta[2 + i]) for tid, i in index.items()}
        out.dfn[stat] = {tid: float(beta[2 + n + i]) for tid, i in index.items()}
    return out


# ------------------------------------------------------------- player estimates
def position_priors(
    players: pd.DataFrame, positions: dict[str, str]
) -> dict[str, dict[str, float]]:
    """What is assumed of a player with little or no history.

    Efficiency: the average of his position. Role: the average share of players in
    their FIRST games in the data, which is far below the position's average (a
    receiver nobody has seen yet is usually a backup: about 7% of the targets, not
    the 15% of an established one). The first weeks of the data are skipped for
    that, because there everybody looks new.
    """
    df = players.assign(pos=players["athlete_id"].map(positions).fillna("WR"))
    df = df.assign(
        ts=df["targets"] / df["team_targets"].clip(lower=1),
        cs=df["rush_att"] / df["team_rush_att"].clip(lower=1),
    )
    new = df[(df["game_no"] < 2) & (df["t"] > df["t"].min() + 8)]
    league = {
        "ypa": df["pass_yds"].sum() / max(df["pass_att"].sum(), 1.0),
        "cmp": df["pass_cmp"].sum() / max(df["pass_att"].sum(), 1.0),
        "ptd": df["pass_td"].sum() / max(df["pass_att"].sum(), 1.0),
        "int": df["pass_int"].sum() / max(df["pass_att"].sum(), 1.0),
    }
    out: dict[str, dict[str, float]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        g, fresh = df[df["pos"] == pos], new[new["pos"] == pos]
        tg, ca = max(g["targets"].sum(), 1.0), max(g["rush_att"].sum(), 1.0)
        if len(fresh) >= 25:
            tshare, cshare = float(fresh["ts"].mean()), float(fresh["cs"].mean())
        else:  # not enough newcomers seen yet: half the position's average
            tshare = 0.5 * float(g["ts"].mean()) if len(g) else 0.0
            cshare = 0.5 * float(g["cs"].mean()) if len(g) else 0.0
        out[pos] = {
            "tshare": tshare,
            "cshare": cshare,
            "catch": float(g["rec"].sum() / tg),
            "ypt": float(g["rec_yds"].sum() / tg),
            "rec_td": float(g["rec_td"].sum() / tg),
            "ypc": float(g["rush_yds"].sum() / ca),
            "rush_td": float(g["rush_td"].sum() / ca),
            **league,
        }
    out["QB"]["tshare"] = 0.0
    return out


def player_estimates(
    players: pd.DataFrame,
    now: int,
    params: Params,
    positions: dict[str, str],
    priors: dict,
    teams: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Each player's role and efficiency from the games he played before week `now`.

    A box score only lists the players who touched the ball. So a backup's share
    "in the games he played" is his share in the games he got the ball, which
    overstates him: in the others he was on the field and got nothing. `appear` is
    how often he shows up in his team's box score, and `weight` discounts his share
    by it. A regular always shows up when he is healthy, so his weight is 1.
    """
    past = players[players["t"] < now]
    if past.empty:
        return pd.DataFrame()
    age = (now - past["t"]).to_numpy(dtype=float)
    w_share = 0.5 ** (age / params.hl_share)
    w_eff = 0.5 ** (age / params.hl_eff)
    cols = {
        "s_w": w_share,
        "s_targets": w_share * past["targets"].to_numpy(),
        "s_team_targets": w_share * past["team_targets"].to_numpy(),
        "s_rush": w_share * past["rush_att"].to_numpy(),
        "s_team_rush": w_share * past["team_rush_att"].to_numpy(),
        "s_pass": w_share * past["pass_att"].to_numpy(),
        "s_team_pass": w_share * past["team_pass_att"].to_numpy(),
    }
    for col in ("targets", "rec", "rec_yds", "rec_td", "rush_att", "rush_yds", "rush_td",
                "pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int"):  # fmt: skip
        cols[f"e_{col}"] = w_eff * past[col].to_numpy()
    started = (past["pass_att"] >= 15).to_numpy(dtype=float)
    cols["s_starts"] = w_share * started
    cols["s_start_att"] = w_share * started * past["pass_att"].to_numpy()
    frame = pd.DataFrame(cols, index=past.index)
    frame["athlete_id"] = past["athlete_id"]
    sums = frame.groupby("athlete_id").sum()
    last = past.groupby("athlete_id").agg(
        last_t=("t", "max"), games=("t", "size"), name=("name", "last"), team_id=("team_id", "last")
    )
    est = sums.join(last)
    est["pos"] = [positions.get(aid, "WR") for aid in est.index]
    pri = pd.DataFrame([priors[p] for p in est["pos"]], index=est.index)
    # Role: share of the team's targets and carries in the games he played.
    mean_targets = float(past["team_targets"].mean())
    mean_rush = float(past["team_rush_att"].mean())
    kt, kr = params.k_targets * mean_targets, params.k_carries * mean_rush
    est["tshare"] = (est["s_targets"] + kt * pri["tshare"]) / (est["s_team_targets"] + kt)
    est["cshare"] = (est["s_rush"] + kr * pri["cshare"]) / (est["s_team_rush"] + kr)
    est["pshare"] = est["s_pass"] / est["s_team_pass"].clip(lower=1e-9)
    est["raw_tshare"] = est["s_targets"] / est["s_team_targets"].clip(lower=1e-9)
    est["raw_cshare"] = est["s_rush"] / est["s_team_rush"].clip(lower=1e-9)
    # Efficiency, shrunk toward the position's average.
    est["catch"] = (est["e_rec"] + params.k_catch * pri["catch"]) / (
        est["e_targets"] + params.k_catch
    )
    est["ypt"] = (est["e_rec_yds"] + params.k_ypt * pri["ypt"]) / (est["e_targets"] + params.k_ypt)
    est["rec_td"] = (est["e_rec_td"] + params.k_rec_td * pri["rec_td"]) / (
        est["e_targets"] + params.k_rec_td
    )
    k_ypc = np.where(est["pos"] == "QB", params.k_ypc_qb, params.k_ypc)
    est["ypc"] = (est["e_rush_yds"] + k_ypc * pri["ypc"]) / (est["e_rush_att"] + k_ypc)
    est["rush_td"] = (est["e_rush_td"] + params.k_rush_td * pri["rush_td"]) / (
        est["e_rush_att"] + params.k_rush_td
    )
    est["appear"] = _appearance(past, est, teams, now, params) if teams is not None else 1.0
    est["regular"] = (
        (est["tshare"] >= params.regular_tshare)
        | (est["cshare"] >= params.regular_cshare)
        | ((est["pos"] == "QB") & (est["pshare"] >= 0.5))
    )
    est["weight"] = np.where(est["regular"], 1.0, est["appear"].clip(params.appear_floor, 1.0))
    return est


def _appearance(past, est, teams, now: int, params: Params) -> pd.Series:
    """Share of his current team's recent games in which the player touched the ball."""
    with_team = past.merge(
        est[["team_id"]].rename(columns={"team_id": "last_team"}),
        left_on="athlete_id",
        right_index=True,
    )
    with_team = with_team[with_team["team_id"] == with_team["last_team"]]
    weight = 0.5 ** ((now - with_team["t"]) / params.hl_share)
    mine = (
        with_team.assign(w=weight)
        .groupby("athlete_id")
        .agg(first=("t", "min"), played=("w", "sum"))
    )
    team_games = teams[teams["t"] < now][["team_id", "t"]].drop_duplicates()
    schedule = {tid: np.sort(grp["t"].to_numpy()) for tid, grp in team_games.groupby("team_id")}
    out = pd.Series(1.0, index=est.index)
    for aid, first, played in zip(mine.index, mine["first"], mine["played"]):
        weeks = schedule.get(est.at[aid, "team_id"])
        if weeks is None:
            continue
        since = weeks[weeks >= first]
        possible = float((0.5 ** ((now - since) / params.hl_share)).sum())
        if possible > 0:
            out.at[aid] = min(played / possible, 1.0)
    return out


def _qb_rate(est_row, stat: str, num: str, k: float, prior: float) -> float:
    if est_row is None:
        return prior
    return float((est_row[f"e_{num}"] + k * prior) / (est_row["e_pass_att"] + k))


# ------------------------------------------------------------------- projection
def split_shares(
    shares: dict[str, float],
    active: list[str],
    missing: list[str],
    positions: dict[str, str],
    same_pos: float,
    redistribute: bool,
    power: float = 1.0,
    even: float = 0.0,
) -> dict[str, float]:
    """Shares of the players who play, adding up to 1.

    `missing` are teammates with a role who are out. Their share is handed out:
    `same_pos` of it to the active players at the same position, the rest to all
    active players, always in proportion to the role each one already had.
    """
    out = {aid: max(shares.get(aid, 0.0), 0.0) ** power for aid in active}
    shares = {aid: max(value, 0.0) ** power for aid, value in shares.items()}
    total = sum(out.values())
    if total <= 0:
        return dict.fromkeys(active, 0.0)
    if redistribute:
        base = dict(out)
        for gone in missing:
            share = max(shares.get(gone, 0.0), 0.0)
            if share <= 0:
                continue
            peers = [a for a in active if positions.get(a) == positions.get(gone) and base[a] > 0]
            peer_total = sum(base[a] for a in peers)
            inside = same_pos if peer_total > 0 else 0.0
            for a in peers:
                # `even`: the man who steps into the lineup takes the vacant role, so the
                # same-position part is split by head rather than by previous role.
                slice_ = even / len(peers) + (1.0 - even) * base[a] / peer_total
                out[a] += share * inside * slice_
            for a in active:
                out[a] += share * (1.0 - inside) * base[a] / total
    scale = sum(out.values())
    return {aid: value / scale for aid, value in out.items()}


def project_team(
    team: str,
    opp: str,
    home: bool,
    effects: TeamEffects,
    est: pd.DataFrame,
    active: list[str],
    qb: str | None,
    missing: list[str],
    positions: dict[str, str],
    priors: dict,
    params: Params,
    market: dict | None = None,
    usual_qb: str | None = None,
) -> tuple[list[dict], dict]:
    """Project every active player of one team for one game.

    `market` (optional): {"points": the team's implied points from the betting line}.
    `usual_qb` (optional): who has been throwing for the team; if the starter is
    someone else, part of the pass volume follows the new man's own habit.
    Returns the player rows and the team-level numbers they add up to.
    """
    opponent = params.use_opponent

    def team_value(stat: str) -> float:
        return effects.expect(stat, team, opp, home, opponent)

    pass_att = max(team_value("pass_att"), 10.0)
    rush_att = max(team_value("rush_att"), 8.0)
    mu = effects.mu

    # --- the quarterback's own level, shifted by the opposing defense
    qb_row = est.loc[qb] if (qb is not None and qb in est.index) else None
    changed = qb is not None and usual_qb is not None and qb != usual_qb
    if changed:
        moved = pass_att
        if qb_row is not None and qb_row["s_starts"] >= 1.0:
            own_volume = float(qb_row["s_start_att"] / qb_row["s_starts"])
            shift = effects.defense("pass_att", opp) if opponent else 0.0
            lam = params.qb_volume_weight
            moved = (1.0 - lam) * pass_att + lam * (own_volume + shift)
        moved *= params.qb_change_volume
        # The plays he does not throw are runs, and the other way round.
        rush_att = max(rush_att - (moved - pass_att), 8.0)
        pass_att = max(moved, 10.0)
    w = params.qb_team_weight
    rates = {}
    for stat, num, k in (
        ("ypa", "pass_yds", params.k_ypa),
        ("cmp", "pass_cmp", params.k_cmp),
        ("ptd", "pass_td", params.k_ptd),
        ("int", "pass_int", params.k_int),
    ):
        prior = mu[stat] + w * effects.offense(stat, team)
        own = _qb_rate(qb_row, stat, num, k, prior)
        shift = effects.defense(stat, opp) + effects.home[stat] * (0.5 if home else -0.5)
        rates[stat] = max(own + (shift if opponent else 0.0), 0.0)
    ypc_team = max(team_value("ypc"), 2.0)
    rtd_team = max(team_value("rtd"), 0.0)

    # --- the betting line as information: is the offense expected to score more than usual?
    factor = 1.0
    if params.use_market and market and market.get("points"):
        model_points = max(team_value("points"), 6.0)
        factor = float(np.clip(market["points"] / model_points, 0.6, 1.6))
    f_td, f_ypa, f_rush = (
        factor**params.script_td,
        factor**params.script_ypa,
        factor**params.script_rush_yds,
    )
    pass_att = pass_att * factor**params.script_pass_att

    team_pass_yds = pass_att * rates["ypa"] * f_ypa
    team_cmp = pass_att * min(rates["cmp"], 0.95)
    # Touchdowns follow the points, not the number of passes: undo the volume part.
    team_ptd = pass_att * rates["ptd"] * f_td / factor**params.script_pass_att
    team_int = pass_att * rates["int"]
    team_rush_yds = rush_att * ypc_team * f_rush
    team_rtd = rush_att * rtd_team * f_td
    team_targets = pass_att * priors["_targets_per_att"]

    # --- roles among the players who play
    playing = list(active)
    if not params.use_teammates:
        # Ablation: the model does not know who is out, so the missing teammates
        # keep their share and nobody inherits it.
        active, missing = list(active) + list(missing), []
    receivers = [a for a in active if positions.get(a) != "QB"]
    playing_set = set(active)
    tshares = split_shares(
        _column(
            est,
            "tshare",
            receivers + missing,
            positions,
            priors,
            "tshare",
            playing_set,
            params.appear_new,
        ),
        receivers,
        [m for m in missing if positions.get(m) != "QB"],
        positions,
        params.same_pos_targets,
        True,
        params.sharpen_targets,
        params.even_targets,
    )
    # The quarterback's runs are scrambles and designed keepers: they do not depend
    # on which running backs are healthy, so his share is set first and the backs
    # split the rest.
    qb_carries = 0.0
    if qb is not None and qb in active:
        qb_carries = float(est.at[qb, "cshare"]) if qb in est.index else priors["QB"]["cshare"]
        qb_carries = min(max(qb_carries, 0.0), 0.5)
    runners = [a for a in active if a != qb]
    cshares = split_shares(
        _column(
            est,
            "cshare",
            runners + missing,
            positions,
            priors,
            "cshare",
            playing_set,
            params.appear_new,
        ),
        runners,
        missing,
        positions,
        params.same_pos_carries,
        True,
        params.sharpen_carries,
        params.even_carries,
    )
    cshares = {aid: value * (1.0 - qb_carries) for aid, value in cshares.items()}
    if qb is not None and qb in active:
        cshares[qb] = qb_carries

    def eff(aid: str, col: str) -> float:
        if aid in est.index:
            return float(est.at[aid, col])
        return float(priors[positions.get(aid, "WR")][col])

    raw = {}
    for aid in active:
        targets = team_targets * tshares.get(aid, 0.0)
        carries = rush_att * cshares.get(aid, 0.0)
        raw[aid] = {
            "targets": targets,
            "rec": targets * eff(aid, "catch"),
            "rec_yds": targets * eff(aid, "ypt"),
            "rec_td": targets * eff(aid, "rec_td"),
            "rush_att": carries,
            "rush_yds": carries * eff(aid, "ypc"),
            "rush_td": carries * eff(aid, "rush_td"),
        }
    # Players add up to the team: the team total is the better-estimated number.
    passer_share = params.qb_share if qb is not None else 0.0
    for col, total in (
        ("rec", team_cmp),
        ("rec_yds", team_pass_yds),
        ("rec_td", team_ptd),
        ("rush_yds", team_rush_yds),
        ("rush_td", team_rtd),
    ):
        current = sum(r[col] for r in raw.values())
        scale = total / current if current > 0 else 0.0
        for r in raw.values():
            r[col] *= scale

    rows = []
    for aid in playing:
        r = raw[aid]
        row = dict.fromkeys(STATS, 0.0)
        row.update(r)
        if aid == qb:
            row["pass_att"] = pass_att * passer_share
            row["pass_cmp"] = team_cmp * passer_share
            row["pass_yds"] = team_pass_yds * passer_share
            row["pass_td"] = team_ptd * passer_share
            row["pass_int"] = team_int * passer_share
        row["td"] = row["rush_td"] + row["rec_td"]
        rows.append(
            {
                "athlete_id": aid,
                "pos": positions.get(aid, "WR"),
                "is_qb": aid == qb,
                "tshare": tshares.get(aid, 0.0),
                "cshare": cshares.get(aid, 0.0),
                **row,
            }
        )
    context = {
        "pass_att": pass_att,
        "rush_att": rush_att,
        "pass_yds": team_pass_yds,
        "rush_yds": team_rush_yds,
        "pass_td": team_ptd,
        "rush_td": team_rtd,
        "ypa": rates["ypa"] * f_ypa,
        "ypc": ypc_team * f_rush,
        "market_factor": factor,
        "qb_change": bool(changed),
        # How the opposing defense moves the numbers, against an average defense.
        "opp_pass": effects.defense("ypa", opp) / max(mu["ypa"], 1e-9),
        "opp_rush": effects.defense("ypc", opp) / max(mu["ypc"], 1e-9),
        "missing_targets": sum(
            float(est.at[m, "tshare"])
            for m in missing
            if m in est.index and positions.get(m) != "QB"
        ),
        "missing_carries": sum(float(est.at[m, "cshare"]) for m in missing if m in est.index),
    }
    return rows, context


def _column(
    est, col, ids, positions, priors, prior_col, discount=(), new_weight=1.0
) -> dict[str, float]:
    """A share per player; the ones in `discount` are weighted by how often they get the ball."""
    out = {}
    for aid in ids:
        if aid in est.index:
            value = float(est.at[aid, col])
            if aid in discount:
                value *= float(est.at[aid, "weight"])
        else:
            value = float(priors[positions.get(aid, "WR")][prior_col])
            if aid in discount:
                value *= new_weight
        out[aid] = value
    return out


def build_priors(players: pd.DataFrame, positions: dict[str, str]) -> dict:
    priors = position_priors(players, positions)
    priors["_targets_per_att"] = float(
        players["targets"].sum() / max(players["pass_att"].sum(), 1.0)
    )
    return priors
