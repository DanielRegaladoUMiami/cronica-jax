"""European soccer board: removing the margin, pairing matches with Kalshi and pricing contracts."""

import pytest
from src.data import kalshi
from src.soccer import board


def market(ticker, label, yes_bid, yes_ask, volume=100):
    return {"ticker": ticker, "yes_sub_title": label, "volume_fp": str(volume),
            "yes_bid_dollars": f"{yes_bid:.4f}", "yes_ask_dollars": f"{yes_ask:.4f}",
            "no_bid_dollars": f"{1 - yes_ask:.4f}", "no_ask_dollars": f"{1 - yes_bid:.4f}"}  # fmt: skip


def kalshi_event(tail, title, home, away, prices):
    base = f"KXUELGAME-{tail}"
    return {"event_ticker": base, "title": title, "markets": [
        market(f"{base}-{home[0]}", home[1], *prices[0]),
        market(f"{base}-{away[0]}", away[1], *prices[2]),
        market(f"{base}-TIE", "Tie", *prices[1]),
    ]}  # fmt: skip


def espn_event(event_id, date, home, away, close, opened=None):
    def side(name, abbr, where):
        return {"homeAway": where, "team": {"id": abbr, "displayName": name, "shortDisplayName": name, "abbreviation": abbr},
                "records": [{"summary": "3-1-0"}], "form": "WDDWW", "score": "0"}  # fmt: skip

    def line(values):
        return {k: {"close": {"odds": c}, "open": {"odds": o}} for k, c, o in
                zip(("home", "draw", "away"), values, opened or values)}  # fmt: skip

    return {"id": event_id, "date": date, "status": {"type": {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False}},
            "competitions": [{"competitors": [side(*home, "home"), side(*away, "away")],
                              "venue": {"address": {"city": "Piraeus"}},
                              "odds": [{"provider": {"name": "DraftKings"}, "overUnder": 2.5, "moneyline": line(close)}]}]}  # fmt: skip


def test_margin_is_removed():
    assert board.implied(-200) == pytest.approx(2 / 3)
    assert board.implied(+300) == pytest.approx(0.25)
    raw = [board.implied(v) for v in (-225, 400, 550)]
    fair = board.shin(raw)
    assert sum(raw) > 1.04 and sum(fair) == pytest.approx(1.0)
    even = [r / sum(raw) for r in raw]
    # The margin sits on the long shot: it loses more than an even split would take, the favorite less.
    assert fair[2] < even[2] and fair[0] > even[0]
    assert board.shin([0.5, 0.3, 0.2]) == pytest.approx([0.5, 0.3, 0.2])


def test_club_names_are_matched_without_guessing():
    assert board.name_score("AC Milan", "Milan") == 1.0
    assert board.name_score("Paris Saint-Germain", "PSG") == 1.0
    assert board.name_score("Internazionale", "Inter") == 1.0
    assert board.name_score("FC Twente", "Twente") == 1.0
    assert board.name_score("Mjällby AIF", "Mjallby") >= 0.8
    assert board.name_score("Real Madrid", "Atletico Madrid") < 0.55
    assert board.name_score("Manchester City", "Manchester United") < 0.55
    assert board.name_score("", "Milan") == 0.0


def test_matches_are_paired_with_kalshi_by_day_and_names():
    comp = board.COMPETITIONS[1]
    olympiacos = board.parse_match(
        espn_event(
            "1",
            "2026-11-26T17:45Z",
            ("Olympiacos", "OLY"),
            ("AC Milan", "MIL"),
            ("+185", "+245", "+145"),
        ),
        comp,
    )
    gent = board.parse_match(
        espn_event(
            "2",
            "2026-11-26T20:00Z",
            ("KAA Gent", "KAA"),
            ("Red Star Belgrade", "CZV"),
            ("+120", "+240", "+220"),
        ),
        comp,
    )
    alone = board.parse_match(
        espn_event(
            "3",
            "2026-11-26T20:00Z",
            ("Ferencvaros", "FTC"),
            ("NK Celje", "CEL"),
            ("-250", "+380", "+650"),
        ),
        comp,
    )
    events = [
        kalshi_event("26NOV26OLYACM", "Olympiacos vs AC Milan", ("OLY", "Olympiacos"), ("ACM", "Milan"), [(0.32, 0.33), (0.27, 0.28), (0.38, 0.39)]),
        kalshi_event("26NOV26KAACZV", "Gent vs Crvena Zvezda", ("KAA", "Gent"), ("CZV", "Crvena Zvezda"), [(0.43, 0.45), (0.27, 0.28), (0.28, 0.29)]),
        kalshi_event("26DEC10OLYACM", "Olympiacos vs AC Milan", ("OLY", "Olympiacos"), ("ACM", "Milan"), [(0.5, 0.6), (0.2, 0.3), (0.2, 0.3)]),
    ]  # fmt: skip
    pairs = board.match_kalshi([olympiacos, gent, alone], events)
    assert pairs["1"]["event"] == "KXUELGAME-26NOV26OLYACM"  # the December rematch is another day
    assert pairs["1"]["away"]["label"] == "Milan" and pairs["1"]["draw"]["label"] == "Tie"
    assert pairs["2"]["event"].endswith("KAACZV")  # the team codes carry it when the names differ
    assert "3" not in pairs  # no Kalshi market: no price, nothing guessed


def test_contracts_value_no_as_draw_or_loss():
    comp = board.COMPETITIONS[1]
    match = board.parse_match(espn_event("1", "2026-11-26T17:45Z", ("Olympiacos", "OLY"), ("AC Milan", "MIL"),
                                         ("+185", "+245", "+145"), opened=("+135", "+240", "+185")), comp)  # fmt: skip
    assert match["home"]["record"] == "3-1-0" and match["book"]["provider"] == "DraftKings"
    assert sum(match["fair"].values()) == pytest.approx(1.0)
    assert (
        match["fair_open"]["home"] > match["fair"]["home"]
    )  # the line moved away from the home team
    event = kalshi_event(
        "26NOV26OLYACM",
        "Olympiacos vs AC Milan",
        ("OLY", "Olympiacos"),
        ("ACM", "Milan"),
        [(0.32, 0.33), (0.27, 0.28), (0.38, 0.39)],
    )
    pair = board.match_kalshi([match], [event])["1"]
    match["kalshi"] = {"event": pair["event"], "markets": {k: pair[k] for k in board.OUTCOMES}}
    rows = {(c["outcome"], c["side"]): c for c in board.contracts(match, "quadratic")}
    assert len(rows) == 6
    no_home = rows[("home", "no")]
    assert no_home["text"] == "Olympiacos do not win"
    assert no_home["fair"] == pytest.approx(match["fair"]["draw"] + match["fair"]["away"])
    assert no_home["price"] == pytest.approx(0.68)  # 1 - the yes bid
    assert no_home["edge"] == pytest.approx(no_home["fair"] - 0.68 - kalshi.taker_fee(0.68))
    assert no_home["edge_rest"] > no_home["edge"]  # a resting order at the bid costs less
    assert rows[("draw", "no")]["text"] == "No draw (either team wins)"
    # Without odds or without Kalshi there is nothing to price.
    match["kalshi"] = None
    assert board.contracts(match, "quadratic") == []


def test_match_without_odds_and_event_day():
    comp = board.COMPETITIONS[2]
    event = espn_event(
        "9",
        "2026-11-26T17:45Z",
        ("Pafos", "PAF"),
        ("FC Midtjylland", "FCM"),
        ("+185", "+245", "+145"),
    )
    event["competitions"][0]["odds"] = []
    match = board.parse_match(event, comp)
    assert match["fair"] is None and match["book"] is None and match["competition"] == "uecl"
    assert board._event_day("KXUELGAME-26NOV26OLYACM") == "2026-11-26"
    assert board._event_day("nonsense") is None


def test_notes_are_attached_by_team(tmp_path):
    comp = board.COMPETITIONS[0]
    match = board.parse_match(
        espn_event(
            "1",
            "2026-11-25T20:00Z",
            ("Lille", "LIL"),
            ("Bayern Munich", "MUN"),
            ("+380", "+320", "-150"),
        ),
        comp,
    )
    path = tmp_path / "soccer_notes.csv"
    path.write_text(
        "date,team,note\n2026-11-25,Bayern Munich,without Kane and Olise\n2026-11-26,Lille,another day\n",
        encoding="utf-8",
    )
    board.attach_notes([match], board.load_notes(path))
    assert match["note"] == "Bayern Munich: without Kane and Olise"
