from datetime import date

import pandas as pd

from svincolati import score


def _spells():
    return pd.DataFrame([
        {"tm_id": "1", "start": date(2026, 7, 1), "end": date(2026, 8, 20), "next_club": "Rimini", "days": 50,
         "mv_at_start": 1, "retired": False},
        {"tm_id": "2", "start": date(2026, 7, 1), "end": None, "next_club": "", "days": None,
         "mv_at_start": 1, "retired": False},
        {"tm_id": "3", "start": date(2026, 7, 1), "end": date(2026, 8, 1), "next_club": "Carriera terminata",
         "days": 31, "mv_at_start": 1, "retired": True},
    ])


def _pred(d="2026-08-01"):
    return pd.DataFrame([{"rank": i + 1, "tm_id": t, "name": f"P{t}", "p30": p, "p60": p + 0.1}
                         for i, (t, p) in enumerate([("1", 0.8), ("2", 0.5), ("3", 0.3)])]).assign(prediction_date=d)


def test_resolve_statuses():
    assert score.resolve(date(2026, 8, 1), "1", _spells(), date(2026, 9, 13)) == ("signed", date(2026, 8, 20), 19, "Rimini")
    assert score.resolve(date(2026, 8, 1), "2", _spells(), date(2026, 10, 15))[0] == "expired"
    assert score.resolve(date(2026, 8, 1), "3", _spells(), date(2026, 9, 13))[0] == "retired"


def test_pending_until_60_days():
    assert score.resolve(date(2026, 8, 1), "2", _spells(), date(2026, 9, 1)) is None


def test_update_outcomes_append_only(tmp_path):
    out = tmp_path / "outcomes.csv"
    df = score.update_outcomes([_pred()], _spells(), date(2026, 9, 1), out)
    assert set(df["tm_id"]) == {"1", "3"}
    df2 = score.update_outcomes([_pred()], _spells(), date(2026, 10, 15), out)
    assert set(df2["tm_id"]) == {"1", "2", "3"} and len(df2) == 3
    board = score.scoreboard(df2)
    assert board["n_resolved"] == 3 and board["hit30"] == 1 and board["hit60"] == 1
