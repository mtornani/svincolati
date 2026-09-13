import random
from datetime import date, timedelta

import pandas as pd

from svincolati import train


def _players(n):
    return pd.DataFrame([{"tm_id": str(i), "name": f"P{i}", "birth_date": f"{1995 + i % 12}-05-01",
                          "position": ["Porta", "Difesa", "Centrocampo", "Attacco"][i % 4],
                          "nationality": "Italia" if i % 3 else "Francia", "fetched_at": "2026-09-13"}
                         for i in range(n)])


def _spells(n, seed=0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        start = date(2018 + i % 7, 7, 1)
        quick = (i % 4 in (0, 1)) and (i % 12 < 8)
        days = rnd.randint(5, 40) if quick else rnd.randint(30, 300)
        rows.append({"tm_id": str(i), "start": start, "end": start + timedelta(days=days),
                     "next_club": "X", "days": days, "mv_at_start": 50_000 * (1 + i % 5), "retired": False})
    return pd.DataFrame(rows)


def test_training_rows_labels_and_censoring():
    players = _players(2)
    sp = pd.DataFrame([
        {"tm_id": "0", "start": date(2026, 1, 1), "end": date(2026, 2, 15), "next_club": "X", "days": 45,
         "mv_at_start": 1, "retired": False},
        {"tm_id": "1", "start": date(2026, 8, 1), "end": None, "next_club": "", "days": None,
         "mv_at_start": 1, "retired": False}])
    rows = train.training_rows(players, sp, date(2026, 9, 13), step=14)
    r0 = [r for r in rows if r["tm_id"] == "0"]
    assert [r["y30"] for r in r0] == [0, 0, 1, 1]
    assert all(r["y60"] == 1 for r in r0)
    assert not [r for r in rows if r["tm_id"] == "1"]


def test_fit_writes_artifacts(tmp_path):
    meta = train.fit(_players(200), _spells(200), date(2026, 9, 13), models_dir=tmp_path)
    assert (tmp_path / f"meta_{meta['version']}.json").exists() and (tmp_path / "meta_latest.json").exists()
    assert (tmp_path / f"eval_{meta['version']}.md").exists()
    for h in (30, 60):
        assert f"auc_model_{h}" in meta and f"auc_baseline_{h}" in meta and f"use_model_{h}" in meta
        assert (tmp_path / f"model_{meta['version']}_{h}.pkl").exists()
