import hashlib
import json
import pickle
from datetime import date

import pandas as pd
import pytest

from svincolati import predict


class Const:
    def predict_proba(self, X):
        return [[0.1, 0.9]] * len(X)


def _setup(tmp_path, use_model):
    snap = pd.DataFrame([{"snapshot_date": "2026-09-13", "girone": "A", "tm_id": "1", "name": "P1", "role_group": "GK",
                          "prev_club": "X", "nationality": "Italia", "apps_prev": 3, "goals_prev": 0, "mv_eur": 100000}])
    players = pd.DataFrame([{"tm_id": "1", "name": "P1", "birth_date": "2004-01-10", "position": "Porta",
                             "nationality": "Italia", "fetched_at": "2026-09-13"}])
    spells = pd.DataFrame([{"tm_id": "1", "start": date(2026, 7, 1), "end": None, "next_club": "", "days": None,
                            "mv_at_start": 100000, "retired": False}])
    meta = {"version": "vtest", "use_model_30": use_model, "use_model_60": use_model,
            "baseline_calib_30": {"5": 0.4}, "baseline_calib_60": {"5": 0.7}, "base_rate_30": 0.2, "base_rate_60": 0.3}
    for h in (30, 60):
        with open(tmp_path / f"model_vtest_{h}.pkl", "wb") as fh:
            pickle.dump(Const(), fh)
    (tmp_path / "meta_latest.json").write_text(json.dumps(meta))
    return snap, players, spells


def test_predict_baseline_path(tmp_path):
    snap, players, spells = _setup(tmp_path, False)
    out = predict.run(snap, players, spells, "2026-09-13", models_dir=tmp_path, out_dir=tmp_path)
    df = pd.read_csv(out, dtype={"tm_id": str})
    assert df.iloc[0]["p30"] == 0.4 and df.iloc[0]["p60"] == 0.7 and df.iloc[0]["rank"] == 1
    assert (tmp_path / "2026-09-13.sha256").read_text().split()[0] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_predict_model_path_and_immutability(tmp_path):
    snap, players, spells = _setup(tmp_path, True)
    out = predict.run(snap, players, spells, "2026-09-13", models_dir=tmp_path, out_dir=tmp_path)
    assert pd.read_csv(out).iloc[0]["p30"] == 0.9
    with pytest.raises(FileExistsError):
        predict.run(snap, players, spells, "2026-09-13", models_dir=tmp_path, out_dir=tmp_path)
