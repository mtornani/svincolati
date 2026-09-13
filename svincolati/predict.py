"""Snapshot -> predictions/DATE.csv (+ .sha256). Never overwrites."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from . import features as F
from .paths import MODELS, PREDICTIONS

COLS = ["rank", "tm_id", "name", "girone", "role_group", "prev_club", "mv_eur", "days_free",
        "p30", "p60", "baseline", "model_version"]


def _load(models_dir: Path):
    meta = json.loads((Path(models_dir) / "meta_latest.json").read_text(encoding="utf-8"))
    models = {}
    for h in (30, 60):
        if meta.get(f"use_model_{h}"):
            with open(Path(models_dir) / f"model_{meta['version']}_{h}.pkl", "rb") as fh:
                models[h] = pickle.load(fh)
    return meta, models


def _prob(meta, models, h, f, X):
    if h in models:
        return float(models[h].predict_proba(X)[0][1])
    return float(meta[f"baseline_calib_{h}"].get(str(F.baseline(f)), meta[f"base_rate_{h}"]))


def run(snapshot: pd.DataFrame, players: pd.DataFrame, spells: pd.DataFrame, as_of_str: str,
        models_dir: Path = MODELS, out_dir: Path = PREDICTIONS) -> Path:
    out = Path(out_dir) / f"{as_of_str}.csv"
    if out.exists():
        raise FileExistsError(f"{out} exists: predictions are immutable")
    as_of = date.fromisoformat(as_of_str)
    meta, models = _load(models_dir)
    pmap = {r["tm_id"]: r for r in players.to_dict("records")}
    sp_by = {k: sorted(g.to_dict("records"), key=lambda s: s["start"]) for k, g in spells.groupby("tm_id")}
    rows = []
    for r in snapshot.drop_duplicates("tm_id").to_dict("records"):
        player, sps = pmap.get(r["tm_id"]), sp_by.get(r["tm_id"], [])
        cur = [s for s in sps if s["start"] <= as_of and (not isinstance(s["end"], date) or s["end"] > as_of)]
        if player is None or not cur:
            print(f"[predict] skip {r['tm_id']} {r['name']}: no profile or open spell")
            continue
        spell = cur[-1]
        prior = [s for s in sps if isinstance(s["end"], date) and s["end"] <= as_of]
        f = F.build(player, spell, prior, as_of)
        X = pd.DataFrame([f])[meta.get("features", F.NAMES)]
        rows.append({"tm_id": r["tm_id"], "name": r["name"], "girone": r["girone"], "role_group": r["role_group"],
                     "prev_club": r["prev_club"], "mv_eur": r["mv_eur"], "days_free": f["days_free"],
                     "p30": round(_prob(meta, models, 30, f, X), 4), "p60": round(_prob(meta, models, 60, f, X), 4),
                     "baseline": F.baseline(f), "model_version": meta["version"]})
    df = pd.DataFrame(rows).sort_values(["p30", "p60", "tm_id"], ascending=[False, False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df[COLS].to_csv(out, index=False, lineterminator="\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (Path(out_dir) / f"{as_of_str}.sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    return out
