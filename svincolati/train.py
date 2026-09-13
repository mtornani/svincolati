"""Training rows (multiple as_of per spell), temporal split, model vs baseline, artifacts."""
from __future__ import annotations

import json
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import features as F
from .paths import MODELS

HORIZONS = (30, 60)
MARGIN = 0.02


def training_rows(players: pd.DataFrame, spells: pd.DataFrame, today: date, step: int = 14) -> list[dict]:
    pmap = {r["tm_id"]: r for r in players.to_dict("records")}
    rows: list[dict] = []
    for tm_id, grp in spells.groupby("tm_id"):
        player = pmap.get(tm_id)
        if player is None:
            continue
        sp_list = sorted(grp.to_dict("records"), key=lambda s: s["start"])
        for i, sp in enumerate(sp_list):
            prior = sp_list[:i]
            end = sp["end"] if isinstance(sp["end"], date) else None
            as_of = sp["start"]
            while True:
                if end is None and as_of + timedelta(days=60) > today:
                    break
                if end is not None and as_of >= end:
                    break
                f = F.build(player, sp, prior, as_of)
                left = (end - as_of).days if end is not None else None
                signed = end is not None and not sp["retired"]
                rows.append({"tm_id": tm_id, "as_of": as_of, **f, "baseline": F.baseline(f),
                             "y30": int(signed and left <= 30), "y60": int(signed and left <= 60)})
                as_of += timedelta(days=step)
    return rows


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def fit(players: pd.DataFrame, spells: pd.DataFrame, today: date, models_dir: Path = MODELS) -> dict:
    df = pd.DataFrame(training_rows(players, spells, today)).sort_values("as_of").reset_index(drop=True)
    cutoff = df["as_of"].iloc[int(len(df) * 0.8)]
    tr, te = df[df["as_of"] < cutoff], df[df["as_of"] >= cutoff]
    version = "v" + today.isoformat()
    feats = [c for c in F.NAMES if tr[c].nunique(dropna=True) >= 2]  # HGB rejects constant/all-NaN columns
    meta = {"version": version, "trained_at": today.isoformat(), "n_rows": int(len(df)),
            "n_train": int(len(tr)), "n_test": int(len(te)), "cutoff": cutoff.isoformat(), "features": feats}
    lines = [f"# eval {version}", "", f"rows {len(df)} | train {len(tr)} | test {len(te)} | cutoff {cutoff}", ""]
    for h in HORIZONS:
        y = f"y{h}"
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
        clf.fit(tr[feats], tr[y])
        p = clf.predict_proba(te[feats])[:, 1]
        auc_m, auc_b = _safe_auc(te[y], p), _safe_auc(te[y], te["baseline"])
        brier_m = float(brier_score_loss(te[y], p))
        calib = tr.groupby("baseline")[y].mean().round(3).to_dict()
        use = bool(auc_m == auc_m and auc_b == auc_b and auc_m >= auc_b + MARGIN)
        meta.update({f"auc_model_{h}": auc_m, f"auc_baseline_{h}": auc_b, f"brier_model_{h}": brier_m,
                     f"use_model_{h}": use, f"base_rate_{h}": float(te[y].mean()),
                     f"baseline_calib_{h}": {str(k): v for k, v in calib.items()}})
        with open(Path(models_dir) / f"model_{version}_{h}.pkl", "wb") as fh:
            pickle.dump(clf, fh)
        lines += [f"## horizon {h}d", f"- base rate (test): {te[y].mean():.3f}",
                  f"- AUC model: {auc_m:.3f} | AUC baseline: {auc_b:.3f} | Brier model: {brier_m:.3f}",
                  f"- ship model: **{use}** (needs AUC >= baseline + {MARGIN})", ""]
    (Path(models_dir) / f"meta_{version}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (Path(models_dir) / "meta_latest.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (Path(models_dir) / f"eval_{version}.md").write_text("\n".join(lines), encoding="utf-8")
    return meta
