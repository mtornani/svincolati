"""python -m svincolati <harvest|history|label|train|predict|score|site|weekly> [--date YYYY-MM-DD] [--headful]"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date

import pandas as pd

from . import harvest, history, label, predict, score, site, train
from .fetch import TM
from .paths import MODELS, OUTCOMES_CSV, PREDICTIONS, SNAPSHOTS, SPELLS_CSV


def today() -> str:
    return date.today().isoformat()


def cmd_harvest(a):
    with TM(headless=not a.headful) as f:
        harvest.harvest(f, a.date)


def cmd_history(a):
    snap = pd.read_csv(SNAPSHOTS / f"{a.date}.csv", dtype={"tm_id": str})
    with TM(headless=not a.headful) as f:
        history.update(f, snap["tm_id"].tolist(), a.date)


def cmd_label(a):
    df = label.build_spells(history.load_transfers(), date.fromisoformat(a.date))
    df.to_csv(SPELLS_CSV, index=False)
    print(f"[label] {len(df)} spells, {int(df['end'].isna().sum())} open")


def cmd_train(a):
    meta = train.fit(history.load_players(), label.load_spells(SPELLS_CSV), date.fromisoformat(a.date))
    print(json.dumps({k: v for k, v in meta.items() if k.startswith(("auc", "use", "n_"))}, indent=1))


def cmd_predict(a):
    snap = pd.read_csv(SNAPSHOTS / f"{a.date}.csv", dtype={"tm_id": str})
    out = predict.run(snap, history.load_players(), label.load_spells(SPELLS_CSV), a.date)
    print(f"[predict] {out}")


def cmd_score(a):
    preds = [pd.read_csv(p, dtype={"tm_id": str}).assign(prediction_date=p.stem)
             for p in sorted(PREDICTIONS.glob("*.csv"))]
    df = score.update_outcomes(preds, label.load_spells(SPELLS_CSV), date.fromisoformat(a.date))
    print(json.dumps(score.scoreboard(df), indent=1))


def cmd_site(a):
    files = sorted(PREDICTIONS.glob("*.csv"))
    latest = files[-1]
    pred = pd.read_csv(latest, dtype={"tm_id": str})
    digest = hashlib.sha256(latest.read_bytes()).hexdigest()
    archive = [(p.stem, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files[:-1]]
    outcomes = (pd.read_csv(OUTCOMES_CSV, dtype={"tm_id": str}) if OUTCOMES_CSV.exists()
                else pd.DataFrame(columns=score.COLS))
    meta = json.loads((MODELS / "meta_latest.json").read_text(encoding="utf-8"))
    site.write(site.render(pred, latest.stem, digest, score.scoreboard(outcomes), archive, meta))
    print("[site] docs/index.html")


def cmd_weekly(a):
    for fn in (cmd_harvest, cmd_history, cmd_label, cmd_score, cmd_predict, cmd_site):
        fn(a)


def main(argv=None):
    p = argparse.ArgumentParser(prog="svincolati")
    p.add_argument("--date", default=today())
    p.add_argument("--headful", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("harvest", cmd_harvest), ("history", cmd_history), ("label", cmd_label),
                     ("train", cmd_train), ("predict", cmd_predict), ("score", cmd_score),
                     ("site", cmd_site), ("weekly", cmd_weekly)]:
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)
