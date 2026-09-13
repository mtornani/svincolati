"""Predictions + spells -> outcomes.csv (append-only) + scoreboard dict."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .paths import OUTCOMES_CSV

COLS = ["prediction_date", "tm_id", "name", "rank", "p30", "p60", "status", "signed_date", "club", "days"]
WAIT_DAYS = 60


def resolve(pred_date: date, tm_id: str, spells: pd.DataFrame, today: date):
    for s in spells[spells["tm_id"] == tm_id].to_dict("records"):
        end = s["end"] if isinstance(s["end"], date) else None
        if s["start"] <= pred_date and (end is None or end >= pred_date):
            if end is None:
                return ("expired", None, None, "") if (today - pred_date).days >= WAIT_DAYS else None
            return ("retired" if s["retired"] else "signed", end, (end - pred_date).days, s["next_club"])
    return None


def update_outcomes(predictions: list[pd.DataFrame], spells: pd.DataFrame, today: date,
                    path: Path = OUTCOMES_CSV) -> pd.DataFrame:
    old = pd.read_csv(path, dtype={"tm_id": str}) if Path(path).exists() else pd.DataFrame(columns=COLS)
    done = set(zip(old["prediction_date"], old["tm_id"]))
    new = []
    for pdf in predictions:
        for r in pdf.to_dict("records"):
            if (r["prediction_date"], r["tm_id"]) in done:
                continue
            res = resolve(date.fromisoformat(r["prediction_date"]), r["tm_id"], spells, today)
            if res is None:
                continue
            status, sd, days, club = res
            new.append({"prediction_date": r["prediction_date"], "tm_id": r["tm_id"], "name": r["name"],
                        "rank": r["rank"], "p30": r["p30"], "p60": r["p60"], "status": status,
                        "signed_date": sd.isoformat() if sd else "", "club": club,
                        "days": days if days is not None else ""})
    df = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def scoreboard(outcomes: pd.DataFrame, top_n: int = 10) -> dict:
    df = outcomes.copy()
    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["p30"] = pd.to_numeric(df["p30"], errors="coerce")
    signed = df["status"] == "signed"
    top = df[df["rank"] <= top_n]
    tsigned = top["status"] == "signed"
    return {
        "n_resolved": int(len(df)),
        "hit30": int((signed & (df["days"] <= 30)).sum()),
        "hit60": int((signed & (df["days"] <= 60)).sum()),
        "top_n": top_n,
        "top_resolved": int(len(top)),
        "top_hit30": int((tsigned & (top["days"] <= 30)).sum()),
        "top_hit60": int((tsigned & (top["days"] <= 60)).sum()),
        "mean_p30": float(df["p30"].mean()) if len(df) else float("nan"),
        "obs_rate30": float((signed & (df["days"] <= 30)).mean()) if len(df) else float("nan"),
    }
