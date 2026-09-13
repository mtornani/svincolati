"""Transfers -> free-agent spells. See LABELING.md."""
from __future__ import annotations

from datetime import date

import pandas as pd

from .tm import FREE

COLS = ["tm_id", "start", "end", "next_club", "days", "mv_at_start", "retired"]


def spells_from_transfers(transfers: list[dict], today: date) -> list[dict]:
    rows = sorted(transfers, key=lambda t: t["date"])
    spells: list[dict] = []
    cur: dict | None = None
    for t in rows:
        d = date.fromisoformat(t["date"])
        if t["to_club"].lower() == FREE:
            if cur is None:
                cur = {"tm_id": t["tm_id"], "start": d, "mv_at_start": t["mv_eur"]}
        elif cur is not None and t["from_club"].lower() == FREE:
            spells.append({**cur, "end": d, "next_club": t["to_club"], "days": (d - cur["start"]).days,
                           "retired": bool(t["to_special"])})
            cur = None
    if cur is not None:
        spells.append({**cur, "end": None, "next_club": "", "days": None, "retired": False})
    return spells


def build_spells(transfers: list[dict], today: date) -> pd.DataFrame:
    by_id: dict[str, list[dict]] = {}
    for t in transfers:
        by_id.setdefault(t["tm_id"], []).append(t)
    out = [s for tid in sorted(by_id) for s in spells_from_transfers(by_id[tid], today)]
    return pd.DataFrame(out, columns=COLS)


def load_spells(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"tm_id": str, "next_club": str}).fillna({"next_club": ""})
    df["start"] = pd.to_datetime(df["start"]).dt.date
    end = pd.to_datetime(df["end"], errors="coerce")
    df["end"] = [e.date() if pd.notna(e) else None for e in end]
    df["retired"] = df["retired"].astype(bool)
    return df
