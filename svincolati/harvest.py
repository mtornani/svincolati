"""Free-agent lists -> data/snapshots/DATE.csv (one row per girone x tm_id)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import tm
from .paths import SNAPSHOTS

COLS = ["snapshot_date", "girone", "tm_id", "name", "role_group", "prev_club",
        "nationality", "apps_prev", "goals_prev", "mv_eur"]
MAX_PAGES = 5


def harvest(fetcher, snapshot_date: str, gironi=("A", "B", "C"), out_dir: Path = SNAPSHOTS) -> Path:
    rows: list[dict] = []
    for g in gironi:
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            html = fetcher.html(tm.list_url(g, page), "table.items")
            batch = [r for r in tm.parse_free_list(html) if r["tm_id"] not in seen]
            if not batch:
                break
            for r in batch:
                seen.add(r["tm_id"])
                rows.append({"snapshot_date": snapshot_date, "girone": g, **r})
        print(f"[harvest] girone {g}: {len(seen)} players")
    df = pd.DataFrame(rows, columns=COLS)
    out = Path(out_dir) / f"{snapshot_date}.csv"
    df.to_csv(out, index=False)
    return out
