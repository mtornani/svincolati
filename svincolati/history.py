"""Per-player profile (once) + transfer history (refresh >30d) cache."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import tm
from .paths import HISTORY, PLAYERS_CSV

PLAYER_COLS = ["tm_id", "name", "birth_date", "position", "nationality", "fetched_at"]
STALE_DAYS = 30


def load_players(players_csv: Path = PLAYERS_CSV) -> pd.DataFrame:
    if not Path(players_csv).exists():
        return pd.DataFrame(columns=PLAYER_COLS)
    return pd.read_csv(players_csv, dtype=str).fillna("")


def update(fetcher, tm_ids: list[str], today: str, players_csv: Path = PLAYERS_CSV,
           hist_dir: Path = HISTORY) -> int:
    players = load_players(players_csv)
    known = dict(zip(players["tm_id"], players["fetched_at"]))
    t0 = date.fromisoformat(today)
    n = 0
    for tm_id in dict.fromkeys(tm_ids):
        fetched = known.get(tm_id)
        if fetched and (t0 - date.fromisoformat(fetched)).days < STALE_DAYS:
            continue
        try:
            data = fetcher.json(tm.history_url(tm_id))
            (Path(hist_dir) / f"{tm_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            if not fetched:
                prof = tm.parse_profile(fetcher.html(tm.profile_url(tm_id), ".data-header"))
                players = pd.concat(
                    [players, pd.DataFrame([{"tm_id": tm_id, **prof, "fetched_at": today}])], ignore_index=True
                )
            else:
                players.loc[players["tm_id"] == tm_id, "fetched_at"] = today
        except Exception as e:  # network/anti-bot: skip this player, keep the run alive
            print(f"[history] {tm_id} FAILED {type(e).__name__}", flush=True)
            continue
        n += 1
        players[PLAYER_COLS].to_csv(players_csv, index=False)  # incremental: a crash keeps progress
        print(f"[history] {tm_id} updated", flush=True)
    players[PLAYER_COLS].to_csv(players_csv, index=False)
    return n


def load_transfers(hist_dir: Path = HISTORY) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(Path(hist_dir).glob("*.json")):
        rows.extend(tm.parse_history(json.loads(f.read_text(encoding="utf-8")), f.stem))
    return rows
