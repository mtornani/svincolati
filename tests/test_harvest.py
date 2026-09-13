from pathlib import Path

import pandas as pd

from svincolati import harvest

FX = Path(__file__).parent / "fixtures"


class FakeTM:
    def __init__(self):
        self.calls = []

    def html(self, url, wait_selector):
        self.calls.append(url)
        return (FX / "tm_list.html").read_text(encoding="utf-8") if url.endswith("IT3A") else "<html></html>"


def test_harvest_writes_snapshot(tmp_path):
    out = harvest.harvest(FakeTM(), "2026-09-13", gironi=("A",), out_dir=tmp_path)
    df = pd.read_csv(out, dtype={"tm_id": str})
    assert set(df.columns) >= {"snapshot_date", "girone", "tm_id", "name", "role_group",
                               "prev_club", "nationality", "apps_prev", "goals_prev", "mv_eur"}
    assert (df["snapshot_date"] == "2026-09-13").all()
    assert (df["girone"] == "A").all()
    assert len(df) >= 20
