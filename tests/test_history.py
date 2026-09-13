import json
from pathlib import Path

import pandas as pd

from svincolati import history

FX = Path(__file__).parent / "fixtures"


class FakeTM:
    def html(self, url, wait_selector):
        return (FX / "tm_profile.html").read_text(encoding="utf-8")

    def json(self, url):
        return json.loads((FX / "tm_history.json").read_text(encoding="utf-8"))


def test_update_fetches_new_and_skips_fresh(tmp_path):
    players_csv = tmp_path / "players.csv"
    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    n = history.update(FakeTM(), ["253267"], "2026-09-13", players_csv=players_csv, hist_dir=hist_dir)
    assert n == 1
    df = pd.read_csv(players_csv, dtype=str)
    assert df.iloc[0]["tm_id"] == "253267" and df.iloc[0]["birth_date"] == "1998-06-15"
    assert (hist_dir / "253267.json").exists()
    n2 = history.update(FakeTM(), ["253267"], "2026-09-20", players_csv=players_csv, hist_dir=hist_dir)
    assert n2 == 0


def test_load_transfers(tmp_path):
    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    (hist_dir / "253267.json").write_text((FX / "tm_history.json").read_text(encoding="utf-8"), encoding="utf-8")
    rows = history.load_transfers(hist_dir)
    assert rows and rows[0]["tm_id"] == "253267"


class FlakyTM(FakeTM):
    def json(self, url):
        if url.endswith("/1"):
            raise TimeoutError("boom")
        return super().json(url)


def test_update_skips_failed_player_and_keeps_progress(tmp_path):
    players_csv = tmp_path / "players.csv"
    hist_dir = tmp_path / "history"
    hist_dir.mkdir()
    n = history.update(FlakyTM(), ["1", "253267"], "2026-09-13", players_csv=players_csv, hist_dir=hist_dir)
    assert n == 1
    df = pd.read_csv(players_csv, dtype=str)
    assert list(df["tm_id"]) == ["253267"]
