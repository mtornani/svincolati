# svincolati Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Weekly pipeline that harvests Serie C free agents from Transfermarkt, predicts p(signs within 30/60 days), publishes immutable named predictions and a measured scoreboard as a static site.

**Architecture:** Pure-function parsers + label logic (unit tested on saved fixtures), one Playwright wrapper for fetching (integration only), CSV/JSON files in repo as the only storage, scikit-learn HistGradientBoosting vs a mandatory heuristic baseline, Python-generated HTML. One CLI `python -m svincolati <cmd>`.

**Tech Stack:** Python 3.12, playwright (Chromium headless), pandas, scikit-learn, pytest. No LLM, no DB, no cloud.

**Spec deviations locked here:** list page has no birth date → profile fetched once per player and cached in `data/players.csv`. Transfer history has no apps/goals → `apps_prev`, `goals_prev` are display-only, not model features.

**Project root:** `01_Products/svincolati/` (all paths below relative to it). Fixtures come from `%TEMP%\tm_list_fixture.html`, `%TEMP%\tm_profile_fixture.html`, `%TEMP%\tm_history_fixture.json` captured 2026-09-13.

---

## File structure

```
svincolati/
  pyproject.toml            metadata + pytest config
  LICENSE                   MIT
  README.md                 what/why/how, Task Scheduler line
  LABELING.md               label definition, edge cases
  svincolati/
    __init__.py
    __main__.py             -> cli.main()
    paths.py                ROOT and data paths
    tm.py                   pure parsers: list, profile, history, market value
    fetch.py                Playwright wrapper (rate-limited)
    harvest.py              lists -> data/snapshots/DATE.csv
    history.py              profile + ceapi -> data/players.csv, data/history/ID.json
    label.py                history -> spells
    features.py             feature vector, baseline score, leakage guard
    train.py                training rows, temporal split, eval, model files
    predict.py              snapshot -> predictions/DATE.csv + .sha256
    score.py                predictions + spells -> outcomes.csv + scoreboard
    site.py                 -> docs/index.html
    cli.py                  argparse subcommands + weekly
  tests/
    fixtures/tm_list.html tm_profile.html tm_history.json
    test_tm.py test_harvest.py test_history.py test_label.py test_features.py
    test_train.py test_predict.py test_score.py test_site.py
  data/snapshots/ data/history/ data/players.csv data/spells.csv data/outcomes.csv
  predictions/  models/  docs/  logs/
```

---

### Task 1: Scaffold + fixtures

**Files:** Create `pyproject.toml`, `LICENSE`, `.gitignore`, `svincolati/__init__.py`, `svincolati/paths.py`, `tests/fixtures/*`

- [ ] **Step 1: Create package skeleton**

```toml
# pyproject.toml
[project]
name = "svincolati"
version = "0.1.0"
description = "Predicts when Serie C free agents sign. Public, immutable, measured."
requires-python = ">=3.12"
dependencies = ["playwright>=1.45", "pandas>=2.2", "scikit-learn>=1.5"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# svincolati/paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SNAPSHOTS = DATA / "snapshots"
HISTORY = DATA / "history"
PLAYERS_CSV = DATA / "players.csv"
SPELLS_CSV = DATA / "spells.csv"
OUTCOMES_CSV = DATA / "outcomes.csv"
PREDICTIONS = ROOT / "predictions"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

for _p in (SNAPSHOTS, HISTORY, PREDICTIONS, MODELS, DOCS, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
```

`.gitignore`: `__pycache__/`, `*.pyc`, `logs/`, `.pytest_cache/`

- [ ] **Step 2: Copy fixtures** from `%TEMP%` into `tests/fixtures/` as `tm_list.html`, `tm_profile.html`, `tm_history.json`.
- [ ] **Step 3: Install deps, run empty pytest.** `pip install pandas scikit-learn pytest` (playwright present). `python -m pytest` → "no tests ran".
- [ ] **Step 4: Commit** `feat(svincolati): scaffold package and fixtures`

---

### Task 2: tm.py parsers (TDD on fixtures)

**Files:** Create `svincolati/tm.py`, `tests/test_tm.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_tm.py
import json
from pathlib import Path
from svincolati import tm

FX = Path(__file__).parent / "fixtures"

def test_parse_mv():
    assert tm.parse_mv("50 mila €") == 50_000
    assert tm.parse_mv("1,00 mln €") == 1_000_000
    assert tm.parse_mv("-") is None
    assert tm.parse_mv("") is None

def test_parse_free_list():
    rows = tm.parse_free_list((FX / "tm_list.html").read_text(encoding="utf-8"))
    assert len(rows) >= 20
    r = rows[0]
    assert r["tm_id"] == "1105892"
    assert r["name"] == "Alessandro De Luca"
    assert r["role_group"] == "MF"
    assert r["prev_club"] == "Portogruaro Calcio"
    assert r["nationality"] == "Italia"
    assert r["apps_prev"] == 0 and r["goals_prev"] == 0
    assert r["mv_eur"] == 10_000
    assert len({x["tm_id"] for x in rows}) == len(rows)

def test_parse_profile():
    p = tm.parse_profile((FX / "tm_profile.html").read_text(encoding="utf-8"))
    assert p["name"] == "Hachim Mastour"
    assert p["birth_date"] == "1998-06-15"
    assert p["position"] == "Centrocampo - Trequartista"
    assert "Marocco" in p["nationality"]

def test_parse_history():
    data = json.loads((FX / "tm_history.json").read_text(encoding="utf-8"))
    rows = tm.parse_history(data, "253267")
    assert rows[0]["date"] < rows[-1]["date"]
    last = rows[-1]
    assert last["to_club"] == "svincolato" and last["to_special"] is True
    assert last["mv_eur"] == 50_000
    assert all(r["tm_id"] == "253267" for r in rows)

def test_role_group():
    assert tm.role_group("Centrocampo - Trequartista") == "MF"
    assert tm.role_group("Porta") == "GK"
    assert tm.role_group("") == "UNK"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_tm.py -v` → FAIL (no module `tm`).
- [ ] **Step 3: Implement**

```python
# svincolati/tm.py
"""Pure Transfermarkt parsers. No network. Fixtures in tests/fixtures."""
from __future__ import annotations
import html as _html
import re

FREE = "svincolato"
ROLE_MAP = {"Porta": "GK", "Difesa": "DF", "Centrocampo": "MF", "Attacco": "FW"}
BASE = "https://www.transfermarkt.it"
LISTS = {
    "A": f"{BASE}/serie-c-girone-a/vertragslosespieler/wettbewerb/IT3A",
    "B": f"{BASE}/serie-c-girone-b/vertragslosespieler/wettbewerb/IT3B",
    "C": f"{BASE}/serie-c-girone-c/vertragslosespieler/wettbewerb/IT3C",
}

def list_url(girone: str, page: int) -> str:
    u = LISTS[girone]
    return u if page == 1 else f"{u}/page/{page}"

def profile_url(tm_id: str) -> str:
    return f"{BASE}/x/profil/spieler/{tm_id}"

def history_url(tm_id: str) -> str:
    return f"{BASE}/ceapi/transferHistory/list/{tm_id}"

def strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = _html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()

def parse_mv(text: str | None) -> int | None:
    t = _html.unescape(text or "").replace("\xa0", " ")
    if "€" not in t:
        return None
    m = re.search(r"([\d.,]+)\s*(mln|mila|m|k)?", t, re.I)
    if not m:
        return None
    raw, unit = m.group(1), (m.group(2) or "").lower()
    try:
        val = float(raw.replace(".", "").replace(",", "."))
    except ValueError:
        return None
    if unit in ("mln", "m"):
        val *= 1_000_000
    elif unit in ("mila", "k"):
        val *= 1_000
    return int(val)

def role_group(position: str) -> str:
    head = (position or "").split(" - ")[0].strip()
    return ROLE_MAP.get(head, "UNK")

def _int(s: str) -> int:
    s = strip_tags(s)
    return int(s) if s.isdigit() else 0

def parse_free_list(html: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for m in re.finditer(r'<tr class="(?:odd|even)">(.*?)</tr>', html, re.S):
        tr = m.group(1)
        who = re.search(
            r'<a title="[^"]*" href="/[^"]*?/profil/spieler/(\d+)">([^<]*)</a><br>([^<]*)</td>', tr
        )
        if not who:
            continue
        tm_id, name, role = who.group(1), strip_tags(who.group(2)), strip_tags(who.group(3))
        if tm_id in seen:
            continue
        seen.add(tm_id)
        club = re.search(r'<td class="zentriert"><a title="([^"]*)" href="/[^"]*/verein/', tr)
        flags = re.findall(r'<img src="[^"]*flagge[^"]*" title="([^"]*)"', tr)
        nums = re.search(
            r'<td class="zentriert">([^<]*)</td>\s*<td class="zentriert">([^<]*)</td>\s*'
            r'<td class="zentriert">([^<]*)</td>\s*<td class="rechts">([^<]*)</td>', tr
        )
        rows.append({
            "tm_id": tm_id,
            "name": name,
            "role_group": role_group(role),
            "prev_club": strip_tags(club.group(1)) if club else "",
            "nationality": "/".join(flags),
            "apps_prev": _int(nums.group(1)) if nums else 0,
            "goals_prev": _int(nums.group(2)) if nums else 0,
            "mv_eur": parse_mv(nums.group(4)) if nums else None,
        })
    return rows

def _field(html: str, label: str) -> str:
    m = re.search(label + r":?\s*</span>\s*<span[^>]*>(.*?)</span>", html, re.S)
    return m.group(1) if m else ""

def parse_profile(html: str) -> dict:
    out: dict = {}
    h1 = re.search(r"data-header__headline-wrapper[^>]*>(.*?)</h1>", html, re.S)
    out["name"] = re.sub(r"^#\d+\s*", "", strip_tags(h1.group(1))) if h1 else ""
    bd = re.search(r"(\d{2})/(\d{2})/(\d{4})", strip_tags(_field(html, "Nato il")))
    out["birth_date"] = f"{bd.group(3)}-{bd.group(2)}-{bd.group(1)}" if bd else ""
    out["position"] = strip_tags(_field(html, "Posizione"))
    nat_raw = _field(html, "Nazionalità")
    titles = re.findall(r'title="([^"]*)"', nat_raw)
    out["nationality"] = "/".join(titles) if titles else strip_tags(nat_raw)
    return out

def parse_history(data: dict, tm_id: str) -> list[dict]:
    rows = []
    for t in data.get("transfers", []):
        if t.get("upcoming") or t.get("futureTransfer"):
            continue
        d = t.get("dateUnformatted") or ""
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            continue
        rows.append({
            "tm_id": tm_id,
            "date": d,
            "from_club": t["from"]["clubName"],
            "from_special": bool(t["from"].get("isSpecial")),
            "to_club": t["to"]["clubName"],
            "to_special": bool(t["to"].get("isSpecial")),
            "mv_eur": parse_mv(t.get("marketValue")),
        })
    rows.sort(key=lambda r: r["date"])
    return rows
```

- [ ] **Step 4: Run tests** → PASS. If a regex misses the fixture, fix the regex against the fixture, never the test expectation.
- [ ] **Step 5: Commit** `feat(svincolati): TM parsers with fixture tests`

---

### Task 3: fetch.py Playwright wrapper

**Files:** Create `svincolati/fetch.py`. No unit test (network). Smoke in Task 13.

```python
# svincolati/fetch.py
"""Rate-limited Playwright fetcher. Only file that touches the network."""
from __future__ import annotations
import json
import time
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

class TM:
    def __init__(self, sleep_s: float = 1.5, headless: bool = True):
        self.sleep_s = sleep_s
        self.headless = headless

    def __enter__(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.headless)
        self._page = self._browser.new_context(locale="it-IT", user_agent=UA).new_page()
        return self

    def __exit__(self, *exc):
        self._browser.close()
        self._pw.stop()

    def _goto(self, url: str, wait_selector: str | None) -> None:
        for attempt in range(2):
            try:
                self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                if wait_selector:
                    self._page.wait_for_selector(wait_selector, timeout=30_000)
                return
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(5)

    def html(self, url: str, wait_selector: str) -> str:
        self._goto(url, wait_selector)
        time.sleep(self.sleep_s)
        return self._page.content()

    def json(self, url: str) -> dict:
        self._goto(url, None)
        txt = self._page.inner_text("body")
        time.sleep(self.sleep_s)
        return json.loads(txt)
```

- [ ] Commit `feat(svincolati): playwright fetcher`

---

### Task 4: harvest.py

**Files:** Create `svincolati/harvest.py`, `tests/test_harvest.py`

- [ ] **Step 1: Failing test** (fake fetcher, no network)

```python
# tests/test_harvest.py
from pathlib import Path
import pandas as pd
from svincolati import harvest

FX = Path(__file__).parent / "fixtures"

class FakeTM:
    def __init__(self): self.calls = []
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
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

```python
# svincolati/harvest.py
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
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): harvest free-agent lists to snapshot CSV`

---

### Task 5: history.py (profile + transfers cache)

**Files:** Create `svincolati/history.py`, `tests/test_history.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_history.py
import json
from pathlib import Path
import pandas as pd
from svincolati import history

FX = Path(__file__).parent / "fixtures"

class FakeTM:
    def html(self, url, wait_selector): return (FX / "tm_profile.html").read_text(encoding="utf-8")
    def json(self, url): return json.loads((FX / "tm_history.json").read_text(encoding="utf-8"))

def test_update_fetches_new_and_skips_fresh(tmp_path):
    players_csv = tmp_path / "players.csv"; hist_dir = tmp_path / "history"; hist_dir.mkdir()
    n = history.update(FakeTM(), ["253267"], "2026-09-13", players_csv=players_csv, hist_dir=hist_dir)
    assert n == 1
    df = pd.read_csv(players_csv, dtype=str)
    assert df.iloc[0]["tm_id"] == "253267" and df.iloc[0]["birth_date"] == "1998-06-15"
    assert (hist_dir / "253267.json").exists()
    n2 = history.update(FakeTM(), ["253267"], "2026-09-20", players_csv=players_csv, hist_dir=hist_dir)
    assert n2 == 0

def test_load_transfers(tmp_path):
    hist_dir = tmp_path / "history"; hist_dir.mkdir()
    (hist_dir / "253267.json").write_text((FX / "tm_history.json").read_text(encoding="utf-8"), encoding="utf-8")
    rows = history.load_transfers(hist_dir)
    assert rows and rows[0]["tm_id"] == "253267"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/history.py
"""Per-player profile (once) + transfer history (refresh >30d) cache."""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import pandas as pd
from . import tm
from .paths import PLAYERS_CSV, HISTORY

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
        data = fetcher.json(tm.history_url(tm_id))
        (Path(hist_dir) / f"{tm_id}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if not fetched:
            prof = tm.parse_profile(fetcher.html(tm.profile_url(tm_id), ".data-header"))
            players = pd.concat([players, pd.DataFrame([{"tm_id": tm_id, **prof, "fetched_at": today}])],
                                ignore_index=True)
        else:
            players.loc[players["tm_id"] == tm_id, "fetched_at"] = today
        n += 1
        print(f"[history] {tm_id} updated")
    players[PLAYER_COLS].to_csv(players_csv, index=False)
    return n

def load_transfers(hist_dir: Path = HISTORY) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(Path(hist_dir).glob("*.json")):
        rows.extend(tm.parse_history(json.loads(f.read_text(encoding="utf-8")), f.stem))
    return rows
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): player profile and transfer history cache`

---

### Task 6: label.py

**Files:** Create `svincolati/label.py`, `tests/test_label.py`, `LABELING.md`

- [ ] **Step 1: Failing tests**

```python
# tests/test_label.py
from datetime import date
from svincolati import label

def T(d, frm, to, special_to=False, mv=None):
    return {"tm_id": "1", "date": d, "from_club": frm, "from_special": frm == "svincolato",
            "to_club": to, "to_special": special_to, "mv_eur": mv}

def test_closed_and_open_spells():
    tr = [T("2019-03-04", "Lamia", "svincolato", True, 100_000),
          T("2019-10-18", "svincolato", "Reggina"),
          T("2026-07-01", "Verona", "svincolato", True, 50_000)]
    sp = label.spells_from_transfers(tr, date(2026, 9, 13))
    assert len(sp) == 2
    assert sp[0]["start"] == date(2019, 3, 4) and sp[0]["end"] == date(2019, 10, 18)
    assert sp[0]["days"] == 228 and sp[0]["next_club"] == "Reggina" and sp[0]["mv_at_start"] == 100_000
    assert sp[1]["end"] is None and sp[1]["days"] is None

def test_retirement_closes_spell_as_retired():
    tr = [T("2024-07-01", "X", "svincolato", True), T("2025-01-01", "svincolato", "Carriera terminata", True)]
    sp = label.spells_from_transfers(tr, date(2026, 1, 1))
    assert sp[0]["retired"] is True and sp[0]["end"] == date(2025, 1, 1)

def test_orphan_close_ignored_and_order_independent():
    tr = [T("2020-01-01", "svincolato", "Club"), T("2021-07-01", "Club", "svincolato", True)]
    sp = label.spells_from_transfers(list(reversed(tr)), date(2022, 1, 1))
    assert len(sp) == 1 and sp[0]["start"] == date(2021, 7, 1)

def test_build_spells_groups_by_player():
    tr = [T("2021-07-01", "A", "svincolato", True), {**T("2022-07-01", "B", "svincolato", True), "tm_id": "2"}]
    df = label.build_spells(tr, date(2023, 1, 1))
    assert set(df["tm_id"]) == {"1", "2"} and list(df.columns) == label.COLS
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/label.py
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
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Write LABELING.md** (Italian, ~40 lines): spell definition, label_30/label_60, censoring (open spell only counted if ≥60 days observed), retirement = negative, orphan close ignored, outcome measured from prediction date, known limits (TM date granularity, missing histories, retroactive TM edits).
- [ ] **Step 6: Commit** `feat(svincolati): spell labeling with tests and LABELING.md`

---

### Task 7: features.py

**Files:** Create `svincolati/features.py`, `tests/test_features.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_features.py
from datetime import date
import math, pytest
from svincolati import features as F

P = {"tm_id": "1", "birth_date": "2004-01-10", "position": "Porta", "nationality": "Italia"}
S = {"tm_id": "1", "start": date(2026, 7, 1), "end": None, "mv_at_start": 100_000, "retired": False}

def test_vector():
    f = F.build(P, S, [], date(2026, 9, 13))
    assert f["age"] == pytest.approx(22.68, abs=0.01)
    assert f["role_code"] == 0 and f["nat_it"] == 1 and f["days_free"] == 74
    assert f["log_mv"] == pytest.approx(math.log(100_000)) and f["n_prior"] == 0
    assert math.isnan(f["prior_mean_days"]) and f["window_open"] == 0 and f["month"] == 9
    assert list(f) == F.NAMES

def test_missing_mv_and_birth_are_nan():
    f = F.build({**P, "birth_date": ""}, {**S, "mv_at_start": None}, [], date(2026, 9, 13))
    assert math.isnan(f["age"]) and math.isnan(f["log_mv"])

def test_leakage_guard():
    with pytest.raises(AssertionError):
        F.build(P, S, [{"end": date(2026, 10, 1), "days": 10}], date(2026, 9, 13))
    with pytest.raises(AssertionError):
        F.build(P, {**S, "start": date(2026, 10, 1)}, [], date(2026, 9, 13))

def test_baseline():
    f = F.build(P, S, [], date(2026, 9, 13))
    assert F.baseline(f) == 5
    f2 = F.build({**P, "birth_date": "1995-01-01", "position": "Attacco"}, {**S, "start": date(2026, 1, 1)}, [], date(2026, 9, 13))
    assert F.baseline(f2) == -1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/features.py
"""Feature vector as of a date. Every input must be dated <= as_of (asserted)."""
from __future__ import annotations
import math
from datetime import date
from .tm import role_group

NAMES = ["age", "role_code", "log_mv", "days_free", "month", "nat_it", "n_prior", "prior_mean_days", "window_open"]
ROLE_CODE = {"GK": 0, "DF": 1, "MF": 2, "FW": 3, "UNK": 4}
NAN = float("nan")

def window_open(d: date) -> int:
    summer = (d.month in (7, 8)) or (d.month == 9 and d.day <= 1)
    winter = d.month == 1 or (d.month == 2 and d.day == 1)
    return int(summer or winter)

def build(player: dict, spell: dict, prior_spells: list[dict], as_of: date) -> dict:
    assert spell["start"] <= as_of, "spell starts after as_of"
    for p in prior_spells:
        assert p["end"] is not None and p["end"] <= as_of, "prior spell not closed before as_of"
    bd = player.get("birth_date") or ""
    age = (as_of - date.fromisoformat(bd)).days / 365.25 if bd else NAN
    mv = spell.get("mv_at_start")
    log_mv = math.log(mv) if mv and mv == mv and mv > 0 else NAN
    prior_days = [p["days"] for p in prior_spells if p.get("days") is not None and p["days"] == p["days"]]
    return {
        "age": age,
        "role_code": ROLE_CODE[role_group(player.get("position", ""))],
        "log_mv": log_mv,
        "days_free": (as_of - spell["start"]).days,
        "month": as_of.month,
        "nat_it": int("Italia" in (player.get("nationality") or "")),
        "n_prior": len(prior_spells),
        "prior_mean_days": (sum(prior_days) / len(prior_days)) if prior_days else NAN,
        "window_open": window_open(as_of),
    }

def baseline(f: dict) -> int:
    s = 0
    if not math.isnan(f["age"]) and f["age"] < 23: s += 2
    if f["role_code"] in (0, 1): s += 2
    if not math.isnan(f["log_mv"]) and math.exp(f["log_mv"]) < 500_000: s += 1
    if f["days_free"] > 120: s -= 2
    return s
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): features with leakage guard and baseline`

---

### Task 8: train.py

**Files:** Create `svincolati/train.py`, `tests/test_train.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_train.py
from datetime import date, timedelta
import random
import pandas as pd
from svincolati import train

def _players(n):
    return pd.DataFrame([{"tm_id": str(i), "name": f"P{i}", "birth_date": f"{1995 + i % 12}-05-01",
                          "position": ["Porta", "Difesa", "Centrocampo", "Attacco"][i % 4],
                          "nationality": "Italia" if i % 3 else "Francia", "fetched_at": "2026-09-13"}
                         for i in range(n)])

def _spells(n, seed=0):
    rnd = random.Random(seed); rows = []
    for i in range(n):
        start = date(2018 + i % 7, 7, 1)
        quick = (i % 4 in (0, 1)) and (i % 12 < 8)
        days = rnd.randint(5, 40) if quick else rnd.randint(30, 300)
        rows.append({"tm_id": str(i), "start": start, "end": start + timedelta(days=days),
                     "next_club": "X", "days": days, "mv_at_start": 50_000 * (1 + i % 5), "retired": False})
    return pd.DataFrame(rows)

def test_training_rows_labels_and_censoring():
    players = _players(2)
    sp = pd.DataFrame([{"tm_id": "0", "start": date(2026, 1, 1), "end": date(2026, 2, 15), "next_club": "X", "days": 45, "mv_at_start": 1, "retired": False},
                       {"tm_id": "1", "start": date(2026, 8, 1), "end": None, "next_club": "", "days": None, "mv_at_start": 1, "retired": False}])
    rows = train.training_rows(players, sp, date(2026, 9, 13), step=14)
    r0 = [r for r in rows if r["tm_id"] == "0"]
    assert [r["y30"] for r in r0] == [0, 0, 1, 1]
    assert all(r["y60"] == 1 for r in r0)
    assert not [r for r in rows if r["tm_id"] == "1"]

def test_fit_writes_artifacts(tmp_path):
    meta = train.fit(_players(200), _spells(200), date(2026, 9, 13), models_dir=tmp_path)
    assert (tmp_path / f"meta_{meta['version']}.json").exists() and (tmp_path / "meta_latest.json").exists()
    assert (tmp_path / f"eval_{meta['version']}.md").exists()
    for h in (30, 60):
        assert f"auc_model_{h}" in meta and f"auc_baseline_{h}" in meta and f"use_model_{h}" in meta
        assert (tmp_path / f"model_{meta['version']}_{h}.pkl").exists()
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/train.py
"""Training rows (multiple as_of per spell), temporal split, model vs baseline, artifacts."""
from __future__ import annotations
import json, pickle
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
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
    meta = {"version": version, "trained_at": today.isoformat(), "n_rows": int(len(df)),
            "n_train": int(len(tr)), "n_test": int(len(te)), "cutoff": cutoff.isoformat(), "features": F.NAMES}
    lines = [f"# eval {version}", "", f"rows {len(df)} | train {len(tr)} | test {len(te)} | cutoff {cutoff}", ""]
    for h in HORIZONS:
        y = f"y{h}"
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=0)
        clf.fit(tr[F.NAMES], tr[y])
        p = clf.predict_proba(te[F.NAMES])[:, 1]
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
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): training with temporal split and baseline gate`

---

### Task 9: predict.py

**Files:** Create `svincolati/predict.py`, `tests/test_predict.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_predict.py
from datetime import date
import hashlib, json, pickle
import pandas as pd, pytest
from svincolati import predict

class Const:
    def predict_proba(self, X): return [[0.1, 0.9]] * len(X)

def _setup(tmp_path, use_model):
    snap = pd.DataFrame([{"snapshot_date": "2026-09-13", "girone": "A", "tm_id": "1", "name": "P1", "role_group": "GK",
                          "prev_club": "X", "nationality": "Italia", "apps_prev": 3, "goals_prev": 0, "mv_eur": 100000}])
    players = pd.DataFrame([{"tm_id": "1", "name": "P1", "birth_date": "2004-01-10", "position": "Porta", "nationality": "Italia", "fetched_at": "2026-09-13"}])
    spells = pd.DataFrame([{"tm_id": "1", "start": date(2026, 7, 1), "end": None, "next_club": "", "days": None, "mv_at_start": 100000, "retired": False}])
    meta = {"version": "vtest", "use_model_30": use_model, "use_model_60": use_model,
            "baseline_calib_30": {"5": 0.4}, "baseline_calib_60": {"5": 0.7}, "base_rate_30": 0.2, "base_rate_60": 0.3}
    for h in (30, 60):
        with open(tmp_path / f"model_vtest_{h}.pkl", "wb") as fh: pickle.dump(Const(), fh)
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
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/predict.py
"""Snapshot -> predictions/DATE.csv (+ .sha256). Never overwrites."""
from __future__ import annotations
import hashlib, json, pickle
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
        X = pd.DataFrame([f])[F.NAMES]
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
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): immutable hashed predictions`

---

### Task 10: score.py

**Files:** Create `svincolati/score.py`, `tests/test_score.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_score.py
from datetime import date
import pandas as pd
from svincolati import score

def _spells():
    return pd.DataFrame([
        {"tm_id": "1", "start": date(2026, 7, 1), "end": date(2026, 8, 20), "next_club": "Rimini", "days": 50, "mv_at_start": 1, "retired": False},
        {"tm_id": "2", "start": date(2026, 7, 1), "end": None, "next_club": "", "days": None, "mv_at_start": 1, "retired": False},
        {"tm_id": "3", "start": date(2026, 7, 1), "end": date(2026, 8, 1), "next_club": "Carriera terminata", "days": 31, "mv_at_start": 1, "retired": True},
    ])

def _pred(d="2026-08-01"):
    return pd.DataFrame([{"rank": i + 1, "tm_id": t, "name": f"P{t}", "p30": p, "p60": p + 0.1}
                         for i, (t, p) in enumerate([("1", 0.8), ("2", 0.5), ("3", 0.3)])]).assign(prediction_date=d)

def test_resolve_statuses():
    assert score.resolve(date(2026, 8, 1), "1", _spells(), date(2026, 9, 13)) == ("signed", date(2026, 8, 20), 19, "Rimini")
    assert score.resolve(date(2026, 8, 1), "2", _spells(), date(2026, 10, 15))[0] == "expired"
    assert score.resolve(date(2026, 8, 1), "3", _spells(), date(2026, 9, 13))[0] == "retired"

def test_pending_until_60_days():
    assert score.resolve(date(2026, 8, 1), "2", _spells(), date(2026, 9, 1)) is None

def test_update_outcomes_append_only(tmp_path):
    out = tmp_path / "outcomes.csv"
    df = score.update_outcomes([_pred()], _spells(), date(2026, 9, 1), out)
    assert set(df["tm_id"]) == {"1", "3"}
    df2 = score.update_outcomes([_pred()], _spells(), date(2026, 10, 15), out)
    assert set(df2["tm_id"]) == {"1", "2", "3"} and len(df2) == 3
    board = score.scoreboard(df2)
    assert board["n_resolved"] == 3 and board["hit30"] == 1 and board["hit60"] == 1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

```python
# svincolati/score.py
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
        if s["start"] <= pred_date and (end is None or end > pred_date):
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
            new.append({"prediction_date": r["prediction_date"], "tm_id": r["tm_id"], "name": r["name"], "rank": r["rank"],
                        "p30": r["p30"], "p60": r["p60"], "status": status,
                        "signed_date": sd.isoformat() if sd else "", "club": club, "days": days if days is not None else ""})
    df = pd.concat([old, pd.DataFrame(new, columns=COLS)], ignore_index=True)
    df.to_csv(path, index=False)
    return df

def scoreboard(outcomes: pd.DataFrame, top_n: int = 10) -> dict:
    df = outcomes.copy()
    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    signed = df["status"] == "signed"
    top = df[df["rank"] <= top_n]
    tsigned = top["status"] == "signed"
    return {
        "n_resolved": int(len(df)),
        "hit30": int((signed & (df["days"] <= 30)).sum()),
        "hit60": int((signed & (df["days"] <= 60)).sum()),
        "top_n": top_n, "top_resolved": int(len(top)),
        "top_hit30": int((tsigned & (top["days"] <= 30)).sum()),
        "top_hit60": int((tsigned & (top["days"] <= 60)).sum()),
        "mean_p30": float(df["p30"].mean()) if len(df) else float("nan"),
        "obs_rate30": float((signed & (df["days"] <= 30)).mean()) if len(df) else float("nan"),
    }
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): outcomes and scoreboard`

---

### Task 11: site.py

**Files:** Create `svincolati/site.py`, `tests/test_site.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_site.py
import re
import pandas as pd
from svincolati import site

def test_html_static_and_complete(tmp_path):
    pred = pd.DataFrame([{"rank": 1, "tm_id": "1", "name": "P1", "girone": "A", "role_group": "GK", "prev_club": "X",
                          "mv_eur": 100000, "days_free": 74, "p30": 0.4, "p60": 0.7, "baseline": 5, "model_version": "vtest"}])
    board = {"n_resolved": 3, "hit30": 1, "hit60": 1, "top_n": 10, "top_resolved": 3, "top_hit30": 1, "top_hit60": 1,
             "mean_p30": 0.5, "obs_rate30": 0.33}
    html = site.render(pred, "2026-09-13", "abc123", board, [("2026-09-06", "deadbeef")], {"version": "vtest", "use_model_30": False, "use_model_60": False})
    assert "P1" in html and "2026-09-13" in html and "abc123" in html and "deadbeef" in html
    assert not re.search(r'<script[^>]+src=|<link[^>]+href="http', html)
    out = site.write(html, tmp_path)
    assert out.read_text(encoding="utf-8") == html
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** single f-string template, inline CSS, TM profile links allowed. Sections: header, top-10 table (rank, name→TM link, ruolo, girone, ex club, valore, giorni svincolato, p30, p60), scoreboard, archive with hashes, method in 10 lines + link LABELING.md, footer with prediction hash, model version, "baseline in uso" flag. `render(pred_df, as_of, digest, board, archive, meta) -> str`; `write(html, docs_dir=DOCS) -> Path` writes `index.html`. Under 120 lines.
- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `feat(svincolati): static site renderer`

---

### Task 12: cli.py + weekly + README

**Files:** Create `svincolati/cli.py`, `svincolati/__main__.py`, `README.md`

```python
# svincolati/cli.py
from __future__ import annotations
import argparse, hashlib, json
from datetime import date
import pandas as pd
from . import harvest, history, label, train, predict, score, site
from .fetch import TM
from .paths import SNAPSHOTS, SPELLS_CSV, OUTCOMES_CSV, PREDICTIONS, MODELS

def today() -> str: return date.today().isoformat()

def cmd_harvest(a):
    with TM(headless=not a.headful) as f: harvest.harvest(f, a.date)

def cmd_history(a):
    snap = pd.read_csv(SNAPSHOTS / f"{a.date}.csv", dtype={"tm_id": str})
    with TM(headless=not a.headful) as f: history.update(f, snap["tm_id"].tolist(), a.date)

def cmd_label(a):
    df = label.build_spells(history.load_transfers(), date.fromisoformat(a.date)); df.to_csv(SPELLS_CSV, index=False)
    print(f"[label] {len(df)} spells, {df['end'].isna().sum()} open")

def cmd_train(a):
    meta = train.fit(history.load_players(), label.load_spells(SPELLS_CSV), date.fromisoformat(a.date))
    print(json.dumps({k: v for k, v in meta.items() if k.startswith(("auc", "use", "n_"))}, indent=1))

def cmd_predict(a):
    snap = pd.read_csv(SNAPSHOTS / f"{a.date}.csv", dtype={"tm_id": str})
    out = predict.run(snap, history.load_players(), label.load_spells(SPELLS_CSV), a.date); print(f"[predict] {out}")

def cmd_score(a):
    preds = [pd.read_csv(p, dtype={"tm_id": str}).assign(prediction_date=p.stem) for p in sorted(PREDICTIONS.glob("*.csv"))]
    df = score.update_outcomes(preds, label.load_spells(SPELLS_CSV), date.fromisoformat(a.date))
    print(json.dumps(score.scoreboard(df), indent=1))

def cmd_site(a):
    files = sorted(PREDICTIONS.glob("*.csv"))
    latest = files[-1]; pred = pd.read_csv(latest, dtype={"tm_id": str})
    digest = hashlib.sha256(latest.read_bytes()).hexdigest()
    archive = [(p.stem, hashlib.sha256(p.read_bytes()).hexdigest()) for p in files[:-1]]
    outcomes = pd.read_csv(OUTCOMES_CSV, dtype={"tm_id": str}) if OUTCOMES_CSV.exists() else pd.DataFrame(columns=score.COLS)
    meta = json.loads((MODELS / "meta_latest.json").read_text(encoding="utf-8"))
    site.write(site.render(pred, latest.stem, digest, score.scoreboard(outcomes), archive, meta)); print("[site] docs/index.html")

def cmd_weekly(a):
    for fn in (cmd_harvest, cmd_history, cmd_label, cmd_score, cmd_predict, cmd_site): fn(a)

def main(argv=None):
    p = argparse.ArgumentParser(prog="svincolati")
    p.add_argument("--date", default=today()); p.add_argument("--headful", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("harvest", cmd_harvest), ("history", cmd_history), ("label", cmd_label), ("train", cmd_train),
                     ("predict", cmd_predict), ("score", cmd_score), ("site", cmd_site), ("weekly", cmd_weekly)]:
        sub.add_parser(name).set_defaults(fn=fn)
    a = p.parse_args(argv); a.fn(a)
```

`__main__.py`: `from .cli import main; main()`.

README (Italian, concise): what, why scoreboard, pipeline table, install (`pip install playwright pandas scikit-learn pytest`, `playwright install chromium`), `python -m svincolati weekly`, Task Scheduler:

```
schtasks /Create /SC WEEKLY /D MON /ST 07:00 /TN svincolati_weekly /TR "cmd /c cd /d D:\AI\01_Products\svincolati && python -m svincolati weekly >> logs\weekly.log 2>&1"
```

Immutability rule, data source + rate limit + attribution, non-goals, MIT.

- [ ] Commit `feat(svincolati): CLI, weekly command, README`

---

### Task 13: First real run (cold start)

- [ ] `python -m svincolati harvest` → ~60-80 rows per girone in `data/snapshots/2026-09-13.csv`.
- [ ] `python -m svincolati history` → ~200 profile+history fetches at 1.5 s → ~10 min. Check `data/players.csv` birth_date non-empty ≥95%.
- [ ] `python -m svincolati label` → `data/spells.csv`; print open vs closed counts.
- [ ] `python -m svincolati train` → read `models/eval_v2026-09-13.md`. Record AUCs in commit message. If closed spells < 100, note in README "cold-start thin, v1.1 expands to full squads".
- [ ] `python -m svincolati predict` → `predictions/2026-09-13.csv` + sha256.
- [ ] `python -m svincolati score` → outcomes empty. Expected.
- [ ] `python -m svincolati site` → open `docs/index.html`, check top-10, hash, method.
- [ ] `python -m pytest` → all green.
- [ ] Commit data + predictions + models + docs: `feat(svincolati): first weekly cycle 2026-09-13`.

---

## Self-review

- Spec §3 sources → Tasks 2,3. §4 data model → Tasks 4,5,6,9,10. §5 pipeline → Task 12. §6 label → Task 6. §7 features (minus apps/goals, deviation stated) → Task 7. §8 baseline gate → Task 8. §9 site → Task 11. §10 tests → each task. §11 risks: `--headful` flag; thin cold-start noted in Task 13. §13 layout → Task 1.
- Names consistent across tasks: `F.build`, `F.baseline`, `F.NAMES`, `label.build_spells/load_spells/COLS`, `history.update/load_players/load_transfers`, `predict.run`, `score.resolve/update_outcomes/scoreboard/COLS`, `site.render/write`, `train.training_rows/fit`.
- No placeholders except site template body (structure fully specified, ≤120 lines).
