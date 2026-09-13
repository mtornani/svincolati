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
