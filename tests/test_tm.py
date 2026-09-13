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
