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


def test_load_spells_roundtrip(tmp_path):
    tr = [T("2021-07-01", "A", "svincolato", True, 10_000), T("2021-08-01", "svincolato", "B"),
          T("2026-07-01", "B", "svincolato", True)]
    df = label.build_spells(tr, date(2026, 9, 13))
    p = tmp_path / "spells.csv"
    df.to_csv(p, index=False)
    back = label.load_spells(p)
    assert back.iloc[0]["end"] == date(2021, 8, 1) and back.iloc[1]["end"] is None
    assert back.iloc[0]["retired"] is False or back.iloc[0]["retired"] == False  # noqa: E712
