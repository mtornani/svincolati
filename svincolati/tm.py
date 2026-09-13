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
            r'<td class="zentriert">([^<]*)</td>\s*<td class="rechts">([^<]*)</td>',
            tr,
        )
        rows.append(
            {
                "tm_id": tm_id,
                "name": name,
                "role_group": role_group(role),
                "prev_club": strip_tags(club.group(1)) if club else "",
                "nationality": "/".join(flags),
                "apps_prev": _int(nums.group(1)) if nums else 0,
                "goals_prev": _int(nums.group(2)) if nums else 0,
                "mv_eur": parse_mv(nums.group(4)) if nums else None,
            }
        )
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
        rows.append(
            {
                "tm_id": tm_id,
                "date": d,
                "from_club": t["from"]["clubName"],
                "from_special": bool(t["from"].get("isSpecial")),
                "to_club": t["to"]["clubName"],
                "to_special": bool(t["to"].get("isSpecial")),
                "mv_eur": parse_mv(t.get("marketValue")),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows
