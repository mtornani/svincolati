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
    if not math.isnan(f["age"]) and f["age"] < 23:
        s += 2
    if f["role_code"] in (0, 1):
        s += 2
    if not math.isnan(f["log_mv"]) and math.exp(f["log_mv"]) < 500_000:
        s += 1
    if f["days_free"] > 120:
        s -= 2
    return s
