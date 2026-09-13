import math
from datetime import date

import pytest

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
    f2 = F.build({**P, "birth_date": "1995-01-01", "position": "Attacco"}, {**S, "start": date(2026, 1, 1)}, [],
                 date(2026, 9, 13))
    assert F.baseline(f2) == -1
