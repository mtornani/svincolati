import re

import pandas as pd

from svincolati import site


def test_html_static_and_complete(tmp_path):
    pred = pd.DataFrame([{"rank": 1, "tm_id": "1", "name": "P1 <x>", "girone": "A", "role_group": "GK", "prev_club": "X",
                          "mv_eur": 100000, "days_free": 74, "p30": 0.4, "p60": 0.7, "baseline": 5,
                          "model_version": "vtest"}])
    board = {"n_resolved": 3, "hit30": 1, "hit60": 1, "top_n": 10, "top_resolved": 3, "top_hit30": 1, "top_hit60": 1,
             "mean_p30": 0.5, "obs_rate30": 0.33}
    html = site.render(pred, "2026-09-13", "abc123", board, [("2026-09-06", "deadbeef")],
                       {"version": "vtest", "use_model_30": False, "use_model_60": False})
    assert "P1 &lt;x&gt;" in html and "2026-09-13" in html and "abc123" in html and "deadbeef" in html
    assert "100.000 €" in html and "40%" in html
    assert not re.search(r'<script[^>]+src=|<link[^>]+href="http', html)
    out = site.write(html, tmp_path)
    assert out.read_text(encoding="utf-8") == html
