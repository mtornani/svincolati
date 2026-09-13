"""CSV -> docs/index.html. Static, inline CSS, no external scripts."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .paths import DOCS

CSS = """
body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111;background:#fff}
h1{margin:0 0 .2rem}h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:.4rem .5rem;border-bottom:1px solid #eee;text-align:left}
th{background:#f5f5f5}td.n{text-align:right;font-variant-numeric:tabular-nums}
.k{display:inline-block;margin:.3rem 1.2rem .3rem 0}.k b{font-size:1.6rem;display:block}
code{font-size:12px;word-break:break-all}.muted{color:#666;font-size:13px}
"""


def _eur(v) -> str:
    try:
        return f"{int(v):,}".replace(",", ".") + " €"
    except (TypeError, ValueError):
        return "-"


def _pct(x) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


def render(pred: pd.DataFrame, as_of: str, digest: str, board: dict, archive: list[tuple[str, str]],
           meta: dict) -> str:
    top = pred.sort_values("rank").head(10)
    rows = "".join(
        f"<tr><td class=n>{r['rank']}</td>"
        f"<td><a href='https://www.transfermarkt.it/x/profil/spieler/{r['tm_id']}'>{html.escape(str(r['name']))}</a></td>"
        f"<td>{r['role_group']}</td><td>{r['girone']}</td><td>{html.escape(str(r['prev_club']))}</td>"
        f"<td class=n>{_eur(r['mv_eur'])}</td><td class=n>{r['days_free']}</td>"
        f"<td class=n>{_pct(r['p30'])}</td><td class=n>{_pct(r['p60'])}</td></tr>"
        for r in top.to_dict("records")
    )
    arch = "".join(f"<li>{d} <code>{h}</code></li>" for d, h in reversed(archive)) or "<li class=muted>nessuna</li>"
    model_note = ("modello ML" if meta.get("use_model_30") else "baseline euristica") + f" · {meta.get('version', '')}"
    n = len(pred)
    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>svincolati · Serie C · {as_of}</title><style>{CSS}</style></head><body>
<h1>svincolati</h1>
<p class=muted>Chi firma per primo tra gli svincolati di Serie C. Predizioni pubbliche, immutabili, misurate. Nessun giudizio sulla qualità del giocatore.</p>

<h2>Top 10 · settimana {as_of}</h2>
<p class=muted>{n} svincolati valutati · stima: {model_note} · p30/p60 = probabilità di firma entro 30/60 giorni</p>
<table><thead><tr><th>#</th><th>Giocatore</th><th>Ruolo</th><th>Gir.</th><th>Ex club</th><th>Valore</th><th>Giorni svinc.</th><th>p30</th><th>p60</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class=muted>hash SHA-256 di <code>predictions/{as_of}.csv</code>: <code>{digest}</code></p>

<h2>Scoreboard</h2>
<div><span class=k><b>{board['n_resolved']}</b>predizioni risolte</span>
<span class=k><b>{board['hit30']}</b>firmati entro 30 gg</span>
<span class=k><b>{board['hit60']}</b>firmati entro 60 gg</span>
<span class=k><b>{board['top_hit30']}/{board['top_resolved']}</b>top-{board['top_n']} firmati entro 30 gg</span></div>
<p class=muted>calibrazione: p30 media {_pct(board['mean_p30'])} vs osservato {_pct(board['obs_rate30'])}. Una predizione si risolve quando il giocatore firma, si ritira, o passano 60 giorni.</p>

<h2>Archivio</h2><ul>{arch}</ul>

<h2>Metodo</h2>
<ol>
<li>Ogni lunedì: lista svincolati Serie C (gironi A, B, C) da Transfermarkt.</li>
<li>Per ogni giocatore: storico trasferimenti, età, ruolo, valore di mercato, giorni senza contratto.</li>
<li>Label storica: quanti giorni sono passati tra "svincolato" e firma successiva, su tutte le carriere raccolte.</li>
<li>Modello: gradient boosting (scikit-learn) contro baseline euristica. Il modello va live solo se batte la baseline su split temporale.</li>
<li>Predizioni scritte in CSV con hash. Mai modificate dopo la pubblicazione.</li>
<li>Esiti letti dallo stesso storico Transfermarkt nelle settimane successive. Append-only.</li>
</ol>
<p class=muted>Dettagli: <a href="https://github.com/mtornani/svincolati/blob/main/LABELING.md">LABELING.md</a> · codice MIT · dati pubblici Transfermarkt, uso non commerciale.</p>
</body></html>
"""


def write(html_text: str, docs_dir: Path = DOCS) -> Path:
    out = Path(docs_dir) / "index.html"
    out.write_text(html_text, encoding="utf-8")
    return out
