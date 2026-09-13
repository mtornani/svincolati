# svincolati — design v1

**Data:** 2026-09-13
**Una riga:** predice quando uno svincolato di Serie C firma. Predizioni pubbliche con nome, immutabili, esiti misurati. €0 ricorrenti.

## 1. Scopo

Non "chi è bravo". Non xG. Predice **mercato**: probabilità che un giocatore attualmente senza contratto (liste TM Serie C girone A/B/C) firmi entro 30 e 60 giorni.

Il prodotto vero è lo **scoreboard pubblico**: ogni lunedì top-10 con nome, ogni settimana esiti. Dopo N settimane: "X segnalati, Y firmati entro finestra". Prova verificabile, non opinione.

Chi lo usa: DS Serie C/D, agenti, chi assume Recruitment Analyst.

## 2. Vincoli Karpathy (costi nascosti = zero)

| Voce | Scelta | Escluso |
|---|---|---|
| Dipendenze | stdlib + `playwright` + `pandas` + `scikit-learn` | lightgbm, torch, LLM |
| Storage | CSV/JSON append-only nel repo | DB server, cloud storage |
| Compute | PC di Mirko, Task Scheduler lunedì 07:00 | Actions scraping, VPS, cron cloud |
| Sito | HTML statico generato da Python → GitHub Pages | React, framework, login |
| Modello | `HistGradientBoostingClassifier` | ensemble, tuning esteso |
| CI | pytest + build sito da CSV committati | scraping in CI |

Costo ricorrente: €0. Crediti build v1: 15-25 €.

## 3. Fonte dati (verificato 2026-09-13)

- Liste svincolati: `transfermarkt.it/serie-c-girone-{a,b,c}/vertragslosespieler/wettbewerb/IT3{A,B,C}` — 3 pagine × ~25 righe per girone. Campi: nome, ruolo, club precedente, nazionalità, presenze/gol/voci stagione precedente, valore di mercato.
- Storico trasferimenti: `transfermarkt.it/ceapi/transferHistory/list/{tm_id}` — JSON. Club speciale `svincolato` (id 515) con `dateUnformatted` e `marketValue` per riga.
- Profilo: `svincolato dal: DD/MM/YYYY`, data di nascita, posizione.

Stato accesso: `curl`/`urllib` → HTTP 202 vuoto (anti-bot). Playwright headless Chromium locale → OK. Wayback: nessuno snapshot. Rate limit autoimposto 1.5 s/richiesta, cache locale, refetch storico max 1/mese per giocatore.

## 4. Modello dati (CSV in repo)

```
data/snapshots/YYYY-MM-DD.csv   una riga per (data, tm_id) presente in lista
    tm_id, name, girone, role, prev_club, nationality, apps_prev, goals_prev,
    mv_eur, birth_date, free_since
data/history/{tm_id}.json       raw ceapi, cache
data/spells.csv                 derivato da history (funzione pura)
    tm_id, spell_start, spell_end|null, next_club, days|null,
    age_at_start, mv_at_start, role, n_prior_spells
data/outcomes.csv               prediction_date, tm_id, signed_date, club, days
predictions/YYYY-MM-DD.csv      tm_id, name, p30, p60, rank, model_version
predictions/YYYY-MM-DD.sha256   hash del CSV; mai riscritto
models/model_{version}.pkl + models/eval_{version}.md
docs/index.html                 sito (GitHub Pages)
```

Regola: `predictions/` e `outcomes.csv` sono append-only. Modifica retroattiva = bug.

## 5. Pipeline (`python -m svincolati <cmd>`)

| Cmd | Fa | Verifica |
|---|---|---|
| `harvest` | Playwright → snapshot CSV, 3 gironi | ≥50 righe per girone, colonne complete |
| `history` | ceapi JSON per tm_id nuovi o stale >30gg | file per ogni tm_id nello snapshot |
| `label` | history → `spells.csv` | test unitari su fixture |
| `train` | spells → modello + `eval_{v}.md` (split temporale) | AUC/Brier vs baseline nel report |
| `predict` | snapshot odierno → `predictions/YYYY-MM-DD.csv` + sha256 | hash riproducibile |
| `score` | snapshot + history → `outcomes.csv`, metriche scoreboard | hit-rate@30/@60, calibrazione |
| `site` | CSV → `docs/index.html` | apre offline, nessuna richiesta esterna |
| `weekly` | harvest → history → label → score → predict → site | un comando, Task Scheduler |

`train` è manuale, non nel `weekly`. Riallenamento quando `spells.csv` cresce ≥20%.

## 6. Label

- Spell = da trasferimento `→ svincolato` (data D0) a successivo trasferimento `svincolato → club` (data D1). `days = D1 - D0`.
- `label_30 = days ≤ 30`, `label_60 = days ≤ 60`.
- Spell aperto a data corrente: escluso se `oggi - D0 < 60`, altrimenti negativo per entrambi.
- Trasferimento con `to.clubName` in {"Carriera terminata", "Ritirato"}: spell chiuso senza firma → negativo.
- Predizione settimanale su spell aperto: `days_free_at_prediction = oggi - D0` è feature; l'esito si misura da data predizione, non da D0.

Documento pubblico `LABELING.md` spiega scelte e casi limite.

## 7. Feature v1 (nessun dato posteriore alla data di predizione)

age, role (GK/DF/MF/FW), log(mv_eur), apps_prev, goals_prev, days_free_at_prediction, mese (finestra aperta/chiusa), nazionalità IT/UE/extra, girone, n_prior_spells, media durata spell precedenti.

Domain flags dalla KB Serie C: scarsità ruolo (GK, DC alta), fascia mv < 500k, U23.

## 8. Baseline obbligatoria

Score euristico: U23 +2, GK/DC +2, mv<500k +1, apps_prev>15 +1, spell già >120 gg −2. Il modello va in produzione solo se batte la baseline su AUC test. Il report mostra sempre entrambi.

## 9. Sito pubblico

Una pagina. Sezioni: (1) top-10 di questa settimana con nome, ruolo, ex club, p30, p60; (2) scoreboard cumulativo: segnalati, firmati @30, @60, dove; (3) archivio settimane con hash e commit SHA; (4) metodo in 10 righe + link `LABELING.md`. Nessun contatto personale, nessun giudizio qualitativo.

## 10. Test

pytest: parser lista (fixture HTML salvata), parser ceapi (fixture JSON), `label` (spell aperto, multipli, ritiro, date mancanti), guardia leakage (nessuna feature con data > prediction_date), hash deterministico, `site` senza URL esterni.

## 11. Rischi

| Rischio | Mitigazione |
|---|---|
| TM blocca headless | Playwright headful con profilo persistente; settimana saltata dichiarata sul sito |
| Cold-start sottile (~200 spell) | v1.1: storico anche per rose complete Serie C (~1600 giocatori) |
| PC spento lunedì | Task ripete martedì; gap visibile, non nascosto |
| ToS Transfermarkt | Solo dati pubblici, rate limit, cache, attribuzione, uso non commerciale del dataset raw |

## 12. Fuori scope v1

Prestiti in scadenza, mercati esteri, Serie B/D, LLM, notifiche, UI dinamica, login, valutazione qualità, multi-sport, API.

## 13. Layout repo (opensource, MIT)

```
svincolati/
  svincolati/   tm.py harvest.py history.py label.py features.py
                train.py predict.py score.py site.py cli.py
  data/  predictions/  models/  docs/  tests/
  README.md  LABELING.md  pyproject.toml  LICENSE
```

Parser `tm.py` portato da `05_Strategy/ssv_clean/tm.py` (copia, non import). Posizione iniziale: `01_Products/svincolati/`; repo GitHub separato al primo ciclo pubblicato.
