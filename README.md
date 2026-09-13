# svincolati

Predice **quando** uno svincolato di Serie C firma. Non quanto è bravo.

Ogni lunedì pubblica una top-10 con nome e cognome, la congela con un hash, e nelle settimane
successive scrive cosa è successo davvero. Il prodotto è lo scoreboard: predizioni contro esiti,
verificabili da chiunque.

Sito: `docs/index.html` (GitHub Pages).

## Perché

Chi recluta in Serie C e D scopre tardi. Quando un nome è su tutti i database, l'asimmetria è finita.
Questo strumento misura una cosa sola: il mercato si muove su questo giocatore nei prossimi 30 giorni?

## Pipeline

| Comando | Cosa fa | Output |
|---|---|---|
| `harvest` | Liste svincolati Serie C A/B/C da Transfermarkt (Playwright) | `data/snapshots/DATE.csv` |
| `history` | Profilo (una volta) + storico trasferimenti (refresh 30 gg) | `data/players.csv`, `data/history/ID.json` |
| `label` | Storico → periodi senza contratto ([LABELING.md](LABELING.md)) | `data/spells.csv` |
| `train` | Gradient boosting vs baseline euristica, split temporale | `models/` |
| `predict` | Snapshot → p30/p60 per giocatore, hash SHA-256 | `predictions/DATE.csv` + `.sha256` |
| `score` | Esiti delle predizioni passate, append-only | `data/outcomes.csv` |
| `site` | HTML statico | `docs/index.html` |
| `weekly` | harvest → history → label → score → predict → site | tutto |

`train` è manuale. Si rilancia quando `spells.csv` cresce.

## Regole non negoziabili

- `predictions/` e `data/outcomes.csv` sono **append-only**. Modifica retroattiva = bug.
- Il modello va in produzione solo se batte la baseline (AUC + 0.02) su split temporale. Altrimenti il sito usa la baseline e lo dice.
- Nessun LLM, nessun database, nessun cloud, nessun framework JS. Costo ricorrente: 0 €.
- Nessuna feature datata dopo la data di predizione (asserito nel codice).

## Setup

```bash
pip install playwright pandas scikit-learn pytest
playwright install chromium
python -m pytest
python -m svincolati weekly
```

Transfermarkt blocca i client HTTP semplici. Serve un browser reale: per questo Playwright, e per questo
l'harvest gira su un PC, non in CI. Rate limit 1.5 s per richiesta. Se il headless viene bloccato: `--headful`.

Windows Task Scheduler, lunedì 07:00:

```
schtasks /Create /SC WEEKLY /D MON /ST 07:00 /TN svincolati_weekly /TR "cmd /c cd /d D:\AI\02_Specialized\svincolati && python -m svincolati weekly >> logs\weekly.log 2>&1"
```

Poi `git add -A && git commit && git push`: GitHub Pages serve `docs/`.

## Fuori scope (v1)

Prestiti in scadenza, mercati esteri, Serie B/D, notifiche, UI dinamica, login, valutazione qualità, API.

## Dati

Dati pubblici Transfermarkt, solo nomi di calciatori professionisti già pubblici, nessun contatto personale.
Cache locale, refetch minimo. Dataset raw non ridistribuito. Uso non commerciale dei dati grezzi.

## Licenza

MIT. Autore: Mirko Tornani, UEFA B, staff FSGC.
