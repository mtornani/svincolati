# LABELING — come nasce l'etichetta

Il modello non predice "quanto è bravo". Predice **se uno svincolato firma entro N giorni**.

## Spell

Uno *spell* è un periodo senza contratto ricavato dallo storico trasferimenti Transfermarkt
(`ceapi/transferHistory/list/{tm_id}`):

- **inizio** = data di un trasferimento con destinazione `svincolato`
- **fine** = data del trasferimento successivo con origine `svincolato`
- `days = fine − inizio`
- `next_club` = destinazione del trasferimento di chiusura
- `mv_at_start` = valore di mercato riportato sulla riga di apertura

Se non esiste trasferimento di chiusura, lo spell è **aperto** (`end = null`).

## Etichette

Per ogni spell si generano più esempi di training, uno ogni 14 giorni (`as_of` = inizio, inizio+14, ...),
finché `as_of < fine`. Così la distribuzione di `days_free` in training coincide con quella in predizione.

- `y30 = 1` se il giocatore firma entro 30 giorni da `as_of`
- `y60 = 1` se firma entro 60 giorni da `as_of`

## Casi limite

| Caso | Regola |
|---|---|
| Spell aperto, osservato da meno di 60 giorni | escluso dal training (censurato) |
| Spell aperto, osservato da 60+ giorni | negativo su entrambi gli orizzonti per gli `as_of` con 60 giorni di osservazione |
| Chiusura verso club speciale (es. `Carriera terminata`) | spell chiuso con `retired = true`, negativo |
| Trasferimento da `svincolato` senza spell aperto (storico troncato) | ignorato |
| Due aperture consecutive senza chiusura | conta la prima |
| Trasferimenti `upcoming` / futuri | ignorati |

## Esito di una predizione pubblicata

Si misura **dalla data di predizione**, non dall'inizio dello spell:

- `signed` se lo spell aperto a quella data si chiude dopo, con `days = firma − data predizione`
- `retired` se si chiude verso un club speciale
- `expired` se dopo 60 giorni è ancora aperto
- in attesa (nessuna riga) fino a una di queste condizioni

Le righe in `data/outcomes.csv` sono append-only.

## Limiti noti

- Le date Transfermarkt hanno granularità giornaliera e a volte coincidono con il 1 luglio per convenzione.
- Transfermarkt può correggere retroattivamente lo storico: lo spell può cambiare tra due settimane.
  Le predizioni già pubblicate non cambiano; gli esiti si calcolano sullo storico più recente.
- Storici mancanti o troncati producono meno esempi, non esempi sbagliati.
- La lista svincolati Transfermarkt copre chi ha giocato in Serie C la stagione precedente: non tutti gli svincolati d'Italia.
