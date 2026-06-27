# Human Evaluation — Instructions for Raters

You are rating English→Hindi (or English→Tamil) translations. Two systems
produced each translation; they are shown **blind** as **A** and **B** (you do
NOT know which is which, and the order is shuffled per row). Rate them fairly.

## Files
- `eval_sheet_hin.csv` — 60 Hindi sentences to rate
- `eval_sheet_tam.csv` — 60 Tamil sentences (needs a Tamil speaker)
- `eval_key_*.csv` — **the answer key. Do NOT open until rating is finished.**

## How to rate
Open the sheet in Excel / Google Sheets. For **each row**, read the `english`
source (the `reference` column is a human translation, for context only — it may
itself be imperfect), then score **both** `translation_A` and `translation_B` on
two 1–5 scales:

**Adequacy** — does the translation convey the meaning of the English?
- 5 = fully conveys the meaning
- 4 = mostly, minor info lost
- 3 = roughly, some meaning lost
- 2 = little of the meaning
- 1 = wrong / unrelated

**Fluency** — is it natural, grammatical Hindi/Tamil (ignore the English)?
- 5 = perfectly natural
- 4 = good, tiny awkwardness
- 3 = understandable but clunky
- 2 = broken grammar
- 1 = not real language

Fill the four columns: `A_adequacy_1to5`, `A_fluency_1to5`, `B_adequacy_1to5`,
`B_fluency_1to5`. Score A and B independently — ties are fine.

## Tips for fair rating
- Don't try to guess which system is which.
- Rate every row; don't skip.
- If two are equal, give equal scores.
- ~60 sentences takes roughly 45–60 minutes.

## After rating
Save the filled CSV (keep the same filename) and run:
```
python3 analyze_eval.py eval_sheet_hin.csv eval_key_hin.csv
```
It un-blinds (joins the key), reports base vs. soup mean adequacy/fluency, and a
paired bootstrap p-value. Two or more independent raters per language is ideal
(lets you report inter-annotator agreement); one is acceptable for a first draft.
