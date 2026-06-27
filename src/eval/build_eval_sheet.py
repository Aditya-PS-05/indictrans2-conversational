#!/usr/bin/env python3
"""
build_eval_sheet.py  (pure stdlib — no GPU/torch)
Build BLIND human-evaluation sheets for base vs mixed-soup on conversational
Hindi & Tamil. Uses the translations already saved by sig_test.py (sig/*.base,
sig/*.soup) re-aligned with their English sources from conv_splits/ml_test.jsonl.

Outputs per language:
  eval_sheet_<lg>.csv  -> for raters (A/B shuffled, blind), empty score columns
  eval_key_<lg>.csv    -> hidden key mapping A/B -> base/soup
"""
import csv
import json
import random

random.seed(42)
LANGS = {"hin_Deva": "hin", "tam_Taml": "tam"}
N_SAMPLE = 60

# load conversational test (english + reference), grouped by lang
conv = {}
for line in open("conv_splits/ml_test.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        conv.setdefault(r["lang"], []).append((r["translation"]["en"], r["translation"]["tgt"]))


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().split("\n")


for lang, short in LANGS.items():
    base = read_lines(f"sig/{lang}_conv.base")
    soup = read_lines(f"sig/{lang}_conv.soup")
    items = conv.get(lang, [])
    n = min(len(items), len(base), len(soup))
    en = [items[i][0] for i in range(n)]
    ref = [items[i][1] for i in range(n)]
    base, soup = base[:n], soup[:n]

    idxs = list(range(n))
    random.shuffle(idxs)
    idxs = idxs[:min(N_SAMPLE, n)]

    sheet_rows, key_rows = [], []
    for new_id, i in enumerate(idxs, 1):
        pair = [("base", base[i]), ("soup", soup[i])]
        random.shuffle(pair)                       # A/B blind assignment
        (a_sys, a_txt), (b_sys, b_txt) = pair
        sheet_rows.append([new_id, en[i], ref[i], a_txt, b_txt, "", "", "", ""])
        key_rows.append([new_id, a_sys, b_sys])

    with open(f"eval_data/eval_sheet_{short}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "english", "reference",
                    "translation_A", "translation_B",
                    "A_adequacy_1to5", "A_fluency_1to5",
                    "B_adequacy_1to5", "B_fluency_1to5"])
        w.writerows(sheet_rows)
    with open(f"eval_data/eval_key_{short}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "A_system", "B_system"])
        w.writerows(key_rows)
    print(f"{lang}: {len(idxs)} sentences -> eval_sheet_{short}.csv (+ key)")

print("DONE")
