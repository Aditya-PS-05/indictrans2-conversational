#!/usr/bin/env python3
"""Select Hindi conv pairs where base & soup differ MEANINGFULLY (not degenerate),
on substantive sentences -> pairwise-preference sheet (A/Tie/B) + hidden key."""
import csv
import difflib
import json
import random
import re

random.seed(11)
norm = lambda s: re.sub(r"[।.,!?\"'\-\s]", "", s)

conv = []
for line in open("conv_splits/ml_test.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        if r["lang"] == "hin_Deva":
            conv.append(r["translation"]["en"])

base = open("sig/hin_Deva_conv.base", encoding="utf-8").read().split("\n")
soup = open("sig/hin_Deva_conv.soup", encoding="utf-8").read().split("\n")
n = min(len(conv), len(base), len(soup))

cand = []
for i in range(n):
    en = conv[i].strip()
    b, s = base[i].strip(), soup[i].strip()
    if not b or not s:
        continue
    if len(en.split()) < 4:                       # skip trivial one/two-word lines
        continue
    if norm(b) == norm(s):                        # skip identical (ignoring punct)
        continue
    lr = len(b) / max(len(s), 1)
    if lr < 0.6 or lr > 1.7:                       # skip degenerate length blowups (repetition loops)
        continue
    div = 1 - difflib.SequenceMatcher(None, b, s).ratio()
    if not (0.18 <= div <= 0.65):                 # meaningful, but not pathological
        continue
    cand.append((div, i, en, b, s))

random.shuffle(cand)                              # representative sample, not just extremes
sel = cand[:30]

rows, key = [], []
for nid, (_, i, en, b, s) in enumerate(sel, 1):
    pair = [("base", b), ("soup", s)]
    random.shuffle(pair)
    (a_sys, a_txt), (b_sys, b_txt) = pair
    rows.append([nid, en, a_txt, b_txt, ""])
    key.append([nid, a_sys, b_sys])

with open("eval_data/eval_pref_hin.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "english", "translation_A", "translation_B", "which_is_better"])
    w.writerows(rows)
with open("eval_data/eval_pref_key_hin.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "A_system", "B_system"])
    w.writerows(key)

print(f"clean meaningful-divergence pool: {len(cand)} / {n}; selected {len(sel)}")
