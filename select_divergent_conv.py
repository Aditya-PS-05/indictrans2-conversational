#!/usr/bin/env python3
"""Select base-vs-soup divergent Hindi pairs BIASED TOWARD CONVERSATIONAL source
sentences (everyday/casual/chatty English), domain-matching the conversational
claim. Selection scores the ENGLISH SOURCE only — it is blind to which model
(base/soup) produced which output, so it cannot rig the win. Same divergence
filters as select_divergent.py; then rank by a conversationality score.

Reads:  conv_splits/ml_test.jsonl, sig/hin_Deva_conv.{base,soup}
Writes: eval_pref_hin.csv, eval_pref_key_hin.csv   (old ones backed up *_mixed.csv)
"""
import csv
import difflib
import json
import os
import re
import shutil

norm = lambda s: re.sub(r"[।.,!?\"'\-\s]", "", s)

# --- conversationality scoring on the ENGLISH source (model-blind) -------------
CASUAL = {"hey", "yeah", "yep", "nope", "ok", "okay", "well", "oh", "hi", "hello",
          "sorry", "thanks", "please", "wow", "hmm", "huh", "gonna", "wanna",
          "gotta", "c'mon", "guys", "dude", "man", "right", "sure", "fine",
          "look", "listen", "come", "let's", "lets", "stop", "wait", "really"}
PRON = {"i", "you", "we", "me", "us", "my", "your", "our", "mine", "yours",
        "i'm", "you're", "we're", "i'll", "you'll", "i've", "you've",
        "i'd", "don't", "can't", "won't", "didn't", "isn't", "it's", "that's",
        "he's", "she's", "they're", "what's", "let's"}
FORMAL = {"henceforth", "thereof", "wherein", "shall", "hereby", "whilst",
          "moreover", "furthermore", "thus", "therefore", "regarding",
          "pursuant", "notwithstanding", "aforementioned", "whereby",
          "manpower", "region", "territory", "province", "empire", "nomads",
          "bandits", "macedonia", "kingdom", "majesty", "decree"}


def conv_score(en):
    toks = re.findall(r"[a-zA-Z']+", en.lower())
    if not toks:
        return -99
    n = len(toks)
    s = 0.0
    s += sum(1.5 for t in toks if t in PRON)          # personal pronouns / contractions
    s += sum(2.0 for t in toks if t in CASUAL)         # casual discourse markers
    s += 1.5 * en.count("?") + 1.0 * en.count("!")     # questions / exclamations
    s += 1.0 if "'" in en else 0                       # contractions present
    s += sum(-2.0 for t in toks if t in FORMAL)        # literary/formal vocab
    # proper-noun / named-entity penalty (capitalised words mid-sentence -> narrative)
    caps = len(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+", en))
    s -= 1.0 * caps
    s -= 0.5 if ";" in en else 0                       # semicolons -> formal
    # length: short = conversational; long = narrative. peak around <=10 words
    s += 2.0 if n <= 10 else (0.0 if n <= 16 else -2.0)
    return s / (1 + 0.05 * n)                          # mild length-normalise


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
    if len(en.split()) < 3:                            # allow short casual lines
        continue
    if norm(b) == norm(s):                             # skip identical
        continue
    lr = len(b) / max(len(s), 1)
    if lr < 0.6 or lr > 1.7:                            # skip degenerate length blowups
        continue
    div = 1 - difflib.SequenceMatcher(None, b, s).ratio()
    if not (0.18 <= div <= 0.65):                       # meaningful, not pathological
        continue
    cand.append((conv_score(en), div, i, en, b, s))

# rank by conversationality (descending); take the most conversational 30
cand.sort(key=lambda x: x[0], reverse=True)
sel = cand[:30]

# stable, reproducible A/B assignment (no RNG: id-parity decides order)
rows, key = [], []
for nid, (cs, _, i, en, b, s) in enumerate(sel, 1):
    if nid % 2 == 0:
        (a_sys, a_txt), (b_sys, b_txt) = ("base", b), ("soup", s)
    else:
        (a_sys, a_txt), (b_sys, b_txt) = ("soup", s), ("base", b)
    rows.append([nid, en, a_txt, b_txt, ""])
    key.append([nid, a_sys, b_sys])

for f in ("eval_pref_hin.csv", "eval_pref_key_hin.csv"):
    if os.path.exists(f):
        shutil.copy(f, f.replace(".csv", "_mixed.csv"))

with open("eval_pref_hin.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "english", "translation_A", "translation_B", "which_is_better"])
    w.writerows(rows)
with open("eval_pref_key_hin.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "A_system", "B_system"])
    w.writerows(key)

print(f"divergent pool: {len(cand)}/{n}; selected 30 most conversational")
print(f"conv-score range of selection: {sel[0][0]:.2f} (top) .. {sel[-1][0]:.2f} (cut)")
print(f"   vs pool median ~{sorted(c[0] for c in cand)[len(cand)//2]:.2f}\n")
print("selected English sources (most-conversational first):")
for nid, (cs, _, i, en, b, s) in enumerate(sel, 1):
    print(f"  {nid:>2} [{cs:+.2f}] {en[:70]}")
print("\nbacked up old eval -> *_mixed.csv; wrote conversational eval_pref_hin.csv + key")
