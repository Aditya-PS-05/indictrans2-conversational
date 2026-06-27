#!/usr/bin/env python3
"""Merge gold (conv_clean) + silver (aug_clean) into the final dataset."""
import json
from collections import Counter
from datasets import Dataset

INPUTS = ["conv_clean/pairs.jsonl", "aug_clean/pairs.jsonl"]
rows, seen = [], set()
for fn in INPUTS:
    for line in open(fn, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r["translation"]
        k = (t["en"].lower(), t["hi"].lower())
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)

c = Counter(r["source"] for r in rows)
print("FINAL dataset composition:")
for s, n in c.most_common():
    print("  %-14s %9d" % (s, n))
print("  %-14s %9d" % ("TOTAL", len(rows)))

ds = Dataset.from_list(rows).train_test_split(test_size=0.02, seed=42)
ds.save_to_disk("final/hf")
with open("final/pairs.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("saved final/  train=%d  test=%d" % (len(ds["train"]), len(ds["test"])))
