#!/usr/bin/env python3
"""Merge all tiers into the final two-tier dataset (all real, no synthetic)."""
import json
from collections import Counter
from datasets import Dataset

INPUTS = [
    "conv_clean/pairs.jsonl",        # OpenSubtitles/TED/Tatoeba (chat+spoken)
    "tier1_iitb_clean/pairs.jsonl",  # IITB chat slice (LaBSE-filtered)
    "bpcc_daily_clean/pairs.jsonl",  # BPCC-H-Daily everyday/assistant (LaBSE-filtered)
    "aug_clean/pairs.jsonl",         # slot augmentation
    "tier2_general.jsonl",           # Samanantar/IITB general bulk
]

rows, seen = [], set()
for fn in INPUTS:
    n0 = len(rows)
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
    print("  %-26s +%d" % (fn, len(rows) - n0))

print("\nFINAL composition by tier:")
for tier, n in Counter(r.get("tier", "?") for r in rows).most_common():
    print("  %-10s %9d" % (tier, n))
print("by source:")
for s, n in Counter(r.get("source", "?") for r in rows).most_common():
    print("  %-16s %9d" % (s, n))
print("  %-16s %9d" % ("TOTAL", len(rows)))

ds = Dataset.from_list(rows).train_test_split(test_size=0.01, seed=42)
ds.save_to_disk("final_v2/hf")
with open("final_v2/pairs.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\nsaved final_v2/  train=%d  test=%d" % (len(ds["train"]), len(ds["test"])))
