#!/usr/bin/env python3
"""Carve the conversational subset (chat+spoken) into train/dev/test, no leakage."""
import json
import random

random.seed(42)
rows, seen = [], set()
for line in open("final_v2/pairs.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r.get("tier") not in ("chat", "spoken"):
        continue
    t = r["translation"]
    k = (t["en"].lower(), t["hi"].lower())
    if k in seen:
        continue
    seen.add(k)
    rows.append(r)

random.shuffle(rows)
n = len(rows)
n_test = min(1500, n // 50)
n_dev = min(1500, n // 50)
test, dev, train = rows[:n_test], rows[n_test:n_test + n_dev], rows[n_test + n_dev:]

import os
os.makedirs("conv_splits", exist_ok=True)
for name, part in [("train", train), ("dev", dev), ("test", test)]:
    with open(f"conv_splits/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"conversational total={n}  train={len(train)}  dev={len(dev)}  test={len(test)}")
