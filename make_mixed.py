#!/usr/bin/env python3
"""Build mixed train/dev (conversational + general) to fight catastrophic forgetting."""
import json
import random

random.seed(42)


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


conv_train = load("conv_splits/train.jsonl")          # ~146k conversational
conv_dev = load("conv_splits/dev.jsonl")              # 1500 conversational
general = load("tier2_general.jsonl")                  # ~840k general
random.shuffle(general)

n_conv = len(conv_train)
gen_train = general[:n_conv]                           # 1:1 general:conv
gen_dev = general[n_conv:n_conv + 750]                 # disjoint general dev

mixed_train = conv_train + gen_train
random.shuffle(mixed_train)
mixed_dev = random.sample(conv_dev, 750) + gen_dev
random.shuffle(mixed_dev)

with open("conv_splits/mixed_train.jsonl", "w", encoding="utf-8") as f:
    for r in mixed_train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("conv_splits/mixed_dev.jsonl", "w", encoding="utf-8") as f:
    for r in mixed_dev:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"mixed_train={len(mixed_train):,} (conv {n_conv:,} + general {len(gen_train):,})")
print(f"mixed_dev={len(mixed_dev):,} (conv 750 + general {len(gen_dev)})")
