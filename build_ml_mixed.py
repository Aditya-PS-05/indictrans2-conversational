#!/usr/bin/env python3
"""
Build mixed multilingual train/dev = conversational (ml_train) + a per-language
general anchor sampled from BPCC-Wiki (~1:1 per lang), to fight forgetting
across all languages. Records: {"translation":{"en","tgt"}, "lang":"<code>"}.
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

csv.field_size_limit(10_000_000)
random.seed(42)
_WS = re.compile(r"\s+")


def clean(en, tgt):
    en, tgt = _WS.sub(" ", (en or "")).strip(), _WS.sub(" ", (tgt or "")).strip()
    if not en or not tgt or en == tgt:
        return None
    we = len(en.split())
    if we < 1 or we > 50 or len(en) > 400 or len(tgt) > 450:
        return None
    return en, tgt


# --- load conversational train, count per lang ---
conv = []
for line in open("conv_splits/ml_train.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        conv.append((r["translation"]["en"], r["translation"]["tgt"], r["lang"]))
conv_counts = Counter(t[2] for t in conv)
print("conv loaded:", len(conv))

# --- general anchor from BPCC-Wiki, sampled to match conv per-lang count ---
general = []
seen = set()
for lang, target_n in conv_counts.items():
    try:
        p = hf_hub_download("ai4bharat/BPCC", f"wiki/{lang}.tsv", repo_type="dataset")
    except Exception as e:
        print(f"[warn] wiki {lang}: {str(e)[:40]}")
        continue
    rows = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            c = clean(row.get("src"), row.get("tgt"))
            if not c:
                continue
            k = (c[0].lower(), c[1].lower(), lang)
            if k in seen:
                continue
            seen.add(k)
            rows.append((c[0], c[1], lang))
    random.shuffle(rows)
    take = rows[:target_n]
    general += take
    print(f"  {lang:10s} conv={target_n:>6d} general_taken={len(take):>6d}")

print(f"general total: {len(general):,}")

# --- combine + small mixed dev ---
mixed = conv + general
random.shuffle(mixed)

dev = []
for line in open("conv_splits/ml_dev.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        dev.append((r["translation"]["en"], r["translation"]["tgt"], r["lang"]))
# add ~ up to 40 general dev per lang
gen_by_lang = defaultdict(list)
for t in general:
    gen_by_lang[t[2]].append(t)
for lang, items in gen_by_lang.items():
    dev += items[:40]
random.shuffle(dev)


def dump(name, rows):
    with open(f"conv_splits/ml_mixed_{name}.jsonl", "w", encoding="utf-8") as f:
        for en, tgt, lang in rows:
            f.write(json.dumps({"translation": {"en": en, "tgt": tgt}, "lang": lang},
                               ensure_ascii=False) + "\n")


dump("train", mixed)
dump("dev", dev)
print(f"\nmixed_train={len(mixed):,} (conv {len(conv):,} + general {len(general):,})  "
      f"mixed_dev={len(dev):,}")
