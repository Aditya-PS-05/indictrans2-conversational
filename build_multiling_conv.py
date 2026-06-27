#!/usr/bin/env python3
"""
Build a multilingual conversational EN->Indic dataset:
  * BPCC-H-Daily: all 21 available langs (everyday/assistant)
  * OpenSubtitles: hi/ml/ta/te/ur (dialogue), capped so they don't dominate
Each record: {"translation":{"en","tgt"}, "lang":"<flores_code>"}.
Splits per-language into train/dev/test (holds out eval).
"""
import csv
import io
import json
import random
import re
import zipfile
import urllib.request
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download

csv.field_size_limit(10_000_000)
random.seed(42)
_WS = re.compile(r"\s+")

DAILY_LANGS = ["asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva",
               "guj_Gujr", "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva",
               "mal_Mlym", "mar_Deva", "mni_Mtei", "npi_Deva", "ory_Orya",
               "pan_Guru", "san_Deva", "sat_Olck", "tam_Taml", "tel_Telu",
               "urd_Arab"]
# OpenSubtitles 2-letter -> flores code, with per-lang cap
OPENSUBS = {"hi": ("hin_Deva", 40000), "ml": ("mal_Mlym", 40000),
            "ta": ("tam_Taml", 40000), "te": ("tel_Telu", 40000),
            "ur": ("urd_Arab", 40000)}

pairs = []          # (en, tgt, lang)
seen = set()


def add(en, tgt, lang):
    en, tgt = _WS.sub(" ", (en or "")).strip(), _WS.sub(" ", (tgt or "")).strip()
    if not en or not tgt or en == tgt:
        return
    we = len(en.split())
    if we < 1 or we > 40 or len(en) > 300 or len(tgt) > 350:
        return
    r = len(tgt) / max(len(en), 1)
    if r < 0.3 or r > 3.5:
        return
    k = (en.lower(), tgt.lower(), lang)
    if k in seen:
        return
    seen.add(k)
    pairs.append((en, tgt, lang))


# --- BPCC-H-Daily ---
for lg in DAILY_LANGS:
    try:
        p = hf_hub_download("ai4bharat/BPCC", f"daily/{lg}.tsv", repo_type="dataset")
        with open(p, encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            for row in r:
                add(row.get("src"), row.get("tgt"), lg)
    except Exception as e:
        print(f"[warn] daily {lg}: {str(e)[:60]}")

# --- OpenSubtitles (capped) ---
for code, (lang, cap) in OPENSUBS.items():
    try:
        url = f"https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-{code}.txt.zip"
        zpath, _ = urllib.request.urlretrieve(url)
        with zipfile.ZipFile(zpath) as z:
            en_name = next(n for n in z.namelist() if n.endswith(".en"))
            tg_name = next(n for n in z.namelist() if n.endswith(f".{code}"))
            with z.open(en_name) as ef, z.open(tg_name) as tf:
                ef = io.TextIOWrapper(ef, encoding="utf-8")
                tf = io.TextIOWrapper(tf, encoding="utf-8")
                n0 = len(pairs)
                for en, tg in zip(ef, tf):
                    if len(pairs) - n0 >= cap:
                        break
                    add(en, tg, lang)
    except Exception as e:
        print(f"[warn] opensubs {code}: {str(e)[:60]}")

# --- per-language split ---
by_lang = defaultdict(list)
for t in pairs:
    by_lang[t[2]].append(t)

train, dev, test = [], [], []
for lang, items in by_lang.items():
    random.shuffle(items)
    n = len(items)
    n_test = min(300, n // 10)
    n_dev = min(200, n // 10)
    test += items[:n_test]
    dev += items[n_test:n_test + n_dev]
    train += items[n_test + n_dev:]

random.shuffle(train)


def dump(name, rows):
    with open(f"conv_splits/ml_{name}.jsonl", "w", encoding="utf-8") as f:
        for en, tgt, lang in rows:
            f.write(json.dumps({"translation": {"en": en, "tgt": tgt}, "lang": lang},
                               ensure_ascii=False) + "\n")


import os
os.makedirs("conv_splits", exist_ok=True)
dump("train", train); dump("dev", dev); dump("test", test)

print(f"TOTAL pairs={len(pairs):,}  train={len(train):,} dev={len(dev):,} test={len(test):,}")
print("per-language counts:")
for lang, c in sorted(Counter(t[2] for t in pairs).items(), key=lambda x: -x[1]):
    print(f"  {lang:10s} {c:>7d}")
