#!/usr/bin/env python3
"""
build_tier1_iitb.py  (pure stdlib -- runs locally)
Mine the conversational/short slice from the IIT Bombay CSV
(archive/Dataset_English_Hindi.csv) for the tier-1 (chat) pool.

The IITB corpus is tokenized + noisy, so we:
  * detokenize: -LRB-/-RRB- -> ( ), strip spaces before punctuation
  * reject Hindi side containing Latin letters (encoding glitches / code-mix)
  * keep only SHORT lines (chat register), drop junk, dedup
Output is then LaBSE-filtered on the GPU box for final quality.

Run (locally):
  python3 build_tier1_iitb.py --in "archive/Dataset_English_Hindi.csv" \
      --out tier1_iitb.jsonl
"""
import argparse
import csv
import json
import re
import sys

csv.field_size_limit(10_000_000)

_JUNK = re.compile(r"[\[\]<>_{}|=*~`]")
_LATIN = re.compile(r"[A-Za-z]")
_SP_PUNCT = re.compile(r"\s+([.,!?;:।])")
_WS = re.compile(r"\s+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="archive/Dataset_English_Hindi.csv")
    p.add_argument("--out", default="tier1_iitb.jsonl")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=int, default=12)
    return p.parse_args()


def detok(s):
    s = s.replace("-LRB-", "(").replace("-RRB-", ")")
    s = s.replace("-LSB-", "").replace("-RSB-", "")
    s = _WS.sub(" ", s).strip()
    s = _SP_PUNCT.sub(r"\1", s)
    return s


def devanagari_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def is_english(s):
    return devanagari_ratio(s) < 0.10 and any("a" <= c.lower() <= "z" for c in s)


def is_hindi(s):
    return devanagari_ratio(s) > 0.60


def main():
    a = parse_args()
    seen = set()
    kept = scanned = 0
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        reader = csv.reader(f)
        header = next(reader, None)  # English,Hindi
        for row in reader:
            scanned += 1
            if len(row) < 2:
                continue
            en, hi = detok(row[0]), detok(row[1])
            if not en or not hi or en == hi:
                continue
            if _JUNK.search(en) or _JUNK.search(hi):
                continue
            if _LATIN.search(hi):              # encoding glitch / code-mix on HI side
                continue
            we, wh = len(en.split()), len(hi.split())
            if not (a.min_words <= we <= a.max_words):
                continue
            if not (a.min_words <= wh <= a.max_words):
                continue
            r = len(hi) / max(len(en), 1)
            if r < 0.4 or r > 2.5:
                continue
            if not is_english(en) or not is_hindi(hi):
                continue
            key = (en.lower(), hi.lower())
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(
                {"translation": {"en": en, "hi": hi},
                 "source": "iitb", "tier": "chat"}, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[done] scanned {scanned:,}  kept {kept:,} -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
