#!/usr/bin/env python3
"""
build_tier2_general.py  (pure stdlib -- runs locally, no GPU/deps needed)
Filter + sample the large general-domain parallel corpus (archive(2),
line-aligned train.en / train.hi ~10M pairs = Samanantar/BPCC) down to a
clean target-size sample for the general (tier-2) bulk.

Surface filters only (Samanantar is already alignment-filtered by AI4Bharat):
length, word-count, ratio, junk chars, Unicode-script language check, dedup.
Probabilistic streaming sample -> ~--target pairs without holding 10M in RAM.

Run (locally):
  python3 build_tier2_general.py --en "archive(2)/train.en" --hi "archive(2)/train.hi" \
      --target 850000 --out tier2_general.jsonl
"""
import argparse
import json
import random
import re

_JUNK = re.compile(r"[\[\]<>_{}|=*~`]")
_WS = re.compile(r"\s+")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--en", required=True)
    p.add_argument("--hi", required=True)
    p.add_argument("--out", default="tier2_general.jsonl")
    p.add_argument("--target", type=int, default=850000)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--max-words", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=400)
    p.add_argument("--total-est", type=int, default=10_125_706,
                   help="line count, for setting sample probability")
    p.add_argument("--pass-rate", type=float, default=0.7,
                   help="estimated fraction passing filters")
    return p.parse_args()


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
    random.seed(42)
    # probability to accept a passing line so we land near target
    est_pass = max(1, int(a.total_est * a.pass_rate))
    p_keep = min(1.0, a.target / est_pass)
    print(f"[info] target={a.target:,}  p_keep={p_keep:.3f}")

    seen = set()
    kept = 0
    scanned = 0
    with open(a.en, encoding="utf-8") as ef, open(a.hi, encoding="utf-8") as hf, \
         open(a.out, "w", encoding="utf-8") as out:
        for en, hi in zip(ef, hf):
            scanned += 1
            if scanned % 1_000_000 == 0:
                print(f"  scanned {scanned:,}  kept {kept:,}")
            if kept >= a.target:
                break
            en = _WS.sub(" ", en).strip()
            hi = _WS.sub(" ", hi).strip()
            if not en or not hi or en == hi:
                continue
            if _JUNK.search(en) or _JUNK.search(hi):
                continue
            if len(en) > a.max_chars or len(hi) > a.max_chars:
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
            if random.random() > p_keep:
                continue
            key = (en.lower(), hi.lower())
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(
                {"translation": {"en": en, "hi": hi},
                 "source": "samanantar-iitb", "tier": "general"},
                ensure_ascii=False) + "\n")
            kept += 1

    print(f"[done] scanned {scanned:,}  kept {kept:,} -> {a.out}")


if __name__ == "__main__":
    main()
