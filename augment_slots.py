#!/usr/bin/env python3
"""
augment_slots.py
Bilingual slot-substitution augmentation for EN-HI conversational pairs.

Idea (user's): take a real pair and make variants by swapping interchangeable
words on BOTH sides, e.g.
  EN: Let's go to the market today.   HI: आज बाज़ार चलते हैं।
  ->  Let's go to the office tomorrow. HI: कल दफ़्तर चलते हैं।

Correctness rules (to avoid breaking Hindi grammar):
  * Only swap grammatically-INERT classes: time adverbs, day names, discourse
    markers, and place-objects (masculine, take "the"). None trigger
    gender/number agreement on the verb.
  * A swap fires only when BOTH the English term AND its paired Hindi term are
    present -> we know the alignment, so we can swap both consistently.
  * Single-slot swap per variant (most natural), capped per pair.
  * Output is meant to then pass through labse_filter.py as a safety net:
      python labse_filter.py --in aug/pairs.jsonl --out aug_clean --threshold 0.70

Run:
  python augment_slots.py --in conv_clean/pairs.jsonl --out aug --per-pair 3
"""
import argparse
import json
import re
from pathlib import Path

# Each group: list of (english, hindi) members that are mutually swappable.
GROUPS = {
    "time": [
        ("today", "आज"), ("tomorrow", "कल"), ("now", "अभी"),
        ("later", "बाद में"), ("right now", "अभी"),
    ],
    "day": [
        ("Monday", "सोमवार"), ("Tuesday", "मंगलवार"), ("Wednesday", "बुधवार"),
        ("Thursday", "गुरुवार"), ("Friday", "शुक्रवार"), ("Saturday", "शनिवार"),
        ("Sunday", "रविवार"),
    ],
    "discourse": [
        ("yes", "हाँ"), ("okay", "ठीक है"), ("thanks", "शुक्रिया"),
        ("please", "कृपया"), ("of course", "बिल्कुल"),
    ],
    "place": [  # masculine, take "the"; appear as objects/locations
        ("the market", "बाज़ार"), ("the office", "दफ़्तर"),
        ("the school", "स्कूल"), ("the hospital", "अस्पताल"),
        ("the station", "स्टेशन"), ("the bank", "बैंक"),
    ],
}

# Devanagari range, for whole-token matching on the Hindi side
_DEV = "ऀ-ॿ"

# Guards to avoid breaking collocations / predicate uses.
GUARDS = {
    # don't swap a time adverb that's part of a time-of-day collocation
    # ("tomorrow morning") or modified by this/that/every/the/right.
    "time": {
        "next_block": {"morning", "afternoon", "evening", "night", "noon", "morn"},
        "prev_block": {"this", "that", "every", "the", "right"},
    },
    # only swap discourse markers when sentence-initial (filler use),
    # never as a predicate ("everything is okay").
    "discourse": {"sentence_initial": True},
}

_WORD = re.compile(r"[A-Za-z']+")


def _neighbors(text, start, end):
    """lowercased word immediately before / after the matched span."""
    before = text[:start]
    after = text[end:]
    pw = _WORD.findall(before)
    nw = _WORD.findall(after)
    return (pw[-1].lower() if pw else ""), (nw[0].lower() if nw else "")


def parse_args():
    p = argparse.ArgumentParser(description="Bilingual slot augmentation")
    p.add_argument("--in", dest="inp", default="conv_clean/pairs.jsonl")
    p.add_argument("--out", default="aug")
    p.add_argument("--per-pair", type=int, default=3, help="max variants per pair")
    return p.parse_args()


def en_pattern(term):
    # word-boundary, case-insensitive
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def hi_pattern(term):
    # bounded by non-Devanagari (so we don't match inside a larger word)
    return re.compile(r"(?<![" + _DEV + r"])" + re.escape(term) + r"(?![" + _DEV + r"])")


def match_case(original_match, replacement):
    """Preserve leading capitalization of the matched English token."""
    if original_match[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[info] loaded {len(rows):,} gold pairs")

    # precompile patterns
    compiled = {}
    for g, members in GROUPS.items():
        compiled[g] = [(en, hi, en_pattern(en), hi_pattern(hi)) for en, hi in members]

    gold_keys = set()
    for r in rows:
        t = r["translation"]
        gold_keys.add((t["en"].lower(), t["hi"].lower()))

    seen = set(gold_keys)
    out_rows = []
    per_group = {g: 0 for g in GROUPS}

    for r in rows:
        t = r["translation"]
        en0, hi0 = t["en"], t["hi"]
        made = 0
        for g, members in compiled.items():
            if made >= args.per_pair:
                break
            for en, hi, ep, hp in members:
                if made >= args.per_pair:
                    break
                m = ep.search(en0)
                if not m or not hp.search(hi0):
                    continue
                # apply guards
                guard = GUARDS.get(g, {})
                if guard.get("sentence_initial") and m.start() != 0:
                    continue
                pw, nw = _neighbors(en0, m.start(), m.end())
                if nw in guard.get("next_block", ()):
                    continue
                if pw in guard.get("prev_block", ()):
                    continue
                # both sides present -> swap to each alternative member
                for en2, hi2, _, _ in members:
                    if en2 == en or hi2 == hi:
                        continue
                    new_en = ep.sub(lambda mm: match_case(mm.group(0), en2), en0, count=1)
                    new_hi = hp.sub(hi2, hi0, count=1)
                    if new_en == en0 or new_hi == hi0:
                        continue
                    key = (new_en.lower(), new_hi.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    out_rows.append({
                        "translation": {"en": new_en, "hi": new_hi},
                        "source": "aug-slot", "tier": r.get("tier", "chat"),
                        "aug_group": g, "from": {"en": en0, "hi": hi0},
                    })
                    per_group[g] += 1
                    made += 1
                    if made >= args.per_pair:
                        break

    print("\n----- augmentation yield -----")
    for g, n in sorted(per_group.items(), key=lambda x: -x[1]):
        print(f"  {g:10s} {n:>9,}")
    print(f"  {'TOTAL':10s} {len(out_rows):>9,}")
    print(f"  gold {len(rows):,} + aug {len(out_rows):,} = {len(rows)+len(out_rows):,}")

    jsonl = out / "pairs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[info] wrote {jsonl}")

    print("\n----- sample augmented pairs -----")
    for r in out_rows[:10]:
        print(f"[{r['aug_group']}] EN: {r['translation']['en']}")
        print(f"          HI: {r['translation']['hi']}")
        print(f"     (from) EN: {r['from']['en']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
