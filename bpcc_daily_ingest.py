#!/usr/bin/env python3
"""Ingest BPCC-H-Daily Hindi TSV (src_lang,tgt_lang,src,tgt) -> jsonl chat pairs."""
import csv
import json
import re
import sys

csv.field_size_limit(10_000_000)
_JUNK = re.compile(r"[\[\]<>_{}|=*~`]")


def dev_ratio(s):
    L = [c for c in s if c.isalpha()]
    return sum(1 for c in L if "ऀ" <= c <= "ॿ") / max(len(L), 1)


def is_english(s):
    return dev_ratio(s) < 0.10 and any("a" <= c.lower() <= "z" for c in s)


def is_hindi(s):
    return dev_ratio(s) > 0.60


def main():
    inp, out = sys.argv[1], sys.argv[2]
    seen = set()
    kept = scanned = 0
    with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as o:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            scanned += 1
            en = (row.get("src") or "").strip()
            hi = (row.get("tgt") or "").strip()
            if not en or not hi or en == hi:
                continue
            if _JUNK.search(en) or _JUNK.search(hi):
                continue
            we, wh = len(en.split()), len(hi.split())
            if not (1 <= we <= 25) or not (1 <= wh <= 25):
                continue
            ratio = len(hi) / max(len(en), 1)
            if ratio < 0.3 or ratio > 3.0:
                continue
            if not is_english(en) or not is_hindi(hi):
                continue
            k = (en.lower(), hi.lower())
            if k in seen:
                continue
            seen.add(k)
            o.write(json.dumps({"translation": {"en": en, "hi": hi},
                                "source": "bpcc-daily", "tier": "chat"},
                               ensure_ascii=False) + "\n")
            kept += 1
    print(f"[done] scanned {scanned}  kept {kept} -> {out}")


if __name__ == "__main__":
    main()
