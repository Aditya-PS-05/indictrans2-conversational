#!/usr/bin/env python3
"""
harvest_chat_dataset.py
Harvest REAL conversational English-Hindi parallel data, clean + filter to
dialogue-style lines, merge your own seed pairs, dedup, split, and save.

Philosophy: use REAL human translations first (no synthetic quality ceiling).
Only after seeing how much real data we get do we decide whether to augment
with back-translation (separate script).

Sources (real parallel):
  - OPUS OpenSubtitles  (en-hi)  -> movie/TV dialogue, genuinely conversational
  - OPUS-100            (en-hi)  -> ~534k pairs, broad, dialogue-heavy
  - your seed pairs     (--seed pairs.jsonl, {"translation":{"en","hi"}})

Conversational filter: short lines (1..--max-words), language-script check,
length-ratio, dedup. Subtitles are inherently conversational; the length cap
keeps OPUS-100 to its chat-like portion.

Run:
  python harvest_chat_dataset.py --out chat_real --seed my_pairs.jsonl
  python harvest_chat_dataset.py --out chat_real --max-words 25

Deps: pip install "datasets>=2.18" tqdm
"""
import argparse
import json
import re
from pathlib import Path

# UI / stage-direction / markup markers -> reject the pair outright
_JUNK = re.compile(r"[\[\]<>_{}|=*~`]")
_LEAD_DASH = re.compile(r"^[\-–—\s]+")


def parse_args():
    p = argparse.ArgumentParser(description="Harvest real conversational EN-HI data")
    p.add_argument("--out", default="chat_real")
    p.add_argument("--target", type=int, default=1_200_000,
                   help="stop harvesting once this many unique pairs collected")
    p.add_argument("--seed", default=None,
                   help="JSONL of your real pairs, {'translation':{'en','hi'}}")
    p.add_argument("--max-words", type=int, default=25,
                   help="max words per side (conversational = short)")
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=200)
    p.add_argument("--test-split", type=float, default=0.02)
    p.add_argument("--limit-per-source", type=int, default=0,
                   help="0 = no cap; else cap rows scanned per source (for testing)")
    return p.parse_args()


# --- language check by Unicode script (robust for EN<->HI, no model needed) ---
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
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    seen = set()          # dedup on (en_lower, hi_lower)
    pairs = []
    stats = {}

    def consider(en, hi, src):
        en, hi = (en or "").strip(), (hi or "").strip()
        # strip leading subtitle dialogue dashes ("- foo" -> "foo")
        en, hi = _LEAD_DASH.sub("", en).strip(), _LEAD_DASH.sub("", hi).strip()
        if not en or not hi or en == hi:
            return
        # reject UI strings / stage directions / markup
        if _JUNK.search(en) or _JUNK.search(hi):
            return
        if len(en) > args.max_chars or len(hi) > args.max_chars:
            return
        we, wh = len(en.split()), len(hi.split())
        if not (args.min_words <= we <= args.max_words):
            return
        if not (args.min_words <= wh <= args.max_words):
            return
        r = len(hi) / max(len(en), 1)
        if r < 0.4 or r > 2.5:
            return
        if not is_english(en) or not is_hindi(hi):
            return
        key = (en.lower(), hi.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append({"translation": {"en": en, "hi": hi}})
        stats[src] = stats.get(src, 0) + 1

    # ---- source loaders. Each wrapped so one failure doesn't kill the run ----
    def iter_translation_ds(name, config=None, **kw):
        ds = load_dataset(name, config, split="train", streaming=True, **kw)
        for row in ds:
            t = row.get("translation") or row
            yield t.get("en"), t.get("hi")

    def iter_samanantar():
        # Samanantar (hi): columns src=English, tgt=Hindi. ~8.5M, pre-filtered.
        ds = load_dataset("ai4bharat/samanantar", "hi", split="train", streaming=True)
        for row in ds:
            yield row.get("src"), row.get("tgt")

    # Order: cleanest/biggest real source first. OpenSubtitles dropped
    # (HF removed script-dataset support); add later via direct OPUS download.
    sources = [
        ("samanantar", iter_samanantar),
        ("opus-100",   lambda: iter_translation_ds("Helsinki-NLP/opus-100", "en-hi")),
    ]

    for src_name, src_iter in sources:
        if len(pairs) >= args.target:
            break
        print(f"\n[info] harvesting source: {src_name}")
        try:
            n = 0
            for en, hi in tqdm(src_iter(), desc=src_name):
                consider(en, hi, src_name)
                n += 1
                if len(pairs) >= args.target:
                    print(f"[info] reached target {args.target}; stopping")
                    break
                if args.limit_per_source and n >= args.limit_per_source:
                    break
        except Exception as e:
            print(f"[warn] source {src_name} failed/skipped: {e}")

    # ---- your seed pairs ----
    if args.seed:
        sp = Path(args.seed)
        if sp.exists():
            print(f"\n[info] merging seed: {sp}")
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        t = t.get("translation", t)
                        consider(t.get("en"), t.get("hi"), "seed")
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"[warn] seed file not found: {sp}")

    # ---- report ----
    print("\n----- harvest summary -----")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:15s} {v:>10,}")
    print(f"  {'TOTAL (unique)':15s} {len(pairs):>10,}")

    if not pairs:
        print("[error] no pairs harvested; check source availability / filters")
        return 1

    # ---- save ----
    jsonl = out / "pairs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[info] wrote {jsonl}")

    from datasets import Dataset
    ds = Dataset.from_list(pairs).train_test_split(test_size=args.test_split, seed=42)
    ds.save_to_disk(str(out / "hf"))
    print(f"[info] saved HF dataset -> {out/'hf'} "
          f"(train={len(ds['train'])}, test={len(ds['test'])})")

    print("\n----- spot check -----")
    for p in pairs[:6]:
        print("EN:", p["translation"]["en"])
        print("HI:", p["translation"]["hi"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
