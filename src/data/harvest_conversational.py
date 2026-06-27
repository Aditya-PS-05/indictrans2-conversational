#!/usr/bin/env python3
"""
harvest_conversational.py
Harvest + merge ALL real conversational English-Hindi parallel data, dedup
across sources, filter, tag by source, save.

Real conversational/spoken sources (no synthetic):
  OPUS moses downloads (line-aligned .en / .hi):
    - OpenSubtitles  movie/TV dialogue        [tier: chat]
    - Tatoeba        everyday short sentences  [tier: chat]
    - QED            educational subtitles     [tier: spoken]
    - TED2020        talks                     [tier: spoken]
  HuggingFace:
    - bajpaideeksha/english-hindi-colloquial-dataset  [tier: chat]
  Your own seed JSONL (--seed)                         [tier: chat]

Each output row: {"translation":{"en","hi"}, "source":..., "tier":"chat|spoken"}
so you can later weight/filter chat vs spoken.

Run:
  python harvest_conversational.py --out conv_real
  python harvest_conversational.py --out conv_real --tier chat   # chat-register only
  python harvest_conversational.py --out conv_real --seed mine.jsonl

Deps: pip install "datasets>=2.18" tqdm
"""
import argparse
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

_JUNK = re.compile(r"[\[\]<>_{}|=*~`]")
_LEAD_DASH = re.compile(r"^[\-–—\s]+")

OPUS = [
    # (source_name, tier, url)
    ("opensubtitles", "chat",
     "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-hi.txt.zip"),
    ("tatoeba", "chat",
     "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-hi.txt.zip"),
    # QED dropped: document-level aligned (whole lecture paragraphs), not
    # sentence-aligned, and it's math/educational lectures -- not conversational.
    ("ted2020", "spoken",
     "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-hi.txt.zip"),
]

HF = [
    # (source_name, tier, repo, config)
    ("colloquial-hf", "chat", "bajpaideeksha/english-hindi-colloquial-dataset", None),
]


def parse_args():
    p = argparse.ArgumentParser(description="Harvest real conversational EN-HI data")
    p.add_argument("--out", default="conv_real")
    p.add_argument("--cache", default="opus_cache")
    p.add_argument("--seed", default=None, help="your real pairs JSONL")
    p.add_argument("--tier", choices=["chat", "spoken", "all"], default="all",
                   help="chat = dialogue register only; all = + spoken (TED/QED)")
    p.add_argument("--max-words", type=int, default=30)
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=250)
    p.add_argument("--test-split", type=float, default=0.02)
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
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm

    seen = set()
    pairs = []
    stats = {}

    def want_tier(tier):
        return args.tier == "all" or args.tier == tier

    def consider(en, hi, src, tier):
        en, hi = (en or "").strip(), (hi or "").strip()
        en, hi = _LEAD_DASH.sub("", en).strip(), _LEAD_DASH.sub("", hi).strip()
        if not en or not hi or en == hi:
            return
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
        pairs.append({"translation": {"en": en, "hi": hi}, "source": src, "tier": tier})
        stats[src] = stats.get(src, 0) + 1

    # ---- OPUS moses sources ----
    for src, tier, url in OPUS:
        if not want_tier(tier):
            continue
        zpath = cache / f"{src}.zip"
        try:
            if not zpath.exists():
                print(f"[info] downloading {src} ...")
                urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                en_name = next(n for n in z.namelist() if n.endswith(".en"))
                hi_name = next(n for n in z.namelist() if n.endswith(".hi"))
                with z.open(en_name) as ef, z.open(hi_name) as hf:
                    ef = io.TextIOWrapper(ef, encoding="utf-8")
                    hf = io.TextIOWrapper(hf, encoding="utf-8")
                    for en, hi in tqdm(zip(ef, hf), desc=src):
                        consider(en, hi, src, tier)
        except Exception as e:
            print(f"[warn] OPUS {src} failed/skipped: {e}")

    # ---- HF datasets (auto-detect en/hi columns by script) ----
    def detect_fields(row):
        if isinstance(row.get("translation"), dict):
            return ("translation",)
        en_f = hi_f = None
        # 1) column-name hints (strip quotes/spaces)
        for k in row:
            if not isinstance(row[k], str):
                continue
            kl = k.lower().strip().strip('"').strip("'")
            if hi_f is None and ("hindi" in kl or kl in ("hi", "tgt", "target")):
                hi_f = k
            if en_f is None and ("english" in kl or kl in ("en", "src", "source")):
                en_f = k
        # 2) fall back to script detection on values
        if en_f is None or hi_f is None:
            for k, v in row.items():
                if not isinstance(v, str) or not v.strip():
                    continue
                if hi_f is None and is_hindi(v):
                    hi_f = k
                elif en_f is None and is_english(v):
                    en_f = k
        return (en_f, hi_f)

    for src, tier, repo, config in HF:
        if not want_tier(tier):
            continue
        try:
            from datasets import load_dataset
            ds = load_dataset(repo, config, split="train", streaming=True)
            it = iter(ds)
            first = next(it)
            fields = detect_fields(first)

            def emit(row):
                if len(fields) == 1:
                    t = row["translation"]
                    return t.get("en"), t.get("hi")
                return row.get(fields[0]), row.get(fields[1])

            if fields == (None, None) or (len(fields) == 2 and None in fields):
                print(f"[warn] HF {src}: could not detect en/hi columns "
                      f"(keys={list(first.keys())}); skipping")
                continue
            print(f"[info] HF {src}: fields={fields}")
            consider(*emit(first), src, tier)
            for row in tqdm(it, desc=src):
                consider(*emit(row), src, tier)
        except Exception as e:
            print(f"[warn] HF {src} failed/skipped: {e}")

    # ---- your seed ----
    if args.seed and Path(args.seed).exists():
        print(f"[info] merging seed {args.seed}")
        with open(args.seed, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    t = t.get("translation", t)
                    consider(t.get("en"), t.get("hi"), "seed", "chat")
                except json.JSONDecodeError:
                    continue

    # ---- report ----
    print("\n----- conversational harvest -----")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} {v:>10,}")
    print(f"  {'TOTAL (unique)':16s} {len(pairs):>10,}")
    chat = sum(1 for p in pairs if p['tier'] == 'chat')
    print(f"  {'  of which chat':16s} {chat:>10,}")

    if not pairs:
        print("[error] nothing harvested")
        return 1

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

    print("\n----- spot check (chat tier) -----")
    shown = 0
    for p in pairs:
        if p["tier"] != "chat":
            continue
        print(f"[{p['source']}] EN:", p["translation"]["en"])
        print(f"[{p['source']}] HI:", p["translation"]["hi"])
        print()
        shown += 1
        if shown >= 8:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
