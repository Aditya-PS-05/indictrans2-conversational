#!/usr/bin/env python3
"""
build_en_hi_dataset.py
Build a large English->Hindi parallel dataset by SYNTHETIC translation.

Pipeline:
  1. Stream monolingual English text from a HF dataset.
  2. Segment into sentences, clean, length-filter.
  3. Translate EN->HI in batches with NLLB on GPU (falls back to CPU).
  4. Filter pairs: language ID, length-ratio, dedup, drop copies.
  5. Save JSONL ({"translation": {"en": ..., "hi": ...}}) + a HF dataset on disk.

Run on the A10G box. Start small to sanity-check, then scale --target.

  python build_en_hi_dataset.py --target 5000 --out data_smoke      # smoke test
  python build_en_hi_dataset.py --target 2000000 --out data_full    # real run

Deps:
  pip install "datasets>=2.18" transformers torch sentencepiece \
              nltk fasttext-wheel tqdm
  # fastText language-ID model (one-time, ~125MB):
  #   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
"""
import argparse
import json
import re
import sys
from pathlib import Path

# ----------------------------- config / cli ------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Build EN->HI parallel dataset")
    p.add_argument("--target", type=int, default=5000,
                   help="number of clean pairs to produce")
    p.add_argument("--out", default="en_hi_corpus", help="output dir")
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                   help="HF translation model (NLLB). 1.3B is better, slower.")
    p.add_argument("--src-dataset", default="wikimedia/wikipedia",
                   help="HF monolingual English source")
    p.add_argument("--src-config", default="20231101.en",
                   help="config/name for the source dataset")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=128, help="max tokens / sentence")
    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--max-words", type=int, default=40)
    p.add_argument("--test-split", type=float, default=0.05)
    return p.parse_args()


# ----------------------------- cleaning ----------------------------------- #

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

def clean_text(t: str) -> str:
    t = _HTML.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def ok_length(s: str, lo: int, hi: int) -> bool:
    n = len(s.split())
    return lo <= n <= hi


# ----------------------------- main --------------------------------------- #

def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # lazy imports so --help is fast and errors are clear
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from datasets import load_dataset
    from tqdm import tqdm

    # sentence segmentation (English)
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize

    # Language check by Unicode script (no external model needed).
    # For EN<->HI this is more robust than fastText: Hindi is Devanagari,
    # English is Latin, so a script-ratio test cleanly separates them.
    def devanagari_ratio(s: str) -> float:
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return 0.0
        dev = sum(1 for c in letters if "ऀ" <= c <= "ॿ")
        return dev / len(letters)

    def is_english(s: str) -> bool:
        # mostly Latin letters, almost no Devanagari
        return devanagari_ratio(s) < 0.10 and any("a" <= c.lower() <= "z" for c in s)

    def is_hindi(s: str) -> bool:
        # predominantly Devanagari
        return devanagari_ratio(s) > 0.60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}  model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    hi_id = tok.convert_tokens_to_ids("hin_Deva")

    @torch.inference_mode()
    def en2hi(batch):
        x = tok(batch, return_tensors="pt", padding=True,
                truncation=True, max_length=args.max_len).to(device)
        out_ids = model.generate(**x, forced_bos_token_id=hi_id,
                                 max_length=args.max_len, num_beams=2)
        return tok.batch_decode(out_ids, skip_special_tokens=True)

    print(f"[info] streaming {args.src_dataset}:{args.src_config}")
    src = load_dataset(args.src_dataset, args.src_config,
                       split="train", streaming=True)

    seen = set()
    pairs = []
    buf = []
    bar = tqdm(total=args.target, desc="pairs")

    def flush():
        """translate the buffered EN sentences and keep good pairs."""
        if not buf:
            return
        his = en2hi(buf)
        for en, hi in zip(buf, his):
            en, hi = en.strip(), hi.strip()
            if not hi or en == hi:
                continue
            # length-ratio filter
            r = len(hi) / max(len(en), 1)
            if r < 0.5 or r > 2.0:
                continue
            # language check (Unicode script)
            if not is_english(en) or not is_hindi(hi):
                continue
            pairs.append({"translation": {"en": en, "hi": hi}})
            bar.update(1)
        buf.clear()

    for row in src:
        if len(pairs) >= args.target:
            break
        doc = clean_text(row.get("text", "") or "")
        if not doc:
            continue
        for sent in sent_tokenize(doc):
            if len(pairs) >= args.target:
                break
            sent = sent.strip()
            if not ok_length(sent, args.min_words, args.max_words):
                continue
            key = sent.lower()
            if key in seen:               # exact dedup on EN side
                continue
            seen.add(key)
            buf.append(sent)
            if len(buf) >= args.batch_size:
                flush()
    flush()
    bar.close()

    print(f"[info] produced {len(pairs)} clean pairs")

    # write JSONL
    jsonl = out / "pairs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[info] wrote {jsonl}")

    # write HF dataset with train/test split
    from datasets import Dataset
    ds = Dataset.from_list(pairs)
    ds = ds.train_test_split(test_size=args.test_split, seed=42)
    ds.save_to_disk(str(out / "hf"))
    print(f"[info] saved HF dataset -> {out/'hf'}  "
          f"(train={len(ds['train'])}, test={len(ds['test'])})")

    # spot-check sample
    print("\n----- spot check (5 random-ish pairs) -----")
    for p in pairs[:5]:
        print("EN:", p["translation"]["en"])
        print("HI:", p["translation"]["hi"])
        print()


if __name__ == "__main__":
    sys.exit(main())
