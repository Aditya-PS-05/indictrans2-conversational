#!/usr/bin/env python3
"""
labse_filter.py
Semantic-alignment filter. Embeds EN and HI sides with LaBSE (multilingual,
shared space) and drops pairs whose cosine similarity is low -- i.e. the two
sides don't mean the same thing (subtitle drift, misalignment).

This catches errors the surface filters can't, e.g.
  EN: I will never salute you!   HI: you will never be a commander  -> low sim -> drop

Run:
  python labse_filter.py --in conv_real/pairs.jsonl --out conv_clean --threshold 0.70

Deps: pip install sentence-transformers
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="LaBSE semantic-alignment filter")
    p.add_argument("--in", dest="inp", default="conv_real/pairs.jsonl")
    p.add_argument("--out", default="conv_clean")
    p.add_argument("--threshold", type=float, default=0.70,
                   help="keep pairs with cosine sim >= this")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--test-split", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    rows = []
    with open(args.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[info] loaded {len(rows):,} pairs from {args.inp}")

    en = [r["translation"]["en"] for r in rows]
    hi = [r["translation"]["hi"] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}; loading LaBSE ...")
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    print("[info] embedding English side ...")
    e_en = model.encode(en, batch_size=args.batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    print("[info] embedding Hindi side ...")
    e_hi = model.encode(hi, batch_size=args.batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)

    sims = (e_en * e_hi).sum(axis=1)  # cosine, since normalized

    # distribution report
    pct = {p: float(np.percentile(sims, p)) for p in (1, 5, 10, 25, 50, 75, 90)}
    print("\n----- similarity distribution -----")
    for p, v in pct.items():
        print(f"  p{p:<2d} = {v:.3f}")
    print("  keep counts at thresholds:")
    for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        k = int((sims >= t).sum())
        print(f"    >= {t:.2f} : {k:>9,}  ({100*k/len(sims):.1f}%)")

    keep_mask = sims >= args.threshold
    kept = [dict(rows[i], labse=float(sims[i])) for i in range(len(rows)) if keep_mask[i]]
    print(f"\n[info] threshold {args.threshold}: keeping {len(kept):,} / {len(rows):,} "
          f"({100*len(kept)/len(rows):.1f}%)")

    # show borderline drops to validate the threshold
    order = np.argsort(sims)
    print("\n----- lowest-sim pairs (being dropped) -----")
    for i in order[:6]:
        print(f"  sim={sims[i]:.2f} EN: {en[i]}")
        print(f"           HI: {hi[i]}")
    print("\n----- pairs just above threshold (kept) -----")
    above = [i for i in order if sims[i] >= args.threshold][:5]
    for i in above:
        print(f"  sim={sims[i]:.2f} EN: {en[i]}")
        print(f"           HI: {hi[i]}")

    # save
    jsonl = out / "pairs.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[info] wrote {jsonl}")

    from datasets import Dataset
    ds = Dataset.from_list(kept).train_test_split(test_size=args.test_split, seed=42)
    ds.save_to_disk(str(out / "hf"))
    print(f"[info] saved HF dataset -> {out/'hf'} "
          f"(train={len(ds['train'])}, test={len(ds['test'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
