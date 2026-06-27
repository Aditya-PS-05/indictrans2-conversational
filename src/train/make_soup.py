#!/usr/bin/env python3
"""
make_soup.py ALPHA OUTDIR
Memory-efficient WiSE-FT weight averaging: theta = (1-alpha)*base + alpha*ft.
Streams safetensors one tensor at a time (peak ~5GB) so it fits the box's 15GB
RAM -- loading two full 1B models at once OOMs.
Output is a full model dir (config + custom-code refs copied from ft_it2_mixed,
weights overwritten) benchmarkable via `ftit2:OUTDIR`.
"""
import os
import shutil
import sys

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

alpha = float(sys.argv[1])
out = sys.argv[2]
FT_DIR = sys.argv[3] if len(sys.argv) > 3 else "ft_it2_mixed"

base_sft = hf_hub_download("ai4bharat/indictrans2-en-indic-1B", "model.safetensors")
ft_sft = os.path.join(FT_DIR, "model.safetensors")

# copy everything except the weights from the ft dir (config, tokenizer, etc.)
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out)
for fn in os.listdir(FT_DIR):
    if fn.endswith(".safetensors"):
        continue
    src = os.path.join(FT_DIR, fn)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(out, fn))

merged = {}
with safe_open(base_sft, framework="pt", device="cpu") as bf, \
     safe_open(ft_sft, framework="pt", device="cpu") as ff:
    ff_keys = set(ff.keys())
    for k in bf.keys():
        bv = bf.get_tensor(k)
        if k in ff_keys:
            fv = ff.get_tensor(k)
            if fv.shape == bv.shape and bv.is_floating_point():
                bv = (1.0 - alpha) * bv + alpha * fv
            del fv
        merged[k] = bv

save_file(merged, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
print(f"[soup] alpha={alpha} -> {out}  ({len(merged)} tensors)")
