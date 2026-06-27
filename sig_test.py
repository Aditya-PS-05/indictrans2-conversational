#!/usr/bin/env python3
"""
sig_test.py — paired bootstrap significance (sacrebleu) for base vs mixed-soup.
For the 3 'honest' (harder-test) langs, translate conv + FLORES with both models,
then run sacrebleu --paired-bs to get chrF + p-value of soup vs base.
"""
import gc
import json
import os
import subprocess
import zipfile

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

LANGS = ["hin_Deva", "tam_Taml", "mal_Mlym"]
LIMIT = 500
MODELS = {"base": "ai4bharat/indictrans2-en-indic-1B", "soup": "ft_it2_ml_mixed_soup"}
BASE_TOK = "ai4bharat/indictrans2-en-indic-1B"

ip = IndicProcessor(inference=True)
tok = AutoTokenizer.from_pretrained(BASE_TOK, trust_remote_code=True)

conv = {}
for line in open("conv_splits/ml_test.jsonl", encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        conv.setdefault(r["lang"], []).append((r["translation"]["en"], r["translation"]["tgt"]))
zp = hf_hub_download("ai4bharat/BPCC", "additional/flores-22_dev.zip", repo_type="dataset")
fz = zipfile.ZipFile(zp)


def testset(lang, kind):
    if kind == "conv":
        items = conv.get(lang, [])[:LIMIT]
        return [x[0] for x in items], [x[1] for x in items]
    p = f"flores-22_dev/all/eng_Latn-{lang}/"
    en = fz.read(p + "dev.eng_Latn").decode("utf-8").splitlines()[:LIMIT]
    ref = fz.read(p + f"dev.{lang}").decode("utf-8").splitlines()[:LIMIT]
    return en, ref


def translate(model, en, lang):
    out = []
    for i in range(0, len(en), 16):
        chunk = en[i:i + 16]
        pre = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=lang)
        enc = tok(pre, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.inference_mode():
            g = model.generate(**enc, num_beams=5, max_length=256)
        out.extend(ip.postprocess_batch(tok.batch_decode(g, skip_special_tokens=True), lang=lang))
    return out


os.makedirs("sig", exist_ok=True)
for tag, mid in MODELS.items():
    model = AutoModelForSeq2SeqLM.from_pretrained(
        mid, trust_remote_code=True, torch_dtype=torch.float16).to("cuda").eval()
    for lang in LANGS:
        for kind in ["conv", "flores"]:
            en, ref = testset(lang, kind)
            hyp = translate(model, en, lang)
            with open(f"sig/{lang}_{kind}.ref", "w", encoding="utf-8") as f:
                f.write("\n".join(ref))
            with open(f"sig/{lang}_{kind}.{tag}", "w", encoding="utf-8") as f:
                f.write("\n".join(hyp))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[done translating] {tag}", flush=True)

print("\n========== PAIRED BOOTSTRAP (chrF, soup vs base) ==========", flush=True)
for lang in LANGS:
    for kind in ["conv", "flores"]:
        ref = f"sig/{lang}_{kind}.ref"
        print(f"\n### {lang} {kind} ###", flush=True)
        r = subprocess.run(
            ["sacrebleu", ref, "-i", f"sig/{lang}_{kind}.base", f"sig/{lang}_{kind}.soup",
             "-m", "chrf", "--paired-bs", "--format", "text"],
            capture_output=True, text=True)
        print(r.stdout or r.stderr[:300], flush=True)
print("SIG_DONE", flush=True)
