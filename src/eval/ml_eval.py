#!/usr/bin/env python3
"""
ml_eval.py MODEL_DIR LANG1,LANG2,...
Evaluate one model on EN->Indic for each language, on BOTH:
  * conversational held-out test (conv_splits/ml_test.jsonl, filtered to lang)
  * FLORES devtest (neutral/general, from BPCC flores-22 zip)
Loads the model once; prints RESULT lines for a table.
"""
import json
import sys
import zipfile

import sacrebleu
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

model_dir = sys.argv[1]
langs = sys.argv[2].split(",")
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 400
BASE_TOK = "ai4bharat/indictrans2-en-indic-1B"

ip = IndicProcessor(inference=True)
tok = AutoTokenizer.from_pretrained(BASE_TOK, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_dir, trust_remote_code=True, torch_dtype=torch.float16).to("cuda").eval()

# conversational test, grouped by lang
conv = {}
for line in open("conv_splits/ml_test.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    conv.setdefault(r["lang"], []).append((r["translation"]["en"], r["translation"]["tgt"]))

# FLORES zip
zp = hf_hub_download("ai4bharat/BPCC", "additional/flores-22_dev.zip", repo_type="dataset")
fz = zipfile.ZipFile(zp)


def translate(en, tgt_lang):
    out = []
    for i in range(0, len(en), 16):
        chunk = en[i:i + 16]
        pre = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=tgt_lang)
        enc = tok(pre, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.inference_mode():
            g = model.generate(**enc, num_beams=5, max_length=256)
        dec = tok.batch_decode(g, skip_special_tokens=True)
        out.extend(ip.postprocess_batch(dec, lang=tgt_lang))
    return out


for lang in langs:
    # conversational
    items = conv.get(lang, [])[:LIMIT]
    if items:
        en = [x[0] for x in items]
        ref = [x[1] for x in items]
        hyp = translate(en, lang)
        c_conv = sacrebleu.corpus_chrf(hyp, [ref]).score
    else:
        c_conv = float("nan")
    # FLORES
    try:
        p = f"flores-22_dev/all/eng_Latn-{lang}/"
        fen = fz.read(p + "dev.eng_Latn").decode("utf-8").splitlines()[:LIMIT]
        fref = fz.read(p + f"dev.{lang}").decode("utf-8").splitlines()[:LIMIT]
        fhyp = translate(fen, lang)
        c_flores = sacrebleu.corpus_chrf(fhyp, [fref]).score
    except Exception:
        c_flores = float("nan")
    print(f"RESULT {model_dir} {lang}: conv={c_conv:.2f} flores={c_flores:.2f}")
