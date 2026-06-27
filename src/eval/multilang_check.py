#!/usr/bin/env python3
"""
multilang_check.py MODEL_DIR TGT_LANG [LIMIT]
Did Hindi fine-tuning break English->other-Indic-language translation?
Translates FLORES EN->TGT with the given model and scores chrF.
TGT_LANG e.g. ben_Beng, tam_Taml, mar_Deva, tel_Telu.
"""
import sys
import zipfile

import sacrebleu
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

model_dir = sys.argv[1]
tgt = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500
BASE_TOK = "ai4bharat/indictrans2-en-indic-1B"

zp = hf_hub_download("ai4bharat/BPCC", "additional/flores-22_dev.zip", repo_type="dataset")
z = zipfile.ZipFile(zp)
p = f"flores-22_dev/all/eng_Latn-{tgt}/"
en = z.read(p + "dev.eng_Latn").decode("utf-8").splitlines()[:limit]
ref = z.read(p + f"dev.{tgt}").decode("utf-8").splitlines()[:limit]

ip = IndicProcessor(inference=True)
tok = AutoTokenizer.from_pretrained(BASE_TOK, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_dir, trust_remote_code=True, torch_dtype=torch.float16).to("cuda").eval()

out = []
bs = 16
for i in range(0, len(en), bs):
    chunk = en[i:i + bs]
    pre = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=tgt)
    enc = tok(pre, return_tensors="pt", padding=True, truncation=True,
              max_length=256).to("cuda")
    with torch.inference_mode():
        g = model.generate(**enc, num_beams=5, max_length=256)
    dec = tok.batch_decode(g, skip_special_tokens=True)
    out.extend(ip.postprocess_batch(dec, lang=tgt))

chrf = sacrebleu.corpus_chrf(out, [ref]).score
print(f"RESULT {model_dir} EN->{tgt}: chrF2={chrf:.2f}")
print(f"  EN : {en[0]}")
print(f"  REF: {ref[0]}")
print(f"  HYP: {out[0]}")
