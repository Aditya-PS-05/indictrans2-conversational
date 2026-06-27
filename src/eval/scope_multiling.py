#!/usr/bin/env python3
"""Scope BPCC-H-Daily conversational volume across all 22 Indic languages."""
import csv
from huggingface_hub import hf_hub_download

csv.field_size_limit(10_000_000)

LANGS = ["asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
         "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
         "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "sat_Olck",
         "snd_Deva", "tam_Taml", "tel_Telu", "urd_Arab"]

total = 0
print(f"{'lang':10s} {'rows':>8s}")
for lg in LANGS:
    try:
        p = hf_hub_download("ai4bharat/BPCC", f"daily/{lg}.tsv", repo_type="dataset")
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1  # minus header
        total += n
        print(f"{lg:10s} {n:>8d}")
    except Exception as e:
        print(f"{lg:10s} {'FAIL':>8s}  {str(e)[:50]}")
print(f"{'TOTAL':10s} {total:>8d}")
