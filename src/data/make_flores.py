#!/usr/bin/env python3
"""Build flores_test.jsonl (EN->HI) from BPCC's FLORES-22 dev (neutral benchmark)."""
import json
import zipfile
from huggingface_hub import hf_hub_download

p = hf_hub_download("ai4bharat/BPCC", "additional/flores-22_dev.zip", repo_type="dataset")
z = zipfile.ZipFile(p)
base = "flores-22_dev/all/eng_Latn-hin_Deva/"
en = z.read(base + "dev.eng_Latn").decode("utf-8").splitlines()
hi = z.read(base + "dev.hin_Deva").decode("utf-8").splitlines()
assert len(en) == len(hi), (len(en), len(hi))

with open("flores_test.jsonl", "w", encoding="utf-8") as f:
    for e, h in zip(en, hi):
        e, h = e.strip(), h.strip()
        if e and h:
            f.write(json.dumps({"translation": {"en": e, "hi": h}}, ensure_ascii=False) + "\n")
print(f"[done] wrote flores_test.jsonl ({len(en)} pairs)")
print("sample EN:", en[0])
print("sample HI:", hi[0])
