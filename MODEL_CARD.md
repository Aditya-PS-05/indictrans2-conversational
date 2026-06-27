---
license: mit
language:
- en
- hi
- bn
- ta
- te
- ml
- mr
- gu
- kn
- pa
- or
- as
- ur
- ne
- sa
- mai
- kok
- doi
- brx
- sat
- mni
- ks
base_model: ai4bharat/indictrans2-en-indic-1B
pipeline_tag: translation
tags:
- machine-translation
- indic
- conversational
- domain-adaptation
- model-soup
- indictrans2
library_name: transformers
---

# IndicTrans2-1B — Conversational (English → 21 Indic languages)

A conversational/casual-register domain adaptation of
[`ai4bharat/indictrans2-en-indic-1B`](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B),
produced with **experience replay** (mixed fine-tuning) + **model souping** (WiSE-FT-style
weight averaging with the base, α=0.6).

> **Scope of the claim — please read.** On the **chrF2** metric, this model **beats
> IndicTrans2-1B on conversational test sets across all 21 Indic languages** (mean **+6.2**,
> paired-bootstrap significant, p ≤ 0.004 on the hardest subtitle tests) while **matching**
> the base on general-domain **FLORES** (mean change −0.17, all within ±0.7 chrF).
> This is a **chrF result**, not a verified human-quality improvement: part of the conversational
> gain plausibly reflects *style-matching* to casual subtitle references rather than better
> adequacy, and a controlled human evaluation has **not** yet confirmed a perceived quality gain.
> Use accordingly. See the paper for the full, honest evaluation and limitations.

## Results (chrF2, base → this model)

| Lang | conv | FLORES |
|---|---|---|
| Hindi | 53.9 → 56.8 (+2.9) | 60.0 → 59.6 |
| Tamil | 52.7 → 55.1 (+2.4) | 64.1 → 64.4 |
| Bengali | 68.7 → 73.6 (+4.8) | 58.1 → 58.1 |
| Malayalam | 45.7 → 48.4 (+2.7) | 61.1 → 61.4 |
| Marathi | 70.2 → 77.3 (+7.1) | 55.0 → 54.5 |
| **mean (21 langs)** | **+6.2** | **−0.17** |

Conversational gains are largest where the test set is BPCC-H-Daily (in-distribution style);
the most reliable gains are on the harder subtitle tests (Hindi/Tamil/Malayalam). See paper §6–§7.

## Intended use & limitations

- **Use for:** English → Indic translation of **casual / conversational** text (chat, dialogue, subtitles).
- **Direction:** English → Indic only. Indic → English / Indic ↔ Indic are untested.
- **Register:** tuned toward casual register; for formal/news text the base model is at least as good.
- **Metric caveat:** improvements are measured by chrF2 and are partly stylistic; **not** human-verified.
- **Low-resource ceiling:** very low-resource languages (e.g. Santali, Sanskrit, Kashmiri) remain
  bounded by the base model's absolute quality.

## How to use

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor   # pip install IndicTransToolkit

REPO = "adipras1407/indictrans2-en-indic-1B-conversational"
# Load the tokenizer from the base checkpoint (recommended).
tok = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True).eval()
ip = IndicProcessor(inference=True)

src = ["Are you coming over tonight?", "I missed you so much, yaar."]
batch = ip.preprocess_batch(src, src_lang="eng_Latn", tgt_lang="hin_Deva")
enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
with torch.inference_mode():
    out = model.generate(**enc, num_beams=5, max_length=256)
hyp = ip.postprocess_batch(tok.batch_decode(out, skip_special_tokens=True), lang="hin_Deva")
print(hyp)
```
Use the appropriate target tag (e.g. `tam_Taml`, `ben_Beng`, `mar_Deva`, …) for other languages.

## Training

- **Base:** `ai4bharat/indictrans2-en-indic-1B` (1.1B params).
- **Recipe:** mixed fine-tuning on 538k pairs (284k conversational + 254k BPCC-H-Wiki general
  anchor), 1 epoch, bf16, Adafactor, single A10G GPU; then weight-averaged with the base at α=0.6.
- **Conversational data:** OPUS OpenSubtitles, BPCC-H-Daily, Tatoeba (English→Indic, per-pair
  target-language tag), filtered for length/script; Hindi additionally LaBSE-filtered ≥0.70.

## Data licensing & attribution

This is a derivative of **IndicTrans2** (© AI4Bharat, MIT license) — please cite their work.
Training data: **BPCC** (AI4Bharat, open license), **Tatoeba** (CC-BY 2.0 FR), and **OpenSubtitles**
via **OPUS** (research use; consult OPUS/OpenSubtitles terms before any commercial use). The released
weights are provided for research; verify the upstream data terms for your use case.

## Citation

```bibtex
@misc{singh2026convindic,
  title  = {Conversational Domain Adaptation of IndicTrans2 across 21 Indic Languages
            via Experience Replay and Model Soups},
  author = {Aditya Pratap Singh},
  year   = {2026},
  eprint = {arXiv:XXXX.XXXXX},   % update once arXiv assigns the ID
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

Please also cite IndicTrans2 (Gala et al., TMLR 2023).

## License

MIT (inherited from the IndicTrans2 base model).
