<!-- <CENTERED SECTION FOR GITHUB DISPLAY> -->

<div align="center">

# 🪔 IndicTrans2-Conversational

**English → 21 Indic languages, adapted for casual / conversational register**

[![License](https://img.shields.io/badge/license-MIT-0073FF?labelColor=black&style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-0073FF?labelColor=black&style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Base model](https://img.shields.io/badge/base-IndicTrans2--1B-0073FF?labelColor=black&style=flat-square)](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
[![Languages](https://img.shields.io/badge/languages-21%20Indic-0073FF?labelColor=black&style=flat-square)](#results)
[![arXiv](https://img.shields.io/badge/arXiv-pending-b31b1b?labelColor=black&style=flat-square&logo=arxiv)](#citation)

[![GitHub Stars](https://img.shields.io/github/stars/Aditya-PS-05/indictrans2-conversational?color=0073FF&labelColor=black&style=flat-square)](https://github.com/Aditya-PS-05/indictrans2-conversational/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Aditya-PS-05/indictrans2-conversational?color=0073FF&labelColor=black&style=flat-square)](https://github.com/Aditya-PS-05/indictrans2-conversational/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/Aditya-PS-05/indictrans2-conversational?color=0073FF&labelColor=black&style=flat-square)](https://github.com/Aditya-PS-05/indictrans2-conversational/issues)

[ 📄 Paper (PDF) ](./paper/paper.pdf) · [ 🤗 Model (HF) ](#models) · [ 🧪 Eval harness ](./src/eval) · [ 📊 Results ](#results)

</div>

<!-- </CENTERED SECTION FOR GITHUB DISPLAY> -->

> A domain-adapted **IndicTrans2-1B** that **beats the base model on conversational chrF across all 21 Indic languages** while **matching it on general-domain FLORES** — built from public data with experience replay + model souping, and evaluated *honestly* (including where the automatic metric overstates the gain).

> [!IMPORTANT]
>
> **Scope of the claim — read this.** "Beats" is on the **chrF2** metric, on conversational test sets. It is a *metric* result, **not** a verified human-quality win.
> Part of the conversational gain plausibly reflects **style-matching to casual subtitle references** rather than better adequacy, and a controlled human evaluation has **not yet confirmed** a perceived quality improvement. The full, blind human + multi-LLM judge study (which leans toward parity/base) is released in [`eval_data/`](./eval_data) and discussed in the paper §6, §9. We keep the claim scoped on purpose.

<div align="center">

| | Conversational chrF2 | General (FLORES) chrF2 |
|:---|:---:|:---:|
| **vs. IndicTrans2-1B base** | ✅ **+6.2 mean** (all 21 langs) | ⟂ **−0.17 mean** (tie, within ±0.7) |
| **Significance** | p ≤ 0.004 (paired bootstrap, hardest tests) | not significantly degraded |

</div>

## Contents

- [Results](#results)
- [Method](#method)
- [Models](#models)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Reproduce](#reproduce)
- [Evaluation — the honest part](#evaluation--the-honest-part)
- [Data & licensing](#data--licensing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Results

Base → adapted (soup), chrF2. Full 21-language table in the [paper](./paper/paper.pdf).

| Lang | Conversational | General (FLORES) |
|------|:---:|:---:|
| Hindi | 53.9 → **56.8** (+2.9) | 60.0 → 59.6 |
| Tamil | 52.7 → **55.1** (+2.4) | 64.1 → 64.4 |
| Bengali | 68.7 → **73.6** (+4.8) | 58.1 → 58.1 |
| Malayalam | 45.7 → **48.4** (+2.7) | 61.1 → 61.4 |
| Marathi | 70.2 → **77.3** (+7.1) | 55.0 → 54.5 |
| **mean (21 langs)** | **+6.2** | **−0.17** |

> [!NOTE]
> The largest conversational gains occur where the test set is in-distribution (BPCC-H-Daily); the most reliable gains are on harder subtitle tests (Hindi / Tamil / Malayalam). See paper §6–§7.

## Method

Two established techniques, applied across the whole Indic family:

1. **Experience replay** — fine-tune IndicTrans2-1B on conversational data mixed ~1:1 with a general anchor (BPCC-H-Wiki) to prevent catastrophic forgetting of FLORES.
2. **Model soup** (WiSE-FT-style) — weight-average the fine-tuned model with the base:
   `θ = (1 − α)·θ_base + α·θ_FT`, with **α = 0.6**.

Naive conversational fine-tuning alone gains +5.4 conv chrF on Hindi but **loses −3.9 on FLORES** (catastrophic forgetting); the replay+soup recipe keeps the conversational gain *and* restores FLORES to base level.

## Models

| Model | What | Link |
|---|---|---|
| **Soup (recommended)** | `ft_it2_ml_mixed_soup` — replay-FT averaged with base (α=0.6) | 🤗 `<HF_USER>/indictrans2-en-indic-1B-conversational` *(upload pending)* |
| Mixed-FT | `ft_it2_ml_mixed` — pre-soup fine-tune | *(optional release)* |

See [`MODEL_CARD.md`](./MODEL_CARD.md) for usage code, intended use, and limitations.

## Repository structure

```
.
├── src/
│   ├── data/      # dataset construction (build_multiling_conv, build_ml_mixed, labse_filter, …)
│   ├── train/     # fine-tuning + souping (finetune_it2_ml, finetune_it2, make_soup)
│   └── eval/      # automatic metrics + significance + human/LLM judge harness
├── eval_data/     # committed evaluation results (divergent pairs, judge verdicts, keys)
├── paper/         # WRITEUP.md, build_pdf.sh -> paper.pdf / paper.tex
├── scripts/       # box-automation drivers (run from repo root)
├── docs/          # DATASET_STEPS.md, EVAL_INSTRUCTIONS.md
└── README.md · MODEL_CARD.md · LICENSE
```

| Stage | Location |
|---|---|
| Data construction | `src/data/` (`build_multiling_conv.py`, `build_ml_mixed.py`, `bpcc_daily_ingest.py`, `labse_filter.py`) |
| Training + souping | `src/train/` (`finetune_it2_ml.py`, `finetune_it2.py`, `make_soup.py`) |
| Automatic eval + significance | `src/eval/` (`ml_eval.py`, `benchmark_translate.py`, `make_flores.py`, `sig_test.py`) |
| Human/LLM eval harness | `src/eval/` (`select_divergent*.py`, `gpt_judge.py`, `multi_model_judge.py`, `dual_criterion_judge.py`, `analyze_pref.py`) + results in `eval_data/` |
| Paper | `paper/` (`build_pdf.sh`: `WRITEUP.md` → `paper.pdf` / `paper.tex`) |

## Quick start

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor          # pip install IndicTransToolkit

REPO = "<HF_USER>/indictrans2-en-indic-1B-conversational"
tok = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(REPO, trust_remote_code=True).eval()
ip = IndicProcessor(inference=True)

src = ["Are you coming over tonight?", "I missed you so much, yaar."]
batch = ip.preprocess_batch(src, src_lang="eng_Latn", tgt_lang="hin_Deva")
enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
with torch.inference_mode():
    out = model.generate(**enc, num_beams=5, max_length=256)
print(ip.postprocess_batch(tok.batch_decode(out, skip_special_tokens=True), lang="hin_Deva"))
```

Use the right target tag for other languages (`tam_Taml`, `ben_Beng`, `mar_Deva`, …).

## Reproduce

> [!TIP]
> Run **all** commands from the repository root (paths to `conv_splits/`, `sig/`, `eval_data/` are root-relative). Environment: Python 3.11, `transformers==4.40.2`, `IndicTransToolkit`, `sacreBLEU` 2.6, PyTorch + CUDA.

```bash
# 1. build data (needs HF access to BPCC; see scripts for sources)
python src/data/build_multiling_conv.py   # -> conv_splits/ml_{train,dev,test}.jsonl
python src/data/build_ml_mixed.py         # -> conv_splits/ml_mixed_{train,dev}.jsonl

# 2. fine-tune IndicTrans2-1B (single A10G/larger)
python src/train/finetune_it2_ml.py       # -> ft_it2_ml_mixed/

# 3. soup with the base (α = 0.6)
python src/train/make_soup.py 0.6 ft_it2_ml_mixed_soup ft_it2_ml_mixed

# 4. evaluate + significance
python src/eval/ml_eval.py ft_it2_ml_mixed_soup hin,tam,ben,mar,mal
python src/eval/sig_test.py

# 5. (optional) reproduce the human/LLM eval + build the paper
python src/eval/analyze_pref.py eval_data/claude_pref_conv_hin.csv eval_data/eval_pref_key_hin.csv
bash paper/build_pdf.sh                    # -> paper/paper.pdf + paper/paper.tex
```

## Evaluation — the honest part

chrF measures n-gram overlap with references, not quality. Because our conversational references are **subtitles** (casual register), a model that simply *sounds* more casual gains chrF whether or not its meaning is better. To test whether the gain is real quality, we ran a **blind** pairwise study on Hindi (the soup's most favorable case):

- **Human** annotator → no significant preference (≈ tie).
- **Claude** (conversational rubric) → base preferred (41% soup, n.s.).
- **OpenAI panel (GPT-4-class → GPT-5.x)**, position-bias-controlled, two rubrics, two test sets → base preferred or tied; **no fair setting significantly prefers the soup**.

All of this — the divergent pairs, the judge verdicts, the keys, and the harness — is in [`eval_data/`](./eval_data) and [`src/eval/`](./src/eval), reproducible with `analyze_pref.py`. The paper reports it openly. **Takeaway: automatic metrics overstate conversational domain-adaptation gains; the chrF win is real but largely stylistic.**

## Data & licensing

Derivative of **IndicTrans2** (© AI4Bharat, MIT) — please cite Gala et al. (TMLR 2023). Training data: **BPCC** (AI4Bharat, open), **Tatoeba** (CC-BY 2.0 FR), **OpenSubtitles** via **OPUS** (research use — check OPUS/OpenSubtitles terms before commercial use).

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

## Acknowledgments

- [**IndicTrans2** / AI4Bharat](https://github.com/AI4Bharat/IndicTrans2) — the base model and BPCC data this work builds on.
- [Model Soups](https://arxiv.org/abs/2203.05482) (Wortsman et al.) and [WiSE-FT](https://arxiv.org/abs/2109.01903) — the weight-averaging recipe.
- [OPUS / OpenSubtitles](https://opus.nlpl.eu/), [Tatoeba](https://tatoeba.org/), [sacreBLEU](https://github.com/mjpost/sacrebleu), [FLORES-200](https://github.com/facebookresearch/flores).

## License

<div align="center">

**MIT** © [Aditya Pratap Singh](https://github.com/Aditya-PS-05) · inherited from the IndicTrans2 base model.

If this is useful, consider **starring the repo ⭐** and citing the paper.

</div>
