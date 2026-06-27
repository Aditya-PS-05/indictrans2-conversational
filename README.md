# Conversational Domain Adaptation of IndicTrans2 (English → 21 Indic languages)

Code and artifacts for the paper *"Conversational Domain Adaptation of IndicTrans2 across 21
Indic Languages via Experience Replay and Model Soups."*

We adapt [`ai4bharat/indictrans2-en-indic-1B`](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B)
to **casual / conversational** register using only public conversational data, while preserving
general-domain quality.

## Headline result (honest scope)

By **chrF2**, the adapted model **beats IndicTrans2-1B on conversational test sets in all 21
Indic languages** (mean **+6.2**, paired-bootstrap significant, p ≤ 0.004 on the hardest tests)
while **matching** the base on general-domain **FLORES** (mean −0.17, within ±0.7).

> ⚠️ This is a **chrF** result, not a verified human-quality win. Part of the conversational gain
> plausibly reflects *style-matching* to casual subtitle references rather than better adequacy;
> a controlled human evaluation has not yet confirmed a perceived quality improvement. The paper
> reports this openly (§6, §9). Claims here are deliberately scoped to the metric.

## Method (two established techniques)

1. **Experience replay** — fine-tune on conversational data mixed ~1:1 with a general anchor
   (BPCC-H-Wiki) to prevent catastrophic forgetting of FLORES.
2. **Model soup** — weight-average the fine-tuned model with the base, θ = (1−α)·θ_base + α·θ_FT
   (α = 0.6).

## Models

- **`<HF_USER>/indictrans2-en-indic-1B-conversational`** — the soup (recommended; `ft_it2_ml_mixed_soup`).
- *(optional)* the pre-soup mixed fine-tune (`ft_it2_ml_mixed`).

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
├── README.md  ·  MODEL_CARD.md  ·  LICENSE
```

| Stage | Location |
|---|---|
| Data construction | `src/data/` (`build_multiling_conv.py`, `build_ml_mixed.py`, `bpcc_daily_ingest.py`, `labse_filter.py`) |
| Training + souping | `src/train/` (`finetune_it2_ml.py`, `finetune_it2.py`, `make_soup.py`) |
| Automatic eval + significance | `src/eval/` (`ml_eval.py`, `benchmark_translate.py`, `make_flores.py`, `sig_test.py`) |
| Human/LLM eval harness | `src/eval/` (`select_divergent*.py`, `gpt_judge.py`, `multi_model_judge.py`, `dual_criterion_judge.py`, `analyze_pref.py`) + results in `eval_data/` |
| Paper | `paper/` (`build_pdf.sh`: `WRITEUP.md` → `paper.pdf` / `paper.tex`) |

> Run all scripts **from the repository root** (paths to `conv_splits/`, `sig/`, `eval_data/` are root-relative).

## Reproduce (outline)

```bash
# run everything from the repository root

# 1. build data (needs HF access to BPCC; see scripts for sources)
python src/data/build_multiling_conv.py   # -> conv_splits/ml_{train,dev,test}.jsonl
python src/data/build_ml_mixed.py         # -> conv_splits/ml_mixed_{train,dev}.jsonl

# 2. fine-tune IndicTrans2-1B (single A10G/larger; Python 3.11, transformers==4.40.2)
python src/train/finetune_it2_ml.py       # -> ft_it2_ml_mixed/

# 3. soup with the base
python src/train/make_soup.py 0.6 ft_it2_ml_mixed_soup ft_it2_ml_mixed

# 4. evaluate + significance
python src/eval/ml_eval.py ft_it2_ml_mixed_soup hin,tam,ben,mar,mal
python src/eval/sig_test.py

# 5. (optional) reproduce the human/LLM eval + build the paper
python src/eval/analyze_pref.py eval_data/claude_pref_conv_hin.csv eval_data/eval_pref_key_hin.csv
bash paper/build_pdf.sh                    # -> paper/paper.pdf + paper/paper.tex
```

Environment: Python 3.11, `transformers==4.40.2`, `IndicTransToolkit`, `sacreBLEU` 2.6, PyTorch (CUDA).

## License & attribution

MIT. This work is a **derivative of IndicTrans2** (© AI4Bharat, MIT) — please cite Gala et al.
(TMLR 2023). Data: BPCC (AI4Bharat), Tatoeba (CC-BY 2.0 FR), OpenSubtitles via OPUS (research use).

## Citation

```bibtex
@misc{singh2026convindic,
  title  = {Conversational Domain Adaptation of IndicTrans2 across 21 Indic Languages
            via Experience Replay and Model Soups},
  author = {Aditya Pratap Singh},
  year   = {2026},
  eprint = {arXiv:XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```
