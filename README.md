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

## Repo layout

| Stage | Scripts |
|---|---|
| Data construction | `build_multiling_conv.py`, `build_ml_mixed.py`, `bpcc_daily_ingest.py`, `labse_filter.py` |
| Training | `finetune_it2_ml.py` (multilingual), `finetune_it2.py` (Hindi) |
| Souping | `make_soup.py` (memory-safe streaming weight averaging) |
| Evaluation | `ml_eval.py`, `make_flores.py`, `benchmark_translate.py` |
| Significance | `sig_test.py` (sacreBLEU paired bootstrap) |
| Human/LLM eval harness | `select_divergent.py`, `select_divergent_conv.py`, `gpt_judge.py`, `multi_model_judge.py`, `dual_criterion_judge.py`, `analyze_pref.py` |
| Paper build | `build_pdf.sh` (`WRITEUP.md` → `paper.pdf` / `paper.tex`) |

## Reproduce (outline)

```bash
# 1. build data (needs HF access to BPCC; see scripts for sources)
python build_multiling_conv.py      # -> conv_splits/ml_{train,dev,test}.jsonl
python build_ml_mixed.py            # -> conv_splits/ml_mixed_{train,dev}.jsonl

# 2. fine-tune IndicTrans2-1B (single A10G/larger; Python 3.11, transformers==4.40.2)
python finetune_it2_ml.py           # -> ft_it2_ml_mixed/

# 3. soup with the base
python make_soup.py 0.6 ft_it2_ml_mixed_soup ft_it2_ml_mixed

# 4. evaluate + significance
python ml_eval.py ft_it2_ml_mixed_soup hin,tam,ben,mar,mal
python sig_test.py
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
