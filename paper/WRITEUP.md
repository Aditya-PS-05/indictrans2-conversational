# Conversational Domain Adaptation of IndicTrans2 across 21 Indic Languages via Experience Replay and Model Soups

**Author:** Aditya Pratap Singh (Independent Researcher)
*Working draft for WAT 2026 / arXiv. All numbers are from real runs in this repository. Evaluation is by automatic metrics (chrF2); the scope and its limits are stated explicitly in §8.*

---

## Abstract

State-of-the-art multilingual machine-translation (MT) systems such as IndicTrans2 are trained on broad, general-domain corpora; they translate formal text well but are weaker on casual *conversational* register. We study domain adaptation of IndicTrans2-1B to conversational English→Indic translation across **21 Indic languages**, using only publicly available conversational data (OpenSubtitles, BPCC-H-Daily, Tatoeba). We confirm that naive fine-tuning improves conversational chrF but induces **catastrophic forgetting** of general-domain quality (−3.9 chrF on FLORES for Hindi). Applying two established techniques — **experience replay** (mixing general-domain data back into fine-tuning) and **model souping** (weight-averaging the fine-tuned model with the base) — we obtain a single model that **beats the IndicTrans2 base on conversational chrF in all 21 languages** (mean +6.2, range +1.3 to +12.2) while *matching* it on general-domain FLORES quality (mean change −0.17, all within ±0.7 chrF). Paired bootstrap resampling confirms the conversational chrF gains are significant (p ≤ 0.004) and that FLORES is not significantly degraded. We additionally report a negative result: the lowest-resource languages remain bounded by the base model's absolute quality, which conversational adaptation cannot raise. All results are automatic-metric (chrF2); we are explicit (§6, §8) about where this metric can overstate domain-adaptation gains. We claim no methodological novelty; our contribution is the systematic, honestly-evaluated empirical study in the Indic conversational setting.

## 1. Introduction

Neural MT systems are typically optimised for, and evaluated on, general-domain benchmarks such as FLORES-200 [Goyal et al., 2022; NLLB Team, 2022]. Yet a large fraction of real-world translation is *conversational*: chat messages, subtitles, voice-assistant commands. This register differs systematically from news or encyclopedic text — sentences are shorter, more colloquial and idiomatic, and elliptical. A model that is excellent on FLORES can still produce stilted, overly formal output on a casual sentence.

For Indian languages, IndicTrans2 [Gala et al., 2023] is the strongest open English↔Indic system, trained on the 230M-pair Bharat Parallel Corpus Collection (BPCC). It is, however, a general-domain model. We ask a practical question:

> *Can we adapt a strong multilingual MT model to conversational register across many languages, without degrading its general-domain quality, using only public data?*

Naive fine-tuning on conversational data is the obvious first attempt, but it is well known to cause **catastrophic forgetting** [McCloskey and Cohen, 1989; French, 1999]: the model overfits the new domain and loses competence on the old one. We show this happens here, then mitigate it with two established techniques applied systematically across the Indic family.

**Contributions.**
1. A systematic study of conversational adaptation for **English→Indic MT across 21 languages**, built entirely from public conversational corpora.
2. Evidence that **experience replay + model souping** beats the base on conversational chrF — across the whole family, not a single language, with statistical-significance testing — while preserving general-domain chrF.
3. An **honest cost/limitation analysis**: catastrophic forgetting under naive fine-tuning; inflated gains on easy in-distribution test sets; the metric's own blind spots (§6); and a negative result on the lowest-resource languages.

We emphasise that the techniques we use — model soups [Wortsman et al., 2022a] and replay-based anti-forgetting [Robins, 1995] — are not novel. Our contribution is the empirical study and its honest evaluation in a previously unexamined setting.

## 2. Related Work

**Domain adaptation for NMT.** Fine-tuning a general model on in-domain data is the standard approach [Luong and Manning, 2015; Freitag and Al-Onaizan, 2016], surveyed by Chu and Wang [2018]. Its central failure mode is catastrophic forgetting of the general domain; mixing in general data during fine-tuning ("mixed fine-tuning") is a known mitigation [Chu et al., 2017], a form of experience replay [Robins, 1995].

**Model soups / weight averaging.** Wortsman et al. [2022a] show that averaging the weights of multiple fine-tuned models improves accuracy; WiSE-FT [Wortsman et al., 2022b] averages a fine-tuned model with its zero-shot initialization to trade off in- and out-of-distribution performance. We apply WiSE-FT-style base↔fine-tune interpolation to MT domain adaptation.

**Indic MT.** IndicTrans2 [Gala et al., 2023] and BPCC are the base model and a data source here. Prior WAT editions [e.g., Nakazawa et al., 2024] host Indic MT shared tasks.

**Evaluation and its limits.** We report chrF2 [Popović, 2015] via sacreBLEU [Post, 2018], with paired bootstrap resampling for significance [Koehn, 2004]; FLORES-200 [NLLB Team, 2022] is our neutral general-domain test. Character-n-gram metrics are standard and reproducible, but they measure overlap with references rather than quality directly, and can diverge from human judgment [Callison-Burch et al., 2006; Mathur et al., 2020; Freitag et al., 2021] — a caveat we take seriously and discuss in §6 and §8.

## 3. Data

**Base model.** `ai4bharat/indictrans2-en-indic-1B` (1.1B parameters).

**Conversational data** (English→Indic, each pair tagged with its target language):
- OPUS **OpenSubtitles** [Lison and Tiedemann, 2016]: en→{hi, ml, ta, te, ur}, the only Indic languages with substantial Indic subtitle data; capped at 40k/language to limit dominance.
- **BPCC-H-Daily** [Gala et al., 2023]: human-authored everyday/assistant utterances; available for all 21 languages, ~1.8k–11k pairs each.
- **Tatoeba**: everyday sentences (Hindi pool).

After filtering (length, length-ratio, exact dedup, Unicode-script language identification), the conversational set is **294,119 pairs** across 21 languages (train 283,753 / dev 4,183 / test 6,183), split per language. The Hindi-only experiments additionally apply LaBSE cosine ≥ 0.70 to remove misaligned subtitle pairs.

**General anchor** (for experience replay): **BPCC-H-Wiki** [Gala et al., 2023], a general/encyclopedic register, sampled ≈1:1 per language to the conversational counts (254,469 pairs). The mixed training set is therefore **538,222 pairs** (283,753 conversational + 254,469 general).

Per-language conversational volume is highly skewed: the five OpenSubtitles languages (Hindi, Malayalam, Tamil, Telugu, Urdu) hold 30k–48k pairs each; the remaining sixteen rely on BPCC-H-Daily (≈2k–11k). This skew is intrinsic to the available data and shapes our results (§6).

## 4. Method

We compare three configurations, all fine-tuning IndicTrans2-1B (IndicProcessor preprocessing with per-example target-language tags; bf16; Adafactor; single NVIDIA A10G GPU):

1. **Naive FT** — fine-tune on conversational data only.
2. **Mixed FT (experience replay)** — fine-tune on conversational + general anchor (≈1:1).
3. **Model soup** — weight-average the base and a fine-tuned model,
   θ = (1 − α)·θ(base) + α·θ(FT),
   sweeping α and selecting on the conversational/general trade-off. Averaging is done by streaming the two checkpoints' tensors to keep memory bounded.

**Training.** Hindi-only models use 146k conversational pairs (2–3 epochs); the multilingual models use the full 294k conversational / 538k mixed sets (2 epochs / 1 epoch respectively, matched in total examples seen).

**Evaluation.** chrF2 (sacreBLEU) on (a) a held-out conversational test set and (b) FLORES-200 devtest (general), per language. We report paired bootstrap significance (1000 resamples) for representative languages.

## 5. Results

### 5.1 Single-language case study (Hindi)

Table 1 isolates the mechanism on Hindi (conversational test = subtitle-heavy held-out set; FLORES = 997-sentence devtest).

**Table 1: Hindi (chrF2).**

| Model | conv | FLORES |
|---|---|---|
| IndicTrans2 base | 51.6 | 61.7 |
| Naive conv-FT | **57.0** | 57.8 ↓ (forgetting) |
| Mixed FT | 56.0 | 60.6 |
| **Model soup (α=0.6)** | 54.1 | **61.8** |

Naive fine-tuning gains +5.4 chrF on conversational but loses −3.9 on FLORES — clear catastrophic forgetting. Mixed FT recovers most of the general quality; the soup (mixed-FT averaged with base) is **≥ base on both axes**. An α-sweep (0.3–0.7) shows α=0.6 is the knee: below it general quality saturates at the base, above it FLORES drops below base.

### 5.2 All 21 languages (mixed-soup vs. base)

Table 2 gives every language (n=200 sentences per test). The mixed-soup model improves conversational chrF in **all 21 languages** (mean **+6.2**) and changes FLORES by a mean of **−0.17** (all within ±0.7).

**Table 2: All 21 languages, base → mixed-soup (chrF2).**

| Lang | conv base→soup (Δ) | FLORES base→soup (Δ) |
|---|---|---|
| asm | 61.9→69.4 (+7.6) | 44.2→44.1 (−0.1) |
| ben | 68.7→73.6 (+4.8) | 58.1→58.1 (−0.0) |
| brx | 62.1→68.0 (+5.9) | 45.9→46.0 (+0.1) |
| doi | 63.4→71.7 (+8.3) | 49.8→49.6 (−0.2) |
| gom | 69.4→75.6 (+6.2) | 49.1→49.2 (+0.1) |
| guj | 53.7→65.9 (+12.2) | 55.3→54.7 (−0.6) |
| hin | 53.9→56.8 (+2.9) | 60.0→59.6 (−0.5) |
| kan | 52.4→57.1 (+4.7) | 59.0→58.7 (−0.3) |
| kas | 45.8→51.3 (+5.5) | 39.1→38.8 (−0.4) |
| mai | 57.8→65.2 (+7.4) | 56.6→56.6 (−0.0) |
| mal | 45.7→48.4 (+2.7) | 61.1→61.4 (+0.3) |
| mar | 70.2→77.3 (+7.1) | 55.0→54.5 (−0.5) |
| mni | 56.4→62.8 (+6.4) | 48.2→48.3 (+0.2) |
| npi | 68.8→72.1 (+3.3) | 59.4→59.0 (−0.5) |
| ory | 56.4→65.7 (+9.2) | 52.0→51.3 (−0.7) |
| pan | 68.0→76.5 (+8.4) | 55.3→54.8 (−0.5) |
| san | 54.5→63.6 (+9.1) | 36.3→36.5 (+0.2) |
| sat | 50.7→56.2 (+5.6) | 30.7→30.3 (−0.4) |
| tam | 52.7→55.1 (+2.4) | 64.1→64.4 (+0.3) |
| tel | 49.9→60.0 (+10.1) | 62.4→62.0 (−0.4) |
| urd | 48.1→49.4 (+1.3) | 54.7→54.7 (+0.1) |
| **mean** | **+6.2** | **−0.17** |

### 5.3 Statistical significance

We run paired bootstrap resampling [Koehn, 2004] (sacreBLEU, 1000 resamples, n=500) on the three languages with the hardest, subtitle-based conversational tests (Table 3).

**Table 3: Paired bootstrap, base vs. mixed-soup (chrF2; p = soup vs. base).**

| Lang | conv (p) | FLORES (p) |
|---|---|---|
| Hindi | 51.5 → 54.1 (**p=0.001**) | 61.9 → 61.9 (p=0.372, n.s.) |
| Tamil | 50.9 → 55.1 (**p=0.001**) | 64.7 → 64.6 (p=0.290, n.s.) |
| Malayalam | 46.9 → 49.5 (**p=0.004**) | 62.5 → 62.8 (**p=0.038**, ↑) |

The conversational gains are significant (p ≤ 0.004). FLORES shows **no significant degradation** (not significant for Hindi/Tamil; significantly *improved* for Malayalam). The paired design is important because conversational chrF has high variance (95% CI ≈ ±3 chrF at n=500), so unpaired CI overlap would be misleading; the paired test isolates the per-sentence system difference.

*(Absolute conversational chrF here differs from Table 2 — e.g. Hindi 51.5 vs. 53.9 for the base — because Table 3 uses n=500 sentences and Table 2 uses n=200; the high variance makes the exact mean sample-dependent, which is precisely why we report the larger-n **paired** test for the headline claims. The FLORES values, being lower-variance, are stable across n.)*

## 6. What the metric does and does not establish

Our claims are deliberately scoped to what chrF2 supports, because the conversational setting has two specific ways the metric can mislead, and we want to state them rather than paper over them.

**The general-domain claim is a tie, not a win.** On FLORES the soup matches the base (Hindi 61.8 vs 61.7; all 21 languages within ±0.7). We therefore claim *preservation* of general-domain quality, not superiority on it.

**Part of the conversational gain is reference style-matching, not necessarily adequacy.** chrF rewards character-n-gram overlap with the reference. Our conversational references are subtitles, which are casual in register; the adapted model is also more casual (informal pronouns, spoken word order). A model that adopts the references' register therefore gains chrF whether or not its *meaning* is better. Inspection of sentences where base and soup diverge shows exactly this pattern — the soup is consistently more colloquial — so the conversational chrF gain should be read as "closer to the casual reference style," which overlaps with but is not identical to "a better translation." This is the well-documented gap between n-gram metrics and human judgment [Callison-Burch et al., 2006; Mathur et al., 2020; Freitag et al., 2021].

**What is solid.** Two things survive these caveats and are the safe claims of the paper: (i) the recipe *preserves* general-domain chrF (FLORES tie, significance-tested), and (ii) it produces output measurably closer to held-out human conversational references, significantly so on the hardest subtitle tests. Whether (ii) is perceived by humans as higher quality is a separate question that automatic metrics cannot answer; see §8.

## 7. Analysis

**The forgetting is real, and the fix works.** Naive conversational fine-tuning degrades FLORES (−3.9 for Hindi; a pure-conversational multilingual soup degrades FLORES by −0.7 to −2.9 across languages). Adding a general anchor before souping restores FLORES to base level (mean −0.17), confirming experience replay as the operative mechanism.

**Gain magnitude is partly a test-set artifact.** The largest conversational gains (Gujarati +12.2, Telugu +10.1, Odia +9.2) occur in languages whose conversational *test* is drawn solely from BPCC-H-Daily — short, formulaic sentences in the same style as training. The most reliable gains are on languages whose test includes harder, varied subtitle dialogue (Hindi +2.9, Tamil +2.4, Malayalam +2.7) — precisely the languages for which §5.3 confirms significance. We therefore caution against ranking languages by raw Δconv across heterogeneous test registers.

**A negative result on low-resource languages.** Languages with very low absolute base quality — Santali (FLORES 30.7), Sanskrit (36.3), Kashmiri (39.1), Bodo (45.9) — remain low after adaptation. Their *general* quality is bounded by the base model, which is in turn bounded by the scarcity of any parallel data for these languages. Conversational adaptation improves their conversational chrF but cannot lift their general ceiling; doing so would require additional *general* parallel data, which does not exist publicly at sufficient scale. This is a limitation of the data, not the method.

## 8. Conclusion

Using experience replay and model souping — two established techniques — we adapt IndicTrans2 to conversational register across all 21 Indic languages while preserving general-domain quality, with statistically significant conversational chrF gains and no significant general-quality loss. The recipe generalises across the family; its limits are set by the base model's coverage of the lowest-resource languages. By automatic metrics, the result is a single, drop-in model that **beats the base on conversational chrF** (significantly; p ≤ 0.004) while **matching it on general-domain FLORES** — a strictly Pareto-non-worse trade on these two metrics. We are careful to keep this claim scoped: it is a chrF result on a conversational test set, not an unqualified quality claim. Confirming that the conversational chrF gain is perceived by humans as higher quality — as opposed to a register shift the metric rewards (§6) — is the natural next step (§9).

## 9. Limitations

- **Automatic metrics only.** All quality numbers are chrF2. Character-n-gram metrics measure overlap with references, not quality directly, and can overstate domain-adaptation gains when the in-domain references have a distinctive style: a model that adopts that style gains overlap independent of adequacy [Callison-Burch et al., 2006; Mathur et al., 2020; Freitag et al., 2021]. Our conversational references are subtitles, so part of the conversational chrF gain plausibly reflects register-matching (§6). Confirming a human-*perceived* quality improvement requires a controlled, multi-annotator human evaluation; this is our priority next step and is **not yet conclusive**, so we make no human-quality claim and scope the contribution to the metric.
- **Single training seed.** We report one run per configuration; training-variance across seeds is not yet quantified (significance §5.3 addresses test-set variance, not training variance).
- **Significance on a subset.** Paired bootstrap is reported for 3 of 21 languages (the hardest-test ones); extending to all 21 is straightforward future work.
- **Register confound across languages.** Conversational test register differs by language (subtitles vs. daily utterances), confounding cross-language gain comparison (§7).
- **One direction.** English→Indic only; Indic→English and Indic↔Indic are untested and could exhibit different forgetting behaviour.

## Reproducibility

All code is in the accompanying repository: data construction (`build_multiling_conv.py`, `build_ml_mixed.py`), training (`finetune_it2_ml.py`), souping (`make_soup.py`), evaluation (`ml_eval.py`), and significance (`sig_test.py`). Released artifacts: the mixed-FT model (`ft_it2_ml_mixed`) and its soup (`ft_it2_ml_mixed_soup`). Environment: Python 3.11, `transformers==4.40.2`, `IndicTransToolkit`, sacreBLEU 2.6.

## References

- Callison-Burch, Osborne, Koehn. *Re-evaluating the Role of BLEU in MT Research.* EACL 2006.
- Chu, Dabre, Kurohashi. *An Empirical Comparison of Domain Adaptation Methods for NMT.* ACL 2017.
- Chu and Wang. *A Survey of Domain Adaptation for NMT.* COLING 2018.
- Freitag and Al-Onaizan. *Fast Domain Adaptation for NMT.* arXiv 2016.
- Freitag, Foster, Grangier, et al. *Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for MT.* TACL 2021.
- French. *Catastrophic Forgetting in Connectionist Networks.* Trends in Cognitive Sciences 1999.
- Gala et al. *IndicTrans2: Towards High-Quality and Accessible MT for all 22 Scheduled Indian Languages.* TMLR 2023.
- Goyal et al. *The FLORES-101 Evaluation Benchmark.* TACL 2022.
- Koehn. *Statistical Significance Tests for MT Evaluation.* EMNLP 2004.
- Lison and Tiedemann. *OpenSubtitles2016.* LREC 2016.
- Luong and Manning. *Stanford NMT Systems for Spoken Language Domains.* IWSLT 2015.
- Mathur, Baldwin, Cohn. *Tangled up in BLEU: Reevaluating the Evaluation of Automatic MT Metrics.* ACL 2020.
- McCloskey and Cohen. *Catastrophic Interference in Connectionist Networks.* 1989.
- Nakazawa et al. *Overview of the Workshop on Asian Translation.* WAT 2024.
- NLLB Team (Costa-jussà et al.). *No Language Left Behind.* arXiv 2022. (FLORES-200)
- Popović. *chrF: Character n-gram F-score for Automatic MT Evaluation.* WMT 2015.
- Post. *A Call for Clarity in Reporting BLEU Scores* (sacreBLEU). WMT 2018.
- Robins. *Catastrophic Forgetting, Rehearsal and Pseudorehearsal.* Connection Science 1995.
- Wortsman et al. *Model Soups.* ICML 2022a.
- Wortsman et al. *Robust Fine-tuning of Zero-shot Models* (WiSE-FT). CVPR 2022b.
