#!/usr/bin/env python3
"""
benchmark_translate.py
Benchmark EN->HI models on the held-out conversational test set.
Reports chrF2 (primary metric for Indic) and BLEU via sacrebleu.

Models:
  it2-1b   ai4bharat/indictrans2-en-indic-1B        (best quality)
  it2-200m ai4bharat/indictrans2-en-indic-dist-200M (fast)
  nllb     facebook/nllb-200-distilled-600M

Run:
  python benchmark_translate.py --models it2-1b,nllb --limit 1000
  python benchmark_translate.py --models it2-200m            # quick
"""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="conv_splits/test.jsonl")
    p.add_argument("--models", default="it2-1b,nllb")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_test(path, limit):
    en, hi = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)["translation"]
        en.append(t["en"]); hi.append(t["hi"])
        if len(en) >= limit:
            break
    return en, hi


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run_indictrans2(model_id, en, bs, device, tok_id=None):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # compat shims for NEWER transformers only. On transformers 4.40 (the
    # py3.11 IndicTrans2 env) the real modules exist -> do NOT shim, or we'd
    # break the genuine transformers.onnx. Each shim applies only if the real
    # import fails.
    try:
        from transformers.tokenization_utils import PreTrainedTokenizerBase  # noqa
    except Exception:
        import transformers.tokenization_utils as _tu
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTB
        _tu.PreTrainedTokenizerBase = _PTB
    try:
        import transformers.onnx  # noqa: real module in transformers 4.40
    except Exception:
        import sys as _sys, types as _types
        _onnx = _types.ModuleType("transformers.onnx")
        _onnx_utils = _types.ModuleType("transformers.onnx.utils")

        class _OnnxStub:
            pass

        _onnx.OnnxConfig = _OnnxStub
        _onnx.OnnxSeq2SeqConfigWithPast = _OnnxStub
        _onnx_utils.compute_effective_axis_dimension = lambda *a, **k: 0
        _onnx.utils = _onnx_utils
        _sys.modules["transformers.onnx"] = _onnx
        _sys.modules["transformers.onnx.utils"] = _onnx_utils
    from IndicTransToolkit import IndicProcessor
    ip = IndicProcessor(inference=True)
    # IT2's custom tokenizer has a save/load round-trip bug (duplicate
    # src_vocab_file). The tokenizer is unchanged by fine-tuning, so load it
    # from the original checkpoint (tok_id) when scoring a fine-tuned dir.
    tok = AutoTokenizer.from_pretrained(tok_id or model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    out = []
    for chunk in batched(en, bs):
        pre = ip.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang="hin_Deva")
        enc = tok(pre, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        with torch.inference_mode():
            g = model.generate(**enc, max_length=256, num_beams=5, num_return_sequences=1)
        dec = tok.batch_decode(g, skip_special_tokens=True)
        out.extend(ip.postprocess_batch(dec, lang="hin_Deva"))
    return out


def run_nllb(en, bs, device, model_id="facebook/nllb-200-distilled-600M"):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    bos = tok.convert_tokens_to_ids("hin_Deva")
    out = []
    for chunk in batched(en, bs):
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        with torch.inference_mode():
            g = model.generate(**enc, forced_bos_token_id=bos, max_length=256, num_beams=5)
        out.extend(tok.batch_decode(g, skip_special_tokens=True))
    return out


def main():
    a = parse_args()
    import torch
    import sacrebleu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    en, ref = load_test(a.test, a.limit)
    print(f"[info] test pairs: {len(en)}  device: {device}")

    results = {}
    for m in a.models.split(","):
        m = m.strip()
        print(f"\n[info] running {m} ...")
        try:
            if m == "it2-1b":
                hyp = run_indictrans2("ai4bharat/indictrans2-en-indic-1B", en, a.batch_size, device)
            elif m == "it2-200m":
                hyp = run_indictrans2("ai4bharat/indictrans2-en-indic-dist-200M", en, a.batch_size, device)
            elif m == "nllb":
                hyp = run_nllb(en, a.batch_size, device)
            elif m == "nllb-1.3b":
                hyp = run_nllb(en, a.batch_size, device, "facebook/nllb-200-distilled-1.3B")
            elif m == "nllb-3.3b":
                hyp = run_nllb(en, a.batch_size, device, "facebook/nllb-200-3.3B")
            elif m.startswith("ft:"):                 # fine-tuned local NLLB dir
                hyp = run_nllb(en, a.batch_size, device, m[3:])
            elif m.startswith("ftit2:"):              # fine-tuned local IT2 dir
                hyp = run_indictrans2(m[6:], en, a.batch_size, device,
                                      tok_id="ai4bharat/indictrans2-en-indic-1B")
            else:
                print(f"  unknown model {m}; skip"); continue
        except Exception as e:
            print(f"  [skip] {m} failed: {str(e)[:160]}")
            continue
        chrf = sacrebleu.corpus_chrf(hyp, [ref]).score
        bleu = sacrebleu.corpus_bleu(hyp, [ref]).score
        results[m] = (chrf, bleu)
        print(f"  {m}: chrF2={chrf:.2f}  BLEU={bleu:.2f}")
        for i in range(min(3, len(hyp))):
            print(f"    EN : {en[i]}")
            print(f"    REF: {ref[i]}")
            print(f"    HYP: {hyp[i]}")
        # free GPU between models so multiple can run sequentially without OOM
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\n===== SUMMARY (EN->HI test) =====")
    print(f"{'model':12s} {'chrF2':>8s} {'BLEU':>8s}")
    for m, (c, b) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{m:12s} {c:>8.2f} {b:>8.2f}")


if __name__ == "__main__":
    main()
