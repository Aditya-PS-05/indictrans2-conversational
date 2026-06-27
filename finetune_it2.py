#!/usr/bin/env python3
"""
finetune_it2.py  (run in the Python-3.11 it2env311 venv)
Fine-tune IndicTrans2-en-indic-1B (EN->HI) on the conversational train split.

IndicTrans2 needs IndicProcessor preprocessing (NOT plain tokenization):
  * source: tagged + normalized  -> "eng_Latn hin_Deva How are you ?"
  * target: normalized, untagged  -> "तुम कैसे हो ?"  (is_target=True)
Then tokenizer(src, text_target=tgt) encodes target with IT2's target vocab.
Eval decodes + ip.postprocess_batch before scoring chrF/BLEU.

Run:
  source ~/it2env311/bin/activate
  python finetune_it2.py --train conv_splits/train.jsonl --dev conv_splits/dev.jsonl \
      --out ft_it2_1b --epochs 3 --bs 8 --grad-accum 4
"""
import argparse
import json

CKPT = "ai4bharat/indictrans2-en-indic-1B"
SRC, TGT = "eng_Latn", "hin_Deva"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="conv_splits/train.jsonl")
    p.add_argument("--dev", default="conv_splits/dev.jsonl")
    p.add_argument("--out", default="ft_it2_1b")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="enable gradient checkpointing (only if low on VRAM)")
    p.add_argument("--limit-train", type=int, default=0)   # >0 for smoke test
    return p.parse_args()


def load_jsonl(path, limit=0):
    en, hi = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t = json.loads(line)["translation"]
        en.append(t["en"]); hi.append(t["hi"])
        if limit and len(en) >= limit:
            break
    return en, hi


def main():
    a = parse_args()
    import numpy as np
    import sacrebleu
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              DataCollatorForSeq2Seq)
    # IndicDataCollator needs a newer-transformers helper absent in 4.40; the
    # stock seq2seq collator does the same padding + label/decoder prep here.
    from IndicTransToolkit import IndicProcessor

    ip = IndicProcessor(inference=False)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT, trust_remote_code=True)
    if a.grad_ckpt:                       # only when VRAM-constrained; it's slow
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    def build(path, limit=0):
        en, hi = load_jsonl(path, limit)
        src = ip.preprocess_batch(en, src_lang=SRC, tgt_lang=TGT)
        tgt = ip.preprocess_batch(hi, src_lang=TGT, tgt_lang=SRC, is_target=True)
        ds = Dataset.from_dict({"src": src, "tgt": tgt})

        def enc(b):
            return tok(b["src"], text_target=b["tgt"],
                       truncation=True, max_length=a.max_len)
        return ds.map(enc, batched=True, remove_columns=["src", "tgt"])

    train_ds = build(a.train, a.limit_train)
    dev_ds = build(a.dev)
    print(f"[info] train={len(train_ds):,}  dev={len(dev_ds):,}")

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        # NOTE: compare decoded strings directly (no ip.postprocess here — the
        # training-time IndicProcessor is in inference=False mode and its
        # postprocess stalls). Both sides are decoded identically, so chrF is a
        # valid relative metric for epoch selection. The authoritative,
        # postprocessed score comes from benchmark_translate.py afterwards.
        dp = tok.batch_decode(preds, skip_special_tokens=True)
        dl = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": sacrebleu.corpus_chrf(dp, [dl]).score,
                "bleu": sacrebleu.corpus_bleu(dp, [dl]).score}

    args = Seq2SeqTrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        optim="adafactor",               # low-memory optimizer for the 1B
        evaluation_strategy="epoch",     # transformers 4.40 arg name

        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=a.max_len,
        generation_num_beams=1,
        logging_steps=100,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator, compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"[done] saved -> {a.out}")
    print("[info] final dev metrics:", trainer.evaluate())


if __name__ == "__main__":
    main()
