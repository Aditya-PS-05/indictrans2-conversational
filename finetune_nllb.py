#!/usr/bin/env python3
"""
finetune_nllb.py
Fine-tune NLLB (EN->HI) on the conversational train split, eval chrF/BLEU on dev.
Clean stack: works with the box's current transformers, no shims.

Run:
  python finetune_nllb.py --base facebook/nllb-200-distilled-600M \
      --train conv_splits/train.jsonl --dev conv_splits/dev.jsonl \
      --out ft_nllb600 --epochs 3 --bs 16

Then benchmark the result on the test set:
  python benchmark_translate.py --models ft:ft_nllb600 --limit 600
"""
import argparse
import json

SRC, TGT = "eng_Latn", "hin_Deva"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--train", default="conv_splits/train.jsonl")
    p.add_argument("--dev", default="conv_splits/dev.jsonl")
    p.add_argument("--out", default="ft_nllb600")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=128)
    return p.parse_args()


def load_jsonl(path):
    en, hi = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            t = json.loads(line)["translation"]
            en.append(t["en"]); hi.append(t["hi"])
    return {"en": en, "hi": hi}


def main():
    a = parse_args()
    import numpy as np
    import sacrebleu
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    tok = AutoTokenizer.from_pretrained(a.base, src_lang=SRC, tgt_lang=TGT)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.base)
    hin_id = tok.convert_tokens_to_ids(TGT)
    # newer transformers: control generation via generation_config ONLY
    # (setting model.config.forced_bos_token_id raises at generate time).
    model.generation_config.forced_bos_token_id = hin_id
    if getattr(model.config, "forced_bos_token_id", None) is not None:
        model.config.forced_bos_token_id = None

    def preprocess(batch):
        enc = tok(batch["en"], text_target=batch["hi"],
                  max_length=a.max_len, truncation=True)
        return enc

    train_ds = Dataset.from_dict(load_jsonl(a.train)).map(
        preprocess, batched=True, remove_columns=["en", "hi"])
    dev_ds = Dataset.from_dict(load_jsonl(a.dev)).map(
        preprocess, batched=True, remove_columns=["en", "hi"])
    print(f"[info] train={len(train_ds):,}  dev={len(dev_ds):,}")

    collator = DataCollatorForSeq2Seq(tok, model=model)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
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
        bf16=True,                       # Ampere (A10G) supports bf16; more stable
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=a.max_len,
        generation_num_beams=1,          # fast eval; final eval uses beams
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
    print(f"[done] saved fine-tuned model -> {a.out}")
    print("[info] final dev metrics:", trainer.evaluate())


if __name__ == "__main__":
    main()
