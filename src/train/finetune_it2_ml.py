#!/usr/bin/env python3
"""
finetune_it2_ml.py  (Python-3.11 it2env311)
Multilingual fine-tune of IndicTrans2-en-indic-1B on conversational EN->Indic
across 21 languages. Each pair is preprocessed with ITS OWN target language tag
(grouped by lang), unlike the Hindi-only finetune_it2.py.
Records: {"translation":{"en","tgt"}, "lang":"<flores_code>"}.
"""
import argparse
import json
from collections import defaultdict

CKPT = "ai4bharat/indictrans2-en-indic-1B"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="conv_splits/ml_train.jsonl")
    p.add_argument("--dev", default="conv_splits/ml_dev.jsonl")
    p.add_argument("--out", default="ft_it2_ml")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load(path, limit=0):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append((r["translation"]["en"], r["translation"]["tgt"], r["lang"]))
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    a = parse_args()
    import numpy as np
    import sacrebleu
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              DataCollatorForSeq2Seq)
    from IndicTransToolkit import IndicProcessor

    ip = IndicProcessor(inference=False)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT, trust_remote_code=True)

    def build(path, limit=0):
        rows = load(path, limit)
        groups = defaultdict(list)          # lang -> [(idx, en, tgt)]
        for i, (en, tgt, lang) in enumerate(rows):
            groups[lang].append((i, en, tgt))
        src_out = [None] * len(rows)
        tgt_out = [None] * len(rows)
        for lang, items in groups.items():
            idxs = [x[0] for x in items]
            ens = [x[1] for x in items]
            tgts = [x[2] for x in items]
            s = ip.preprocess_batch(ens, src_lang="eng_Latn", tgt_lang=lang)
            t = ip.preprocess_batch(tgts, src_lang=lang, tgt_lang="eng_Latn",
                                    is_target=True)
            for j, idx in enumerate(idxs):
                src_out[idx] = s[j]
                tgt_out[idx] = t[j]
        ds = Dataset.from_dict({"src": src_out, "tgt": tgt_out})

        def enc(b):
            return tok(b["src"], text_target=b["tgt"],
                       truncation=True, max_length=a.max_len)
        return ds.map(enc, batched=True, remove_columns=["src", "tgt"])

    train_ds = build(a.train, a.limit)
    dev_ds = build(a.dev)
    print(f"[info] train={len(train_ds):,}  dev={len(dev_ds):,}")

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

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
        output_dir=a.out, num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr,
        warmup_ratio=0.05, weight_decay=0.01, bf16=True, optim="adafactor",
        evaluation_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="chrf",
        greater_is_better=True, predict_with_generate=True,
        generation_max_length=a.max_len, generation_num_beams=1,
        logging_steps=100, report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator, compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(a.out)
    tok.save_pretrained(a.out)
    print(f"[done] saved -> {a.out}")
    # (no final trainer.evaluate() — redundant with epoch evals; avoids end RAM spike)


if __name__ == "__main__":
    main()
