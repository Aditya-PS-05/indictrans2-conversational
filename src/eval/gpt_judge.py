#!/usr/bin/env python3
"""
gpt_judge.py  — second, independent LLM judge (OpenAI) on conversational basis.
Reads the BLIND pref sheet (eval_pref_hin.csv: A/B already shuffled), asks GPT
which Hindi translation is more natural for CASUAL conversation, writes choices
to gpt_pref_hin.csv. Then: python analyze_pref.py gpt_pref_hin.csv eval_pref_key_hin.csv

Setup (on your laptop):
  pip install openai
  export OPENAI_API_KEY=sk-...
  python gpt_judge.py
Cost: ~30 calls, a few cents.
"""
import csv

from openai import OpenAI

MODEL = "gpt-4o"          # strong multilingual; change to gpt-4.1 if you prefer
client = OpenAI()         # reads OPENAI_API_KEY from env

PROMPT = """You are a native Hindi speaker judging two Hindi translations of an \
English sentence for use in CASUAL, EVERYDAY CONVERSATION (chat / spoken dialogue) \
— NOT formal or textbook Hindi.

English: {en}
Translation A: {a}
Translation B: {b}

Which translation sounds more natural to a native speaker in casual conversation? \
Favor idiomatic, spoken phrasing (e.g. casual pronouns, natural tag-questions, \
natural word order). BUT a genuine error — wrong meaning, broken grammar, a \
mistranslation, or a glitch — should lose regardless of how casual it sounds.

Answer with EXACTLY one token: A, B, or Tie."""

rows = list(csv.DictReader(open("eval_data/eval_pref_hin.csv", encoding="utf-8")))
out = []
for r in rows:
    msg = PROMPT.format(en=r["english"], a=r["translation_A"], b=r["translation_B"])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": msg}])
    ans = resp.choices[0].message.content.strip().lower()
    choice = "A is better" if ans.startswith("a") else \
             "B is better" if ans.startswith("b") else "Tie"
    out.append([r["id"], choice])
    print(f"{r['id']:>2}: {ans[:8]:8s} -> {choice}")

with open("eval_data/gpt_pref_hin.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "which_is_better"])
    w.writerows(out)
print(f"\nwrote gpt_pref_hin.csv ({len(out)} judgments)")
print("now run: python analyze_pref.py gpt_pref_hin.csv eval_pref_key_hin.csv")
