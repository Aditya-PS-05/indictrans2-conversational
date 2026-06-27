#!/usr/bin/env python3
"""
multi_model_judge.py  — judge the 30 divergent Hindi pairs with EVERY available
OpenAI chat model, on a conversational basis, with POSITION-BIAS control.

For each model x each pair: ask twice (original A/B order + swapped). A vote only
counts as decisive if the model is CONSISTENT across both orders; if it flips when
you swap A/B, that's position bias -> counted as a tie. Then un-blind vs the key
and print soup/base/tie + a two-sided sign-test p per model, plus an aggregate.

Setup:
  pip install openai
  export OPENAI_API_KEY=sk-...
  python multi_model_judge.py            # all auto-discovered models
  python multi_model_judge.py gpt-4o gpt-4.1 o4-mini   # or name them explicitly
Writes: per-model verdicts to mm_<model>.csv and a summary to mm_summary.csv.
"""
import csv
import math
import sys

from openai import OpenAI

client = OpenAI()

PROMPT = """You are a native Hindi speaker judging two Hindi translations of an \
English sentence for use in CASUAL, EVERYDAY CONVERSATION (chat / spoken dialogue) \
— NOT formal or textbook Hindi.

English: {en}
Translation A: {a}
Translation B: {b}

Which translation sounds more natural to a native speaker in casual conversation? \
Favor idiomatic, spoken phrasing (casual pronouns, natural tag-questions, natural \
word order). BUT a genuine error — wrong meaning, broken grammar, a mistranslation, \
or a glitch — should lose regardless of how casual it sounds.

Answer with EXACTLY one token: A, B, or Tie."""

# ---- which models to judge with -------------------------------------------------
SKIP = ("audio", "realtime", "transcribe", "tts", "whisper", "image", "dall-e",
        "embedding", "moderation", "search", "instruct", "codex", "computer")
def discover():
    out = []
    for m in client.models.list().data:
        mid = m.id
        if any(s in mid for s in SKIP):
            continue
        if mid.startswith("gpt-") or mid.startswith("o1") or mid.startswith("o3") \
           or mid.startswith("o4") or mid.startswith("o5"):
            out.append(mid)
    # collapse dated snapshots (keep base alias if present, else keep all)
    return sorted(set(out))

def ask(model, en, a, b):
    """Return 'A','B', or 'tie' for one ordering. Robust to models that reject
    temperature / use max_completion_tokens (o-series, gpt-5 family)."""
    msg = PROMPT.format(en=en, a=a, b=b)
    kwargs = dict(model=model, messages=[{"role": "user", "content": msg}])
    last = RuntimeError("no attempt ran")
    for attempt in (dict(temperature=0), dict()):           # try temp=0, then default
        try:
            resp = client.chat.completions.create(**kwargs, **attempt)
            ans = (resp.choices[0].message.content or "").strip().lower()
            return "A" if ans.startswith("a") else "B" if ans.startswith("b") else "tie"
        except Exception as e:
            last = e
    raise last

def judge_pair(model, en, ta, tb):
    """Counterbalanced: ask original + swapped; decisive only if consistent."""
    v1 = ask(model, en, ta, tb)                 # A=ta, B=tb
    v2 = ask(model, en, tb, ta)                 # A=tb, B=ta  (swapped)
    # map v2 back to the original A/B frame
    v2_orig = "A" if v2 == "B" else "B" if v2 == "A" else "tie"
    if v1 == v2_orig and v1 in ("A", "B"):
        return v1                               # consistent -> trust it
    return "tie"                                # flipped/inconsistent -> position bias

def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    obs = math.comb(n, k) * p**k * (1 - p)**(n - k)
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
                        for i in range(n + 1)
                        if math.comb(n, i) * p**i * (1 - p)**(n - i) <= obs + 1e-12))

# ---- run ------------------------------------------------------------------------
rows = list(csv.DictReader(open("eval_data/eval_pref_hin.csv", encoding="utf-8")))
key = {r["id"]: (r["A_system"], r["B_system"])
       for r in csv.DictReader(open("eval_data/eval_pref_key_hin.csv", encoding="utf-8"))}

models = sys.argv[1:] or discover()
print(f"judging {len(rows)} pairs (counterbalanced) with {len(models)} models:")
print("  " + ", ".join(models) + "\n")

summary = []
agg_soup = agg_base = 0
for model in models:
    soup = base = tie = err = 0
    per = []
    for r in rows:
        try:
            v = judge_pair(model, r["english"], r["translation_A"], r["translation_B"])
        except Exception as e:
            err += 1; per.append([r["id"], f"ERROR:{type(e).__name__}"]); continue
        a_sys, b_sys = key[r["id"]]
        if v == "A":
            win = a_sys
        elif v == "B":
            win = b_sys
        else:
            tie += 1; per.append([r["id"], "Tie"]); continue
        per.append([r["id"], f"{v} ({win})"])
        if win == "soup":
            soup += 1
        else:
            base += 1
    with open(f"eval_data/mm_{model.replace('/','_')}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "verdict"]); w.writerows(per)
    dec = soup + base
    p = binom_two_sided(soup, dec)
    summary.append((model, soup, base, tie, err, dec, p))
    agg_soup += soup; agg_base += base
    flag = "  *sig*" if p < 0.05 else ""
    print(f"{model:28s} soup {soup:>2}  base {base:>2}  tie {tie:>2}"
          f"{('  err '+str(err)) if err else '':>8}   "
          f"win {('%.0f%%'%(100*soup/dec)) if dec else 'n/a':>4}  p={p:.3f}{flag}")

# ---- aggregate ------------------------------------------------------------------
with open("eval_data/mm_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "soup", "base", "tie", "err", "decisive", "sign_p"])
    w.writerows(summary)

dec = agg_soup + agg_base
print("\n" + "=" * 60)
print(f"AGGREGATE (all models pooled): soup {agg_soup}  base {agg_base}  "
      f"win {('%.0f%%'%(100*agg_soup/dec)) if dec else 'n/a'}  "
      f"p={binom_two_sided(agg_soup, dec):.4f}")
lean_soup = sum(1 for _, s_, b_, *_ in summary if s_ > b_)
lean_base = sum(1 for _, s_, b_, *_ in summary if b_ > s_)
print(f"models leaning soup: {lean_soup}   leaning base: {lean_base}   "
      f"even: {len(summary)-lean_soup-lean_base}")
print("wrote mm_summary.csv + per-model mm_*.csv")
print("\nNOTE: OpenAI models share priors — treat this as ONE correlated family "
      "opinion, not N independent judges.")
