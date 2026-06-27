#!/usr/bin/env python3
"""
dual_criterion_judge.py — judge the 30 real base/soup pairs with the TOP-10 OpenAI
models under BOTH GPT-authored criteria (neutral vs conversational), with the same
position-bias control (judge each pair twice, A/B swapped; flips -> tie). Un-blind
vs eval_pref_key_hin.csv and print a side-by-side table + per-criterion aggregate.

Setup:  export OPENAI_API_KEY=sk-... ; python make_eval_sheets.py   (first, for criteria)
Run:    python dual_criterion_judge.py            # default top-10
        python dual_criterion_judge.py gpt-5.5 o3 # or name models explicitly
Writes: dual_summary.csv
"""
import csv
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

WORKERS = 24      # concurrent API calls; raise if you don't hit rate limits

client = OpenAI()

# top-10 strong, reasonably-priced chat models (override via argv). Pro/o1-pro
# excluded by default — very expensive and the trend is already model-robust.
TOP10 = ["gpt-5.5", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5",
         "gpt-4.1", "gpt-4o", "gpt-4-turbo", "o3", "o4-mini"]


def ask(model, prompt):
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}])
    last = RuntimeError("no attempt ran")
    for attempt in (dict(temperature=0), dict()):
        try:
            resp = client.chat.completions.create(**kwargs, **attempt)
            ans = (resp.choices[0].message.content or "").strip().lower()
            return "A" if ans.startswith("a") else "B" if ans.startswith("b") else "tie"
        except Exception as e:
            last = e
    raise last


def judge_pair(model, crit, en, ta, tb):
    """Counterbalanced under a given criterion template; flip -> tie."""
    v1 = ask(model, crit.format(en=en, a=ta, b=tb))
    v2 = ask(model, crit.format(en=en, a=tb, b=ta))
    v2o = "A" if v2 == "B" else "B" if v2 == "A" else "tie"
    return v1 if (v1 == v2o and v1 in ("A", "B")) else "tie"


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return 1.0
    obs = math.comb(n, k) * p**k * (1 - p)**(n - k)
    return min(1.0, sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
                        for i in range(n + 1)
                        if math.comb(n, i) * p**i * (1 - p)**(n - i) <= obs + 1e-12))


rows = list(csv.DictReader(open("eval_pref_hin.csv", encoding="utf-8")))
key = {r["id"]: (r["A_system"], r["B_system"])
       for r in csv.DictReader(open("eval_pref_key_hin.csv", encoding="utf-8"))}
crit = json.load(open("eval_criteria.json", encoding="utf-8"))
models = sys.argv[1:] or TOP10

print(f"{len(rows)} pairs, counterbalanced, {len(models)} models, 2 criteria, "
      f"{WORKERS} workers\n")

# build every judging unit (model x criterion x pair) and run them concurrently
units = [(m, cname, r) for m in models
         for cname in ("neutral", "conversational") for r in rows]

def run_unit(u):
    m, cname, r = u
    try:
        v = judge_pair(m, crit[cname], r["english"],
                       r["translation_A"], r["translation_B"])
    except Exception:
        return (m, cname, "err")
    a_sys, b_sys = key[r["id"]]
    win = a_sys if v == "A" else b_sys if v == "B" else None
    return (m, cname, win)          # win in {"base","soup",None,"err"}

tally = {(m, c): {"soup": 0, "base": 0, "tie": 0, "err": 0}
         for m in models for c in ("neutral", "conversational")}
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for m, cname, win in ex.map(run_unit, units):
        d = tally[(m, cname)]
        d[win if win in ("soup", "base", "err") else "tie"] += 1

print(f"{'model':14s} | {'NEUTRAL  soup/base/tie  win   p':33s} | "
      f"{'CONVERSATIONAL soup/base/tie win  p'}")
print("-" * 92)

summary = []
agg = {"neutral": [0, 0], "conversational": [0, 0]}
for m in models:
    cells = {}
    for cname in ("neutral", "conversational"):
        d = tally[(m, cname)]
        s, b, t, e = d["soup"], d["base"], d["tie"], d["err"]
        dec = s + b
        p = binom_two_sided(s, dec)
        cells[cname] = (s, b, t, e, dec, p)
        agg[cname][0] += s; agg[cname][1] += b
        summary.append([m, cname, s, b, t, e, p])
    def fmt(c):
        s, b, t, e, dec, p = c
        wr = f"{100*s/dec:.0f}%" if dec else "n/a"
        return f"{s:>2}/{b:>2}/{t:>2}  {wr:>4}  p={p:.3f}" + ("" if e == 0 else f" err{e}")
    print(f"{m:14s} | {fmt(cells['neutral']):33s} | {fmt(cells['conversational'])}")

with open("dual_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "criterion", "soup", "base", "tie", "err", "sign_p"])
    w.writerows(summary)

print("\n" + "=" * 60)
for cname in ("neutral", "conversational"):
    s, b = agg[cname]
    dec = s + b
    wr = f"{100*s/dec:.0f}%" if dec else "n/a"
    print(f"POOLED {cname:14s}: soup {s}  base {b}  win {wr}  "
          f"p={binom_two_sided(s, dec):.4f}")
print("wrote dual_summary.csv")
print("\nNOTE: OpenAI models share priors -> one correlated family, not 10 "
      "independent judges. Read neutral vs conversational as the same family "
      "answering under two framings.")
