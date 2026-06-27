#!/bin/bash
# Autonomous: wait for ML training, soup it, benchmark base/FT/soup across langs.
cd ~/eng-hindi-model
while true; do
  if grep -qa "saved -> ft_it2_ml" ft_it2_ml.log 2>/dev/null; then break; fi
  if ! tmux has-session -t mltrain 2>/dev/null; then break; fi
  sleep 60
done
sleep 10
pkill -f finetune_it2_ml 2>/dev/null; sleep 12   # free GPU/RAM before next steps

export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate

# memory-safe soup at alpha=0.6 (Hindi-proven sweet spot), from the ML fine-tune
python make_soup.py 0.6 ft_it2_ml_soup ft_it2_ml > ml_soup_build.log 2>&1
sleep 5

# 5 representative langs: hi (rich), ta/ml (OpenSubs), bn/mr (BPCC-daily only)
LANGS="hin_Deva,tam_Taml,ben_Beng,mar_Deva,mal_Mlym"
echo "===== BASE IndicTrans2 =====" > ml_results.log
python ml_eval.py ai4bharat/indictrans2-en-indic-1B "$LANGS" >> ml_results.log 2>&1
echo "===== FT (pure multilingual) =====" >> ml_results.log
python ml_eval.py ft_it2_ml "$LANGS" >> ml_results.log 2>&1
echo "===== SOUP alpha=0.6 =====" >> ml_results.log
python ml_eval.py ft_it2_ml_soup "$LANGS" >> ml_results.log 2>&1
echo "ALL_DONE" >> ml_results.log
