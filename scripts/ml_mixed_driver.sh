#!/bin/bash
# Autonomous: wait for mixed-ML training, soup it, benchmark FT + soup across langs.
cd ~/eng-hindi-model
while true; do
  if grep -qa "saved -> ft_it2_ml_mixed" ft_it2_ml_mixed.log 2>/dev/null; then break; fi
  if ! tmux has-session -t mlmix 2>/dev/null; then break; fi
  sleep 60
done
sleep 10
pkill -f finetune_it2_ml 2>/dev/null; sleep 12

export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate

python src/train/make_soup.py 0.6 ft_it2_ml_mixed_soup ft_it2_ml_mixed > ml_mixed_soup_build.log 2>&1
sleep 5

LANGS="hin_Deva,tam_Taml,ben_Beng,mar_Deva,mal_Mlym"
echo "===== MIXED FT =====" > ml_mixed_results.log
python src/eval/ml_eval.py ft_it2_ml_mixed "$LANGS" >> ml_mixed_results.log 2>&1
echo "===== MIXED SOUP alpha=0.6 =====" >> ml_mixed_results.log
python src/eval/ml_eval.py ft_it2_ml_mixed_soup "$LANGS" >> ml_mixed_results.log 2>&1
echo "ALL_DONE" >> ml_mixed_results.log
