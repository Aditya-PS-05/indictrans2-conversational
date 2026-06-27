#!/bin/bash
cd ~/eng-hindi-model
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
source ~/it2env311/bin/activate
rm -f multilang_results.log
for tgt in ben_Beng tam_Taml mar_Deva; do
  for m in ai4bharat/indictrans2-en-indic-1B ft_it2_soup_final; do
    echo "##### $m  EN->$tgt #####" >> multilang_results.log
    python multilang_check.py "$m" "$tgt" 500 >> multilang_results.log 2>&1
  done
done
echo ALL_DONE >> multilang_results.log
