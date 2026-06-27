#!/bin/bash
cd ~/eng-hindi-model
export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate
LANGS="asm_Beng,ben_Beng,brx_Deva,doi_Deva,gom_Deva,guj_Gujr,hin_Deva,kan_Knda,kas_Arab,mai_Deva,mal_Mlym,mar_Deva,mni_Mtei,npi_Deva,ory_Orya,pan_Guru,san_Deva,sat_Olck,tam_Taml,tel_Telu,urd_Arab"
echo "===== BASE =====" > all21_results.log
python src/eval/ml_eval.py ai4bharat/indictrans2-en-indic-1B "$LANGS" 200 >> all21_results.log 2>&1
echo "===== MIXED SOUP =====" >> all21_results.log
python src/eval/ml_eval.py ft_it2_ml_mixed_soup "$LANGS" 200 >> all21_results.log 2>&1
echo "ALL_DONE" >> all21_results.log
