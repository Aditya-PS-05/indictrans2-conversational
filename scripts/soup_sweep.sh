#!/bin/bash
# Model-soup sweep: for each alpha, build the averaged model, benchmark on
# conv-test + FLORES, log, then delete it (recreate the winner later).
cd ~/eng-hindi-model
export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate
rm -f soup_results.log
for a in 0.55 0.6 0.65; do
  echo "########## alpha=$a ##########" >> soup_results.log
  python src/train/make_soup.py "$a" "soup_$a" >> soup_results.log 2>&1
  echo "=== alpha=$a CONV ===" >> soup_results.log
  python src/eval/benchmark_translate.py --test conv_splits/test.jsonl \
    --models "ftit2:soup_$a" --limit 600 --batch-size 16 >> soup_results.log 2>&1
  echo "=== alpha=$a FLORES ===" >> soup_results.log
  python src/eval/benchmark_translate.py --test flores_test.jsonl \
    --models "ftit2:soup_$a" --limit 997 --batch-size 16 >> soup_results.log 2>&1
  rm -rf "soup_$a"
  sleep 3
done
echo "ALL DONE" >> soup_results.log
