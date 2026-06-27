#!/bin/bash
# Overnight chain: after the first mixed run + benchmarks finish, train a gentler
# (lr 1e-5) variant and benchmark it on conv-test + FLORES. Known-good code only.
cd ~/eng-hindi-model

# 1) wait for the first mixed run's benchmarks to fully complete
while true; do
  if grep -qa "ALL DONE" mixed_results.log 2>/dev/null; then break; fi
  sleep 60
done
sleep 10
pkill -f finetune_it2 2>/dev/null; sleep 10   # ensure GPU is free

export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate

# 2) gentler variant: same mixed data, lr 1e-5
python finetune_it2.py \
  --train conv_splits/mixed_train.jsonl --dev conv_splits/mixed_dev.jsonl \
  --out ft_it2_mixed_lr1e5 --epochs 2 --bs 16 --grad-accum 2 --lr 1e-5 \
  > ft_it2_mixed_lr1e5.log 2>&1

pkill -f finetune_it2 2>/dev/null; sleep 12   # free GPU before benchmarking

# 3) benchmark on both test sets
{
  echo "=== dev evals (training) ==="
  grep -aoE "\{.*chrf[^}]*\}" ft_it2_mixed_lr1e5.log | tail -3
  echo "=== CONV-TEST: ft_it2_mixed_lr1e5 ==="
} > mixed_lr1e5_results.log 2>&1
python src/eval/benchmark_translate.py --test conv_splits/test.jsonl \
  --models ftit2:ft_it2_mixed_lr1e5 --limit 600 --batch-size 16 \
  >> mixed_lr1e5_results.log 2>&1
echo "=== FLORES: ft_it2_mixed_lr1e5 ===" >> mixed_lr1e5_results.log
python src/eval/benchmark_translate.py --test flores_test.jsonl \
  --models ftit2:ft_it2_mixed_lr1e5 --limit 997 --batch-size 16 \
  >> mixed_lr1e5_results.log 2>&1
echo "ALL DONE" >> mixed_lr1e5_results.log
