#!/bin/bash
# Autonomous: wait for mixed fine-tune to finish, free GPU, run both benchmarks.
cd ~/eng-hindi-model
while true; do
  if grep -qa "saved -> ft_it2_mixed" ft_it2_mixed.log 2>/dev/null; then break; fi
  if ! tmux has-session -t it2mix 2>/dev/null; then break; fi
  sleep 60
done
sleep 5
tmux kill-session -t it2mix 2>/dev/null
pkill -f finetune_it2 2>/dev/null
sleep 12
export PATH="$HOME/.local/bin:$PATH"
source ~/it2env311/bin/activate
{
  echo "=== mixed dev evals (from training) ==="
  grep -aoE "\{.*chrf[^}]*\}" ft_it2_mixed.log | tail -3
  echo "=== CONV-TEST: ft_it2_mixed ==="
} > mixed_results.log 2>&1
python src/eval/benchmark_translate.py --test conv_splits/test.jsonl --models ftit2:ft_it2_mixed --limit 600 --batch-size 16 >> mixed_results.log 2>&1
echo "=== FLORES: ft_it2_mixed ===" >> mixed_results.log
python src/eval/benchmark_translate.py --test flores_test.jsonl --models ftit2:ft_it2_mixed --limit 997 --batch-size 16 >> mixed_results.log 2>&1
echo "ALL DONE" >> mixed_results.log
