#!/bin/bash
cd /home/kevin/duet-system1
exec taskset -c 0-7 /home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python -B scripts/evaluate_paper_supplement.py \
  --case pull --model-id Pull-baseline_seed721 \
  --checkpoint /home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/model_08050.pt \
  --states /home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/pull_states.json \
  --output-dir /home/kevin/duet-system1/logs_system1/T01/repro_pull_seed721_pm2
