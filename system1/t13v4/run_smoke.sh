#!/usr/bin/env bash
# T13 v4: (1) section 4 A baseline profile, D-Table new config, 2048 envs, 50 iterations (timers + one sync-scan rollout);
# (2) section 2.4 200-iteration smokes, D-Table and D-Push new config. One Isaac process at a time. Run inside tmux.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
TABLE="--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 2048 --steps-per-env 24 --seed 721 --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz"
$PY -B scripts/train_core4d_smalltable.py $TABLE --iterations 50 --save-interval 1000 \
  --t13v4-profile logs_system1/t13v4/speed/baseline_profile.jsonl --t13v4-sync-scan logs_system1/t13v4/speed/baseline_sync_scan.json \
  --output-dir logs_system1/t13v4/speed/baseline_run > logs_system1/t13v4/speed/baseline.log 2>&1
echo "profile exit $? $(date +%T)"
$PY -B scripts/train_core4d_smalltable.py $TABLE --iterations 200 --save-interval 200 --output-dir logs_system1/t13v4/smoke/d_table_200 \
  > logs_system1/t13v4/smoke/d_table_200.log 2>&1
echo "table smoke exit $? $(date +%T)"
$PY -B scripts/train_plan5_push.py --skill push --num-envs 2048 --steps-per-env 24 --seed 721 --iterations 200 --save-interval 200 \
  --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --output-dir logs_system1/t13v4/smoke/d_push_200 \
  > logs_system1/t13v4/smoke/d_push_200.log 2>&1
echo "push smoke exit $? $(date +%T)"
echo SMOKE_DONE
