#!/usr/bin/env bash
# T13 v4: D-Push 200-iteration smoke (retry after the randomisation-aware object check), after run_extra.sh.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
until grep -q PA_CHECK2_DONE logs_system1/t13v4/pa/check2.log 2>/dev/null; do sleep 20; done
$PY -B scripts/train_plan5_push.py --skill push --num-envs 2048 --steps-per-env 24 --seed 721 --iterations 200 --save-interval 200 \
  --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --output-dir logs_system1/t13v4/smoke/d_push_200 \
  > logs_system1/t13v4/smoke/d_push_200.log 2>&1
echo "push smoke exit $? $(date +%T)"
echo PUSH2_DONE
