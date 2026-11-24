#!/usr/bin/env bash
# T13 v4 section 2.4: in-simulator checks (8 envs, 5070), one Isaac process at a time. Run inside tmux from the repo root.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
C=logs_system1/t13v4/checks
for noise in off on; do
  $PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 8 --seed 721 \
    --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --obs-v1-noise $noise --t13v4-check $C/table_noise_$noise \
    --output-dir $C/table_noise_${noise}_rundir > $C/table_noise_$noise.log 2>&1
  echo "table $noise exit $? $(date +%T)"
  $PY -B scripts/train_plan5_push.py --skill push --num-envs 8 --seed 721 --push-placement-proof system1/evidence/preflight_push_registered.json \
    --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --obs-v1-noise $noise --t13v4-check $C/push_noise_$noise \
    --output-dir $C/push_noise_${noise}_rundir > $C/push_noise_$noise.log 2>&1
  echo "push $noise exit $? $(date +%T)"
done
echo CHECKS_DONE
