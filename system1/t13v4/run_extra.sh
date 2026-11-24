#!/usr/bin/env bash
# T13 v4 4 A attribution: 20-iteration profile of the same D-Table config without the contact items (no per-link sensor).
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
until grep -q PA_DONE logs_system1/t13v4/pa/run.log 2>/dev/null; do sleep 20; done
$PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 2048 --steps-per-env 24 \
  --seed 721 --system1-randomization --obs-v1 --critic-v1 --iterations 20 --save-interval 1000 \
  --t13v4-profile logs_system1/t13v4/speed/nocontact_profile.jsonl --output-dir logs_system1/t13v4/speed/nocontact_run > logs_system1/t13v4/speed/nocontact.log 2>&1
echo "nocontact exit $? $(date +%T)"
echo EXTRA_DONE
