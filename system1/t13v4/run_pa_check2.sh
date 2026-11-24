#!/usr/bin/env bash
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
until grep -q TIMING2_DONE logs_system1/t13v4/speed/timing2.log 2>/dev/null; do sleep 20; done
$PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --seed 721 --obs-v1 --critic-v1 \
  --per-agent-reward --contact-v1 $L/table_current_5kgA_labels.npz --num-envs 8 --obs-v1-noise off --t13v4-check logs_system1/t13v4/checks/table_pa \
  --output-dir logs_system1/t13v4/checks/table_pa_rundir > logs_system1/t13v4/checks/table_pa.log 2>&1
echo "pa check exit $? $(date +%T)"
echo PA_CHECK2_DONE
