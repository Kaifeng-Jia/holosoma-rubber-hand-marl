#!/usr/bin/env bash
# T13 v4 D-Table-PA: in-simulator check (8 envs) and 200-iteration smoke (2048 envs), after run_speed.sh. Run inside tmux.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
until grep -q SPEED_DONE logs_system1/t13v4/speed/run.log 2>/dev/null; do sleep 20; done
COMMON="--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --seed 721 --obs-v1 --critic-v1 --per-agent-reward --contact-v1 $L/table_current_5kgA_labels.npz"
$PY -B scripts/train_core4d_smalltable.py $COMMON --num-envs 8 --obs-v1-noise off --t13v4-check logs_system1/t13v4/checks/table_pa \
  --output-dir logs_system1/t13v4/checks/table_pa_rundir > logs_system1/t13v4/checks/table_pa.log 2>&1
echo "pa check exit $? $(date +%T)"
$PY -B scripts/train_core4d_smalltable.py $COMMON --num-envs 2048 --steps-per-env 24 --system1-randomization --iterations 200 --save-interval 200 \
  --output-dir logs_system1/t13v4/smoke/d_table_pa_200 > logs_system1/t13v4/smoke/d_table_pa_200.log 2>&1
echo "pa smoke exit $? $(date +%T)"
echo PA_DONE
