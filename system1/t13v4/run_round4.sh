#!/usr/bin/env bash
# T13 v4: history test of the contact replay. Same commands as the checks, plus S1_T13V4_REPLAY_PASSES=2 (the whole reference
# replayed twice in one process) and the reference frames after reset_all recorded. Table and Push, noise off and on.
# One Isaac process at a time; each launch waits until MemAvailable >= 17 GB. Run inside tmux from the repo root.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
C=logs_system1/t13v4/checks
export S1_T13V4_REPLAY_PASSES=2
gate() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 17 ]; do sleep 30; done; echo "launch $1 avail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB $(date +%T)"; }
table() { gate $1; $PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 8 --seed 721 \
  --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --obs-v1-noise $2 --t13v4-check $C/$1 --output-dir $C/${1}_rundir > $C/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
push() { gate $1; $PY -B scripts/train_plan5_push.py --skill push --num-envs 8 --seed 721 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --obs-v1-noise $2 --t13v4-check $C/$1 --output-dir $C/${1}_rundir > $C/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
table table_noise_off_h2 off
table table_noise_on_h2 on
push push_noise_off_h2 off
push push_noise_on_h2 on
echo ROUND4_DONE
