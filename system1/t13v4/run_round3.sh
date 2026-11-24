#!/usr/bin/env bash
# T13 v4: repeat of the in-simulator checks with the current code, to see whether the replay hand-force fractions move
# between runs (noise-on rerun at 01:53 gave Table 13.9 % / Push 79.8 % vs noise-off at 00:04 15.0 % / 86.9 %).
# Noise off twice for each task, noise on once more for Push. New output directories (_r2/_r3); nothing overwritten.
# One Isaac process at a time; each launch waits until MemAvailable >= 17 GB. Run inside tmux from the repo root.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
C=logs_system1/t13v4/checks
gate() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 17 ]; do sleep 30; done; echo "launch $1 avail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB $(date +%T)"; }
table() { gate $1; $PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 8 --seed 721 \
  --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --obs-v1-noise $2 --t13v4-check $C/$1 --output-dir $C/${1}_rundir > $C/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
push() { gate $1; $PY -B scripts/train_plan5_push.py --skill push --num-envs 8 --seed 721 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --obs-v1-noise $2 --t13v4-check $C/$1 --output-dir $C/${1}_rundir > $C/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
table table_noise_off_r2 off
table table_noise_off_r3 off
push push_noise_off_r2 off
push push_noise_off_r3 off
push push_noise_on_r2 on
echo ROUND3_DONE
