#!/usr/bin/env bash
# T13 v4 4 B: smoke of the --torch-threads switch and the run_config "torch_threads" record (2 iterations, 256 envs).
# Table with --torch-threads 4, Push without it (default path). One Isaac process at a time; MemAvailable gate 17 GB.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
S=logs_system1/t13v4/smoke
gate() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 17 ]; do sleep 30; done; echo "launch $1 avail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB $(date +%T)"; }
gate threads_table
$PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 256 --steps-per-env 24 --seed 721 \
  --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --iterations 2 --save-interval 1000 --torch-threads 4 \
  --output-dir $S/threads_table_2 > $S/threads_table_2.log 2>&1
echo "threads_table exit $? $(date +%T)"
gate threads_push
$PY -B scripts/train_plan5_push.py --skill push --num-envs 256 --steps-per-env 24 --seed 721 --iterations 2 --save-interval 1000 \
  --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --output-dir $S/threads_push_2 > $S/threads_push_2.log 2>&1
echo "threads_push exit $? $(date +%T)"
echo THREADS_DONE
