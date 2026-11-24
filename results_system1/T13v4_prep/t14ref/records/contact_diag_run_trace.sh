#!/usr/bin/env bash
# same-seed 3-iteration Table / Push traces (256 envs) before / after the contact-v1 iteration diagnostics: $1 = tag
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14r6C
L=system1/t13v4/labels_t14r6C
O=logs_system1/t13v4/diag
$PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --seed 721 --obs-v1 --critic-v1 --system1-randomization \
  --contact-v1 $L/table_t14r6C_k1p0_labels.npz --t14-reference $W/table/t14_reference.json --num-envs 256 --steps-per-env 24 --iterations 3 --save-interval 1000 \
  --t13v4-trace $O/table_$1_trace.npz --output-dir $O/table_$1 > $O/table_$1.log 2>&1; echo "table $1 exit $? $(date +%T)"
$PY -B scripts/train_plan5_push.py --skill push --seed 721 --obs-v1 --critic-v1 --system1-randomization --push-placement-proof system1/evidence/preflight_push_registered.json \
  --contact-v1 $L/push_t14r6C_m1p00_labels.npz --t14-reference $W/push/t14_reference.json --num-envs 256 --steps-per-env 24 --iterations 3 --save-interval 1000 \
  --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --output-dir $O/push_$1 > $O/push_$1.log 2>&1; echo "push $1 exit $? $(date +%T)"
echo TRACE_DONE $1
