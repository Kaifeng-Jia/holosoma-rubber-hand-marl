#!/usr/bin/env bash
# T13 v4 section 0 with the T14 round 6 C references (Kevin 2026-11-24): (A) section 2.4 in-simulator checks incl. the reference
# replay (per-robot hand force > 1 N share), noise off twice + noise on once per task; (B) 200-iteration smokes D-Table / D-Push /
# D-Table-PA (2048 envs, training config); (C) observation-v1 validation (evaluate_paper_supplement.py, noise off, deterministic)
# on the smoke checkpoints with the new validation states (5 states each). One Isaac process at a time; each launch waits until
# MemAvailable >= 17 GB. Run inside tmux from the repo root. New output dirs only.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14r6C
L=system1/t13v4/labels_t14r6C
ST=system1/t13v4/states_t14r6C
O=logs_system1/t13v4/t14
mkdir -p $O/checks $O/smoke $O/validation
TREF="--t14-reference $W/table/t14_reference.json"
PREF="--t14-reference $W/push/t14_reference.json --push-placement-proof system1/evidence/preflight_push_registered.json"
TCOM="--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --seed 721 --obs-v1 --critic-v1 --contact-v1 $L/table_t14r6C_k1p0_labels.npz $TREF"
PCOM="--skill push --seed 721 --obs-v1 --critic-v1 --contact-v1 $L/push_t14r6C_m1p00_labels.npz $PREF"
gate() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 17 ]; do sleep 30; done; echo "launch $1 avail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB $(date +%T)"; }
check_table() { gate $1; $PY -B scripts/train_core4d_smalltable.py $TCOM $3 --num-envs 8 --obs-v1-noise $2 --t13v4-check $O/checks/$1 --output-dir $O/checks/${1}_rundir > $O/checks/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
check_push() { gate $1; $PY -B scripts/train_plan5_push.py $PCOM --num-envs 8 --obs-v1-noise $2 --t13v4-check $O/checks/$1 --output-dir $O/checks/${1}_rundir > $O/checks/$1.log 2>&1; echo "$1 exit $? $(date +%T)"; }
# (A)
check_table table_noise_off off ""
check_table table_noise_off_r2 off ""
check_table table_noise_on on ""
check_table table_pa off "--per-agent-reward"
check_push push_noise_off off
check_push push_noise_off_r2 off
check_push push_noise_on on
echo CHECKS_DONE $(date +%T)
# (B)
gate d_table_200; $PY -B scripts/train_core4d_smalltable.py $TCOM --num-envs 2048 --steps-per-env 24 --system1-randomization --iterations 200 --save-interval 200 \
  --output-dir $O/smoke/d_table_200 > $O/smoke/d_table_200.log 2>&1; echo "d_table_200 exit $? $(date +%T)"
gate d_push_200; $PY -B scripts/train_plan5_push.py $PCOM --num-envs 2048 --steps-per-env 24 --iterations 200 --save-interval 200 \
  --actor-learning-rate 0.00001 --critic-learning-rate 0.001 --system1-randomization --output-dir $O/smoke/d_push_200 > $O/smoke/d_push_200.log 2>&1; echo "d_push_200 exit $? $(date +%T)"
gate d_table_pa_200; $PY -B scripts/train_core4d_smalltable.py $TCOM --per-agent-reward --num-envs 2048 --steps-per-env 24 --system1-randomization --iterations 200 --save-interval 200 \
  --output-dir $O/smoke/d_table_pa_200 > $O/smoke/d_table_pa_200.log 2>&1; echo "d_table_pa_200 exit $? $(date +%T)"
echo SMOKES_DONE $(date +%T)
# (C)
val() { gate val_$1; $PY -B scripts/evaluate_paper_supplement.py --case $2 --controller policy --obs-v1 --critic-v1 $3 --checkpoint $O/smoke/$4/model_00200.pt \
  --device cuda:0 --states $ST/$2_validation_states.json --limit 5 --model-id t13v4_t14_validation_test__$1 --output-dir $O/validation/$1 > $O/validation/$1.log 2>&1; echo "val $1 exit $? $(date +%T)"; }
val table table "$TREF" d_table_200
val table_b table "$TREF" d_table_200
val table_pa table "$TREF" d_table_pa_200
val push push "$PREF" d_push_200
echo ALL_DONE $(date +%T)
