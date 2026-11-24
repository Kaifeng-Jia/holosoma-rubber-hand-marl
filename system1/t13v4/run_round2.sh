#!/usr/bin/env bash
# T13 v4 after Kevin's 2026-11-24 decisions: noise-on checks with the shared object-point noise; observation-v1 validation
# tests with scripts/evaluate_paper_supplement.py (deterministic, randomisation off, noise off). One Isaac process at a time;
# each launch waits until MemAvailable >= 8 GB + 9 GB (Isaac process estimate). Run inside tmux.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
C=logs_system1/t13v4/checks
V=logs_system1/t13v4/validation
gate() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 17 ]; do sleep 30; done; echo "launch $1 avail $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB $(date +%T)"; }
gate table_noise_on
$PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 8 --seed 721 \
  --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --obs-v1-noise on --t13v4-check $C/table_noise_on --output-dir $C/table_noise_on_rundir > $C/table_noise_on.log 2>&1
echo "table_noise_on exit $?"
gate push_noise_on
$PY -B scripts/train_plan5_push.py --skill push --num-envs 8 --seed 721 --push-placement-proof system1/evidence/preflight_push_registered.json \
  --obs-v1 --critic-v1 --contact-v1 $L/push_current_a1_labels.npz --obs-v1-noise on --t13v4-check $C/push_noise_on --output-dir $C/push_noise_on_rundir > $C/push_noise_on.log 2>&1
echo "push_noise_on exit $?"
TS=system1/t10b/validation_states/table_validation_states.json
for tag in table_a table_b; do
  gate val_$tag
  $PY -B scripts/evaluate_paper_supplement.py --case table --controller policy --obs-v1 --critic-v1 --checkpoint logs_system1/t13v4/smoke/d_table_200/model_00200.pt \
    --device cuda:0 --states $TS --limit 5 --model-id t13v4_validation_test__d_table_smoke__$tag --output-dir $V/$tag > $V/$tag.log 2>&1
  echo "val $tag exit $?"
done
gate val_table_pa
$PY -B scripts/evaluate_paper_supplement.py --case table --controller policy --obs-v1 --critic-v1 --checkpoint logs_system1/t13v4/smoke/d_table_pa_200/model_00200.pt \
  --device cuda:0 --states $TS --limit 5 --model-id t13v4_validation_test__d_table_pa_smoke --output-dir $V/table_pa > $V/table_pa.log 2>&1
echo "val table_pa exit $?"
gate val_push
$PY -B scripts/evaluate_paper_supplement.py --case push --controller policy --obs-v1 --critic-v1 --checkpoint logs_system1/t13v4/smoke/d_push_200/model_00200.pt \
  --push-placement-proof system1/evidence/preflight_push_registered.json --device cuda:0 \
  --states /home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/push_states.json --limit 5 \
  --model-id t13v4_validation_test__d_push_smoke --output-dir $V/push > $V/push.log 2>&1
echo "val push exit $?"
echo ROUND2_DONE
