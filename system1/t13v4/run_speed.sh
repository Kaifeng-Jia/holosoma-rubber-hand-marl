#!/usr/bin/env bash
# T13 v4 section 4 A: same-seed 10-iteration traces (speed-ups off twice, on once) and 50-iteration timings (off / on), D-Table
# new config, 2048 envs, no profiler hooks. One Isaac process at a time. Run inside tmux after run_smoke.sh.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
S=logs_system1/t13v4/speed
TABLE="--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 2048 --steps-per-env 24 --seed 721 --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --save-interval 1000"
until grep -q SMOKE_DONE logs_system1/t13v4/smoke/run.log 2>/dev/null; do sleep 20; done
for tag in off_a off_b on_a; do
  flag=0; [ "${tag%%_*}" = on ] && flag=1
  HOLOSOMA_S1_SPEEDUP=$flag $PY -B scripts/train_core4d_smalltable.py $TABLE --iterations 10 --t13v4-trace $S/trace_$tag --output-dir $S/trace_${tag}_run > $S/trace_$tag.log 2>&1
  echo "trace $tag exit $? $(date +%T)"
done
for tag in off on; do
  flag=0; [ $tag = on ] && flag=1
  HOLOSOMA_S1_SPEEDUP=$flag $PY -B scripts/train_core4d_smalltable.py $TABLE --iterations 50 --output-dir $S/timing_${tag}_run > $S/timing_$tag.log 2>&1
  echo "timing $tag exit $? $(date +%T)"
done
echo SPEED_DONE
