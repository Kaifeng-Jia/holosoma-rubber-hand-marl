#!/usr/bin/env bash
# T13 v4 4 A: interleaved 30-iteration timings, speed-ups off / on / off / on (same config, same seed), after the Push retry.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/t13v4/labels
S=logs_system1/t13v4/speed
TABLE="--experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --num-envs 2048 --steps-per-env 24 --seed 721 --system1-randomization --obs-v1 --critic-v1 --contact-v1 $L/table_current_5kgA_labels.npz --save-interval 1000"
until grep -q PUSH_DONE logs_system1/t13v4/smoke/push_retry.log 2>/dev/null; do sleep 20; done
for tag in off1 on1 off2 on2; do
  flag=0; [ "${tag:0:2}" = on ] && flag=1
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader > $S/interleaved_${tag}_gpu_start.txt
  HOLOSOMA_S1_SPEEDUP=$flag $PY -B scripts/train_core4d_smalltable.py $TABLE --iterations 30 --output-dir $S/interleaved_${tag}_run > $S/interleaved_$tag.log 2>&1
  echo "interleaved $tag exit $? $(date +%T)"
done
echo TIMING2_DONE
