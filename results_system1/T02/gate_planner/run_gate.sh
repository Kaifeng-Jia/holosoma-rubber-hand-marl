#!/bin/bash
cd /home/kevin/duet-system1
export LOCAL_RANK=71 WORLD_SIZE=1
for fb in shift none; do
  out=logs_system1/T02/gate_planner_${fb}
  echo "$(TZ=America/New_York date '+%F %T %Z') start $fb git=$(git rev-parse HEAD)"
  timeout 3600 taskset -c 16-23 /home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python -B system1/sonic/validate_planner_single.py \
     --ref-prefix logs_system1/T01/sonic/refs/sonic_sample_A001 --out $out --feedback $fb --device cuda:0 > ${out}.log 2>&1
  echo "$(TZ=America/New_York date '+%F %T %Z') end $fb rc=$?"
done
