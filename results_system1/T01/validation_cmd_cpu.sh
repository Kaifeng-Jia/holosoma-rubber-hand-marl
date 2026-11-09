#!/bin/bash
cd /home/kevin/duet-system1
export LOCAL_RANK=31 WORLD_SIZE=1
for ref in sonic_sample_A001 sonic_sample_A001_M pull_agent0 table_agent0; do
  out=logs_system1/T01/sonic/val_cpu_${ref}
  echo "$(TZ=America/New_York date '+%F %T') start $ref"
  timeout 3600 /home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python -B system1/sonic/validate_single.py \
     --ref-prefix logs_system1/T01/sonic/refs/$ref --out $out --device cpu > ${out}.log 2>&1
  echo "$(TZ=America/New_York date '+%F %T') end $ref rc=$?"
done
