#!/bin/bash
# preregistration section 8: last waypoint at H/2, H/4, H/8 in this order; first value whose shift (position feedback)
# run passes is fixed; the none run at the same value is reported only. Same reference, same criteria.
cd /home/kevin/duet-system1
P=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
for frac in 0.5 0.25 0.125; do
  for fb in shift none; do
    o=logs_system1/T02/gate_revision/frac_${frac}_${fb}
    rank=$([ $fb = shift ] && echo 71 || echo 72); cpus=$([ $fb = shift ] && echo 16-23 || echo 24-31)
    ( echo "$(TZ=America/New_York date '+%F %T %Z') start git=$(git rev-parse HEAD) dirty=$(git status --porcelain system1 scripts | wc -l)" > $o.cmd
      LOCAL_RANK=$rank WORLD_SIZE=1 timeout 3600 taskset -c $cpus $P -B system1/sonic/validate_planner_single.py \
        --ref-prefix logs_system1/T01/sonic/refs/sonic_sample_A001 --out $o --feedback $fb --device cuda:0 --last-waypoint-frac $frac > $o.log 2>&1
      echo "rc=$? $(TZ=America/New_York date '+%F %T %Z')" >> $o.cmd ) &
  done
  wait
  passed=$(python3 -c "import json;print(json.load(open('logs_system1/T02/gate_revision/frac_${frac}_shift/result.json'))['gate_passed'])" 2>/dev/null)
  echo "$(TZ=America/New_York date '+%F %T %Z') frac=$frac shift gate_passed=$passed"
  if [ "$passed" = "True" ]; then echo "SELECTED $frac"; exit 0; fi
done
echo "NONE PASSED"; exit 3
