#!/bin/bash
# F-item reruns after fixes: F1 online for the CORE4D cases (termination contract fix) and the
# scheme C preflights (SonicResidualEnv step-recursion fix). Same commands as t01b_f_gpu.sh.
set -uo pipefail
cd /home/kevin/duet-system1
P=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/T01b
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*"; }

for c in chair bucket table; do
  [ -f $L/f1_online/$c/result.json ] && continue
  [ -f $L/f1_online/$c.log ] && mv $L/f1_online/$c.log $L/f1_online/$c.attempt1.log
  log "F1 online $c"
  LOCAL_RANK=62 taskset -c 0-15 $P -B system1/sonic/f1_dual_online.py --case $c --out $L/f1_online/$c > $L/f1_online/$c.log 2>&1
done

log "C preflights (rerun after step-recursion fix)"
$P -B system1/cloud/run_train_queue.py c_pull:20 c_table:5 c_bucket:1 --out $L/preflight_c2 --python $P --gpu 0 \
   --local-rank 64 --cpus 0-15 --preflight 10 > $L/preflight_c2.log 2>&1
log "done"
