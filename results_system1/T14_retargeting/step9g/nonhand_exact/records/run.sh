#!/bin/bash
# T14 step 9g item 1 (Kevin 2026-11-26): re-judge step 8b, 9f, 9g (debug 3) with the non-hand penetration on the actual collision
# shapes (nonhand_exact.py); no solve re-run; one worker (the 22-clip run uses 5 of the 6 processes).
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
PY=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
O=logs_system1/t14/rejudge_exact
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
$PY -B system1/t14/objects/rejudge_nonhand_exact.py --set step9g --driver logs_system1/t14/step9g/driver_debug3.json --out $O/step9g_debug3 --workers 1 --results $O/tables --name step9g_debug3_nonhand_exact > $O/step9g_debug3.log 2>&1; echo "step9g_debug3 rc $?"
$PY -B system1/t14/objects/rejudge_nonhand_exact.py --set step9f --driver logs_system1/t14/step9f/driver.json --out $O/step9f --workers 1 --results $O/tables > $O/step9f.log 2>&1; echo "step9f rc $? $(TZ=America/New_York date -Iseconds)"
$PY -B system1/t14/objects/rejudge_nonhand_exact.py --set step8b --out $O/step8b --workers 1 --results $O/tables > $O/step8b.log 2>&1; echo "step8b rc $? $(TZ=America/New_York date -Iseconds)"
echo "end $(TZ=America/New_York date -Iseconds)"
