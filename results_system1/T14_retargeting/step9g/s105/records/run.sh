#!/bin/bash
# T14 step 9g item 3 (Kevin 2026-11-26): the partner's Stage-1 infeasibility of smallboxa (frame 105), constraint groups; the
# failing levels' exact role_solve commands (driver_debug3.json), output moved here; one process at a time.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
for m in 1.0 1.25 1.5; do
  t=m$(printf '%.2f' $m | tr . p)
  $PY -B system1/t14/objects/step9g_s105_diag.py --out-dir logs_system1/t14/step9g_s105/$t -- --family box --object smallbox --sequence sub4_smallbox_047 \
    --mode role --construction frontback --role 1 --m $m --frames 36:172 --tag smallboxa --fb-turn world --fb-legs mirror --fb-human-frame target \
    > logs_system1/t14/step9g_s105/$t.log 2>&1
  echo "$t rc $? $(TZ=America/New_York date -Iseconds)"
done
echo "end $(TZ=America/New_York date -Iseconds)"
