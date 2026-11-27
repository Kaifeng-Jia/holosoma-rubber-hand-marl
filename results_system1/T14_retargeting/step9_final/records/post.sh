#!/bin/bash
# T14 final test after step 9g2, after the driver (run.sh). Same post-run as step 9g2 (post22.sh) without the before/after
# comparison (no "before" for these clips). CPU only, at most 6 processes.
#  1. non-hand penetration on the actual collision shapes (rejudge_nonhand_exact.py, 6 workers), then the two-robot clearance on
#     the actual collision shapes (rejudge_clearance_exact.py; verdict C = both exact, threshold >= -1 cm);
#  2. step9g_feet_ik_table.py (leg stretch, walk-start foot steps);
#  3. side-by-side sheet, level grid and video of the clips applicable under verdict C (step9g_render_both.py --clearance-exact),
#     at most 2 renders (2 only if MemAvailable >= 20 GB, else 1; waits for 14 GB); levels_both.json copied to results;
#  4. package (step9e_package.py --step step9_final --turns ...).
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
HR=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
L=logs_system1/t14/step9_final
R=results_system1/T14_retargeting/step9_final
O=logs_system1/t14/rejudge_exact
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
until grep -q "^rc " $L/driver.log; do sleep 30; done
grep "^rc " $L/driver.log
taskset -c 16-31 $SO -B system1/t14/objects/rejudge_nonhand_exact.py --set step9_final --driver $L/driver.json --out $O/step9_final --workers 6 \
  --results $R/nonhand_exact --name step9_final_nonhand_exact > $O/step9_final.log 2>&1; echo "rejudge nonhand rc $? $(TZ=America/New_York date -Iseconds)"
python3 -B system1/t14/objects/rejudge_clearance_exact.py --set step9_final --nonhand $R/nonhand_exact/step9_final_nonhand_exact.csv --out $R/clearance_exact \
  > $L/clearance_exact.log 2>&1; echo "rejudge clearance rc $?"; cat $L/clearance_exact.log
taskset -c 16-31 $HR -B system1/t14/objects/step9g_feet_ik_table.py --logdir $L --out $R/feet_ik.csv > $L/feet_ik.log 2>&1; echo "feet_ik rc $?"
CLIPS=$(python3 -c "
import json
s=json.load(open('$R/clearance_exact/step9_final_clearance_exact.json'))['summary']
print(' '.join(c for c,v in s.items() if isinstance(v,dict) and v.get('applicable_C')))")
echo "render clips: $CLIPS"
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
until [ "$(avail)" -ge 14 ]; do sleep 30; done
P=1; if [ "$(avail)" -ge 20 ]; then P=2; fi
echo "render parallel $P avail $(avail) GB $(TZ=America/New_York date -Iseconds)"
[ -n "$CLIPS" ] && taskset -c 16-31 $SO -B system1/t14/objects/step9g_render_both.py $CLIPS --logdir $L --rejudge $O/step9_final --clearance-exact \
  --fig-out $R/figures --video-out exports_system1/T14_videos/step9_final --parallel $P > $L/render.log 2>&1; echo "render rc $? $(TZ=America/New_York date -Iseconds)"
for c in $CLIPS; do mkdir -p $R/levels/$c; cp -p $L/$c/levels_both.json $R/levels/$c/; done
python3 -B system1/t14/objects/step9e_package.py --step step9_final --turns $R/class_a_turns.csv > $L/package.log 2>&1; echo "package rc $?"
echo "end $(TZ=America/New_York date -Iseconds)"
