#!/bin/bash
# T14 step 9g second fix (Kevin 2026-11-26, received 10:29), after the 22-clip run (run_all22.sh). CPU only, at most 6 processes.
#  1. re-judge every evaluated level: non-hand penetration on the actual collision shapes (rejudge_nonhand_exact.py, 6 workers),
#     then the two-robot clearance on the actual collision shapes (rejudge_clearance_exact.py; verdict C = both exact);
#  2. step9g_feet_ik_table.py (leg stretch, walk-start foot steps), step9g2_compare.py (before 9g / after 9g2);
#  3. figures + videos of the clips applicable after (verdict C): step9g_render_both.py --clearance-exact, at most 2 renders
#     (2 only if MemAvailable >= 2 x 5.8 + 8 GB and no T13 local evaluation is running, else 1; waits for 5.8 + 8 GB);
#  4. package (step9e_package.py --step step9g2, after the videos for their sha256; logs renamed *_log.txt by hand).
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
HR=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
L=logs_system1/t14/step9g2
R=results_system1/T14_retargeting/step9g2
O=logs_system1/t14/rejudge_exact
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
until grep -q "^rc " $L/run_all22.log; do sleep 30; done
grep "^rc " $L/run_all22.log
taskset -c 16-31 $SO -B system1/t14/objects/rejudge_nonhand_exact.py --set step9g2 --driver $L/driver.json --out $O/step9g2 --workers 6 \
  --results $R/nonhand_exact --name step9g2_nonhand_exact > $O/step9g2.log 2>&1; echo "rejudge nonhand rc $? $(TZ=America/New_York date -Iseconds)"
python3 -B system1/t14/objects/rejudge_clearance_exact.py --set step9g2 --nonhand $R/nonhand_exact/step9g2_nonhand_exact.csv --out $R/clearance_exact \
  > $L/clearance_exact.log 2>&1; echo "rejudge clearance rc $?"; cat $L/clearance_exact.log
taskset -c 16-31 $HR -B system1/t14/objects/step9g_feet_ik_table.py --logdir $L --out $R/feet_ik.csv > $L/feet_ik.log 2>&1; echo "feet_ik rc $?"
python3 -B system1/t14/objects/step9g2_compare.py --before results_system1/T14_retargeting/step9g --after $R > $L/compare.log 2>&1; echo "compare rc $?"
CLIPS=$(python3 -c "import json;print(' '.join(json.load(open('$R/compare.json'))['totals']['after']['applicable_clip_names']))")
echo "render clips: $CLIPS"
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
t13() { python3 -c "
import json,glob,sys
sys.exit(0 if any(json.load(open(f)).get('state')=='running' for f in glob.glob('results_system1/T13_runs/*/post_run_status.json')) else 1)"; }
until [ "$(avail)" -ge 14 ]; do sleep 30; done
P=1; if [ "$(avail)" -ge 20 ] && ! t13; then P=2; fi
echo "render parallel $P avail $(avail) GB $(TZ=America/New_York date -Iseconds)"
[ -n "$CLIPS" ] && taskset -c 16-31 $SO -B system1/t14/objects/step9g_render_both.py $CLIPS --logdir $L --rejudge $O/step9g2 --clearance-exact \
  --fig-out $R/figures --video-out exports_system1/T14_videos/step9g2 --parallel $P > $L/render.log 2>&1; echo "render rc $? $(TZ=America/New_York date -Iseconds)"
python3 -B system1/t14/objects/step9e_package.py --step step9g2 > $L/package.log 2>&1; echo "package rc $?"
echo "end $(TZ=America/New_York date -Iseconds)"
