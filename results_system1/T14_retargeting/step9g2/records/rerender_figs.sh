#!/bin/bash
# T14 step 9g2: re-render the figures only (no video; the videos carry no level label) after the level-label fix
# (render_box.py first_free_label: "1.5 以上第一个不相交档", the protocol's level; the first render wrote "1.0 以上"). First render
# moved to _to_delete/T14_step9g2_figures_label_1p0/. CPU, one clip at a time, each in a user scope with MemoryMax=4G (no swap),
# started only when MemAvailable >= 12 GB, so that available memory stays >= 8 GB while the T13 local evaluation runs
# (the 5.8 GB measured in step 9g included the video's scenes). Peak memory of each scope is logged.
# Version 2 (13:00): version 1 waited for 14 GB with no cap; stopped by its PID before it rendered anything.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
L=logs_system1/t14/step9g2
R=results_system1/T14_retargeting/step9g2
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
git diff --stat system1/t14/objects/render_box.py system1/t14/objects/step9g_render_both.py
CLIPS=$(python3 -c "import json;print(' '.join(json.load(open('$R/compare.json'))['totals']['after']['applicable_clip_names']))")
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
for c in $CLIPS; do
  until [ "$(avail)" -ge 12 ]; do sleep 30; done
  u=t14fig_$c_$$_$RANDOM
  echo "clip $c avail $(avail) GB $(TZ=America/New_York date -Iseconds)"
  systemd-run --user --scope --quiet --unit=$u -p MemoryMax=4G -p MemorySwapMax=0 \
    bash -c "echo scope_cgroup \$(cat /proc/self/cgroup); taskset -c 16-31 $SO -B system1/t14/objects/step9g_render_both.py $c --logdir $L \
      --rejudge logs_system1/t14/rejudge_exact/step9g2 --clearance-exact --fig-out $R/figures --parallel 1; rc=\$?; \
      cg=/sys/fs/cgroup\$(cut -d: -f3 /proc/self/cgroup); echo peak_GB \$(awk '{printf \"%.2f\", \$1/1073741824}' \$cg/memory.peak) oom \$(grep oom_kill \$cg/memory.events); exit \$rc"
  echo "clip $c rc $? $(TZ=America/New_York date -Iseconds)"
done
echo end
