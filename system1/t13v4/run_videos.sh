#!/bin/bash
# T13 v4 section 5 item 1: side-by-side videos of one run (+/-2 nominal and perturbed_000; full / close / hands), CPU (Mesa
# llvmpipe), at most $3 at a time on cores 0-15 (memory budget: total <= MemAvailable - 8 GB); one log per video
RUN=$1
OUT=${2:-exports_system1/T13_videos/$RUN}
P=${3:-2}
cd /home/kevin/duet-system1
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LP_NUM_THREADS=3
export PY=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python RUN OUT
for state in nominal perturbed_000; do for cam in full close hands; do echo "$state $cam"; done; done | \
  xargs -P "$P" -L 1 bash -c 'taskset -c 0-15 $PY -B system1/t13v4/render_t13.py --run $RUN --state $0 --camera $1 --out "$OUT" > "$OUT/logs/$0_$1.log" 2>&1; echo "$0 $1 rc $? $(date -Is)" >> "$OUT/logs/rc.txt"'
echo VIDEOS_DONE
