#!/bin/bash
# T12-v2: the 10 side-by-side videos (5 tasks x full / close), CPU (Mesa llvmpipe), in parallel; PIDs in $OUT/logs/pids.txt
OUT=${1:-exports_system1/T12_videos_v2}
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LP_NUM_THREADS=3
PY=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
for task in push pull bucket chair table; do
  for cam in full close; do
    nohup $PY -B system1/t12v2/render_t12v2.py --task $task --camera $cam --out "$OUT" > "$OUT/logs/${task}_${cam}.log" 2>&1 &
    echo "$! $task $cam" >> "$OUT/logs/pids.txt"
  done
done
