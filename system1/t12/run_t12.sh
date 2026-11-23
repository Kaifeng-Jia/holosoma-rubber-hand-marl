#!/bin/bash
# T12-v: render the six 2x2 videos (3 states x full / close) on the CPU, in parallel; one log per video, PIDs in pids.txt.
set -u
cd "$(dirname "$0")/../.."
OUT=${1:-exports_system1/T12_videos}
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LP_NUM_THREADS=5
PY=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
: > "$OUT/logs/pids.txt"
for s in perturbed_000 perturbed_001 perturbed_002; do
  for c in full close; do
    $PY -B system1/t12/render_t12.py --state $s --camera $c --out "$OUT" > "$OUT/logs/${s}_${c}.log" 2>&1 &
    echo "$! $s $c $(date -Iseconds)" >> "$OUT/logs/pids.txt"
  done
done
wait
echo "all done $(date -Iseconds)" >> "$OUT/logs/pids.txt"
