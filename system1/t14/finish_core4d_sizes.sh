#!/usr/bin/env bash
# T14 item 2, Bin / Chair: after run_core4d_sizes.sh. Combine both persons per level, scale the training object URDF by k,
# evaluate (exact convex distances), compose the level numbers, render the grid and the video. Run inside tmux.
set -eu
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking
declare -A URDF=([bin]=$W/core4d_bucket003_20231020_071_a1/bucket003_training.urdf [chair]=$W/core4d_chair021_20231020_074_a1/chair021_training.urdf)
for task in "$@"; do
  refs=""
  for k in 0.8 0.9 1.0 1.1 1.2; do
    kk=${k/./p}; out=exports_system1/T14_references/${task}_k${kk}
    if [ ! -f logs_system1/t14/core4d/${task}_k${kk}/person1/run_record.json ] || [ ! -f logs_system1/t14/core4d/${task}_k${kk}/person2/run_record.json ]; then
      echo "skip $task k$k: a person's solve did not finish"; continue
    fi
    OMP_NUM_THREADS=1 $HS -B system1/t14/table_combine.py --task $task --variant-dir logs_system1/t14/core4d/${task}_k${kk} --variant new --out $out
    python3 system1/t14/scale_box_urdf.py --urdf ${URDF[$task]} --k $k --out $out/${task}_k${kk}.urdf
    refs="$refs --ref k$k=$out/${task}_new_fps30.npz@$k"
  done
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_core4d_hulls.py --task $task $refs --out results_system1/T14_retargeting/step2_sizes/$task
  $SO -B system1/t14/size_levels.py --task $task --out logs_system1/t14/levels/$task.json
  CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json OMP_NUM_THREADS=2 \
    $SO -B system1/t14/render_sizes.py --task $task --levels logs_system1/t14/levels/$task.json --out results_system1/T14_retargeting/step2_sizes/figures --video-out exports_system1/T14_videos
done
echo FINISH_DONE
