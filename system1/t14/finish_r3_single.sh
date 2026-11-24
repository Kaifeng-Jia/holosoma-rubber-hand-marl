#!/usr/bin/env bash
# T14 round 3, Push / Pull: a-search and levels a, 1.0, 1.5, 2.0 from the round-3 single solve (exact measure), with round 2
# re-measured (its a and levels), the current reference (50 Hz single, nearest frames) and OmniRetarget as is at 1.0;
# levels json, grid (close-up column = middle contact frame) and video. Run inside tmux.
set -eu
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1
declare -A CUR=([push]=$W/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz [pull]=$W/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz)
declare -A A2=([push]=0.70 [pull]=0.55)
task=$1; tag=${2:-}
STAGE=${STAGE:-all}
S=logs_system1/t14/r3; R=exports_system1/T14_references/r3; OUT=results_system1/T14_retargeting/step3
r2=logs_system1/t14/single/$task/${task}_new_qpos.npz
refs=""
if [ -z "$tag" ]; then
  for m in ${A2[$task]} 1.0 1.5 2.0; do refs="$refs --ref r2_m$m=$r2@$m"; done
  refs="$refs --ref current=${CUR[$task]}@1.0@fps50 --ref omni_asis=$S/single_asis_$task/${task}_omni_asis_qpos.npz@1.0"
fi
mkdir -p $OUT/$task
if [ $STAGE != render ]; then
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $task $refs --single r3${tag}=$S/single_$task$tag/${task}_new_qpos.npz --search-a --levels 1.0 1.5 2.0 \
    --pairs-out $R/${task}_sizes$tag --out $OUT/$task/eval_r3$tag.json
fi
[ $STAGE = eval ] && { echo EVAL_DONE $task $tag $(date +%T); exit 0; }
$SO -B system1/t14/levels_r3.py --task $task --eval $OUT/$task/eval_r3$tag.json --prefix r3$tag --solves $S --solve-tag "$tag" --refs-dir $R --out $S/levels/${task}$tag.json
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json OMP_NUM_THREADS=2 \
  $SO -B system1/t14/render_sizes.py --task $task --levels $S/levels/${task}$tag.json --out $OUT/figures --video-out exports_system1/T14_videos/r3$tag \
  --closeup mid --grid-name ${task}_r3${tag}_grid
echo FINISH_DONE $task $tag $(date +%T)
