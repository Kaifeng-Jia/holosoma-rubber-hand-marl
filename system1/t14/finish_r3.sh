#!/usr/bin/env bash
# T14 round 3, CORE4D tasks: combine both persons per level, scale the object URDF (rendering), evaluate with the exact
# measure (round 3 levels; for the plain round also round 2 re-measured, the current reference and OmniRetarget as is),
# compose the levels and render the grid (with the hand close-up column) and the video. Run inside tmux.
#   bash system1/t14/finish_r3.sh <task> [solve_tag] [k ...]     (solve_tag "" = round 3, "_a10" = round 3 + step 2)
set -eu
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking
declare -A URDF=([table]=$W/core4d_smalltable5kg_A/desk001_5kg_training.urdf [bin]=$W/core4d_bucket003_20231020_071_a1/bucket003_training.urdf [chair]=$W/core4d_chair021_20231020_074_a1/chair021_training.urdf)
declare -A CUR=([table]=$W/core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz [bin]=$W/core4d_bucket003_20231020_071_a1/core4d_pair_compact_fps30.npz [chair]=$W/core4d_chair021_20231020_074_a1/core4d_pair_compact_fps30.npz)
task=$1; tag=${2:-}
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
KS=${*:-0.8 0.9 1.0 1.1 1.2}
S=logs_system1/t14/r3; R=exports_system1/T14_references/r3; OUT=results_system1/T14_retargeting/step3
pairname() { if [ "$1" = table ]; then echo "20231030_001_desk001_move_$2_fps30.npz"; else echo "$1_$2_fps30.npz"; fi; }
refs=""
for k in $KS; do
  kk=k${k/./p}; d=$S/${task}_${kk}${tag}; out=$R/${task}_${kk}${tag}
  if [ ! -f $d/person1/run_record.json ] || [ ! -f $d/person2/run_record.json ]; then echo "skip $task k$k$tag: a solve did not finish"; continue; fi
  [ -d $out ] || OMP_NUM_THREADS=1 $HS -B system1/t14/table_combine.py --task $task --variant-dir $d --variant new --out $out
  python3 system1/t14/scale_box_urdf.py --urdf ${URDF[$task]} --k $k --out $out/${task}_${kk}.urdf
  refs="$refs --ref r3${tag}_k$k=$out/$(pairname $task new)@$k"
  r2=exports_system1/T14_references/${task}_${kk}/$(pairname $task new)
  if [ -z "$tag" ] && [ -f $r2 ]; then refs="$refs --ref r2_k$k=$r2@$k"; fi
done
if [ -z "$tag" ]; then
  refs="$refs --ref current=${CUR[$task]}@1.0"
  out=$R/asis_${task}_k1p0
  [ -d $out ] || OMP_NUM_THREADS=1 $HS -B system1/t14/table_combine.py --task $task --variant-dir $S/asis_${task}_k1p0 --variant omni_asis --out $out
  refs="$refs --ref omni_asis=$out/$(pairname $task omni_asis)@1.0"
fi
mkdir -p $OUT/$task
STAGE=${STAGE:-all}
if [ $STAGE != render ]; then
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $task $refs --out $OUT/$task/eval_r3${tag}.json
fi
[ $STAGE = eval ] && { echo EVAL_DONE $task $tag $(date +%T); exit 0; }
only=""; [ -n "$tag" ] && only="--only $KS"
$SO -B system1/t14/levels_r3.py --task $task --eval $OUT/$task/eval_r3${tag}.json --prefix r3${tag} --solves $S --solve-tag "$tag" --refs-dir $R $only --out $S/levels/${task}${tag}.json
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json OMP_NUM_THREADS=2 \
  $SO -B system1/t14/render_sizes.py --task $task --levels $S/levels/${task}${tag}.json --out $OUT/figures --video-out exports_system1/T14_videos/r3${tag} \
  --closeup lift --grid-name ${task}_r3${tag}_grid
echo FINISH_DONE $task $tag $(date +%T)
