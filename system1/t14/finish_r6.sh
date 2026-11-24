#!/usr/bin/env bash
# T14 round 6, one task x version (C or Cp): combine / make the levels, evaluate (exact measure + hand metrics),
# compose the levels, render the grid (close-up = person and robot side by side) and the video. Run inside tmux.
#   bash system1/t14/finish_r6.sh <task> <C|Cp>
set -eu
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking
declare -A URDF=([table]=$W/core4d_smalltable5kg_A/desk001_5kg_training.urdf [bin]=$W/core4d_bucket003_20231020_071_a1/bucket003_training.urdf [chair]=$W/core4d_chair021_20231020_074_a1/chair021_training.urdf)
task=$1; V=$2
S=logs_system1/t14/r6; R=exports_system1/T14_references/r6; OUT=results_system1/T14_retargeting/step6
mkdir -p $OUT/$task
pairname() { if [ "$1" = table ]; then echo "20231030_001_desk001_move_$2_fps30.npz"; else echo "$1_$2_fps30.npz"; fi; }
if [ $task = push ] || [ $task = pull ]; then
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $task --single r6$V=$S/single_${task}_$V/${task}_new_qpos.npz --search-a --levels 1.0 1.5 2.0 \
    --pairs-out $R/${task}_sizes_$V --hand-metrics --out $OUT/$task/eval_r6_$V.json
  $SO -B system1/t14/levels_r3.py --task $task --eval $OUT/$task/eval_r6_$V.json --prefix r6$V --solves $S --solve-tag "_$V" --refs-dir $R --out $S/levels/${task}_$V.json
  cu=mid
else
  refs=""
  for k in 0.8 0.9 1.0 1.1 1.2; do
    kk=k${k/./p}; d=$S/${task}_${kk}_$V; out=$R/${task}_${kk}_$V
    if [ ! -f $d/person1/run_record.json ] || [ ! -f $d/person2/run_record.json ]; then echo "skip $task k$k $V: a solve did not finish"; continue; fi
    [ -d $out ] || OMP_NUM_THREADS=1 $HS -B system1/t14/table_combine.py --task $task --variant-dir $d --variant new --out $out
    python3 system1/t14/scale_box_urdf.py --urdf ${URDF[$task]} --k $k --out $out/${task}_${kk}.urdf
    refs="$refs --ref r6${V}_k$k=$out/$(pairname $task new)@$k"
  done
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $task $refs --hand-metrics --out $OUT/$task/eval_r6_$V.json
  $SO -B system1/t14/levels_r3.py --task $task --eval $OUT/$task/eval_r6_$V.json --prefix r6$V --solves $S --solve-tag "_$V" --refs-dir $R --out $S/levels/${task}_$V.json
  cu=lift
fi
CUDA_VISIBLE_DEVICES= MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json OMP_NUM_THREADS=2 \
  $SO -B system1/t14/render_sizes.py --task $task --levels $S/levels/${task}_$V.json --out $OUT/figures --video-out exports_system1/T14_videos/r6_$V \
  --closeup $cu --human-panels --axes-arrows --grid-name ${task}_r6${V}_grid
echo FINISH_DONE $task $V $(date +%T)
