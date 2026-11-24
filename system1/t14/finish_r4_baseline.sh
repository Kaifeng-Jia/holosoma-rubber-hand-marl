#!/usr/bin/env bash
# T14 round 4: the same metrics (exact measure + hand metrics) for the comparison references of one task: round 3 as packaged
# (with step 2 where it was run), the current reference and OmniRetarget as is (1.0x). Run inside tmux.
set -eu
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
W=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking
task=$1
R3=exports_system1/T14_references/r3; S3=logs_system1/t14/r3; OUT=results_system1/T14_retargeting/step4
mkdir -p $OUT/$task
pairname() { if [ "$1" = table ]; then echo "20231030_001_desk001_move_$2_fps30.npz"; else echo "$1_$2_fps30.npz"; fi; }
declare -A CUR=([table]=$W/core4d_smalltable/20231030_001_desk001_move_compact_fps30.npz [bin]=$W/core4d_bucket003_20231020_071_a1/core4d_pair_compact_fps30.npz [chair]=$W/core4d_chair021_20231020_074_a1/core4d_pair_compact_fps30.npz
  [push]=$W/rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz [pull]=$W/rubber_hand_largetable_v1/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz)
refs=""
if [ $task = push ] || [ $task = pull ]; then
  a=$(python3 -c "import json; print(json.load(open('results_system1/T14_retargeting/step3/a_levels.json'))['round3']['$task'])")
  for m in $a 1.0 1.5 2.0; do refs="$refs --ref r3_m$m=$S3/single_$task/${task}_new_qpos.npz@$m"; done
  refs="$refs --ref current=${CUR[$task]}@1.0@fps50 --ref omni_asis=$S3/single_asis_$task/${task}_omni_asis_qpos.npz@1.0"
else
  for k in 0.8 0.9 1.0 1.1 1.2; do
    kk=k${k/./p}
    if [ -f $R3/${task}_${kk}_a10/$(pairname $task new) ]; then refs="$refs --ref r3s2_k$k=$R3/${task}_${kk}_a10/$(pairname $task new)@$k"
    elif [ -f $R3/${task}_${kk}/$(pairname $task new) ]; then refs="$refs --ref r3_k$k=$R3/${task}_${kk}/$(pairname $task new)@$k"; fi
  done
  refs="$refs --ref current=${CUR[$task]}@1.0 --ref omni_asis=$R3/asis_${task}_k1p0/$(pairname $task omni_asis)@1.0"
fi
OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $task $refs --hand-metrics --out $OUT/$task/eval_baseline.json
echo BASELINE_DONE $task $(date +%T)
