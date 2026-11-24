#!/usr/bin/env bash
# T14 step 8: role-0 source solves at r = 1 (box family), 6 in parallel (measured 1.9 GB each).
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
D=logs_system1/t14/step8/src
pids=()
for spec in smallbox:sub10_smallbox_007:smallboxb largebox:sub3_largebox_024:largeboxb suitcase:sub1_suitcase_065:suitcaseb smallbox:sub7_smallbox_027:smallboxc largebox:sub7_largebox_027:largeboxc suitcase:sub14_suitcase_016:suitcasec; do
  IFS=: read o s t <<< "$spec"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 $HS -B system1/t14/objects/role_solve.py --family box --object $o --sequence $s --mode source --tag $t --out $D/$t > $D/$t.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; echo "pid $p rc $?"; done
echo SRC_DONE
