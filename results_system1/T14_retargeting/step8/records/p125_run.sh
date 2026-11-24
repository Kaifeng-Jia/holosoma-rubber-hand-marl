#!/usr/bin/env bash
# T14 step 8 item 6: Push / Pull round 6 C, level m = 1.25 with the existing eval path (eval_r3.py, as finish_r6.sh), then
# the 3-D contact-position error (contact_error3d.py) of that pair. CPU only.
set -u
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
OUT=results_system1/T14_retargeting/step8/push_pull_m1p25
mkdir -p $OUT
for t in push pull; do
  OMP_NUM_THREADS=2 $SO -B system1/t14/eval_r3.py --task $t --single r6C=logs_system1/t14/r6/single_${t}_C/${t}_new_qpos.npz --levels 1.25 \
    --pairs-out exports_system1/T14_references/step8/${t}_sizes_C --hand-metrics --out $OUT/eval_r6C_${t}_m1p25.json
  echo "eval $t rc $?"
  $SO -B system1/t14/objects/ce3d_level.py --task $t --pair exports_system1/T14_references/step8/${t}_sizes_C/m1p25/${t}_m1p25_pair_fps30.npz --m 1.25 --out $OUT/contact_error_3d_${t}_m1p25.json
  echo "ce3d $t rc $?"
done
echo P125_DONE
