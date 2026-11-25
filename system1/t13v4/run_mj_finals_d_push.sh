#!/bin/bash
# T13 v4 section 5 item 3: MuJoCo closed-loop evaluation of d_push (T10 setting: Isaac PD of this run's scene export), +/-2 and +/-8,
# observation v1 noise off, the same frozen test states as the Isaac finals. One single-threaded process per level.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
PY=/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python
O=logs_system1/t13v4/mj_finals
for L in 2 8; do
  C=$([ $L = 2 ] && echo 0-3 || echo 4-7)
  ( taskset -c $C $PY -B system1/sim2sim/mj_eval.py --task push --scheme D --seed 721 --level $L \
      --dump logs_system1/t13v4/isaac_dump/d_push/dump --states system1/t13v4/states_t14r6C/push_pm${L}_states.json --obs-v1 \
      --onnx exports_system1/t13v4/onnx/d_push_t13v4.onnx --checkpoint logs_system1/t13v4/runs/d_push/train/model_06025.pt \
      --out $O/d_push_pm$L > $O/d_push_pm$L.stdout.log 2>&1; echo "pm$L rc $? $(date -Is)" >> $O/d_push_rc.txt ) &
done
wait
echo MJ_DONE
