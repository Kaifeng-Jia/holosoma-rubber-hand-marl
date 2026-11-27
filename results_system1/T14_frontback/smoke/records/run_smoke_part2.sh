#!/bin/bash
# Continuation of run_smoke.sh from step 3 (Kevin 2026-11-26 front-back task). The first script waited at its own finals gate
# (MemAvailable >= 22 GB, two Isaac processes at once) while the laptop had 21 GB; stopped by PID 906105 at 01:5x EST.
# Same commands, except run_finals_isaac.py is called once per level (--levels 2, then --levels 8), each behind the 17 GB gate.
set -u
cd /home/kevin/duet-system1
SIM=/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python
SM=logs_system1/t14/frontback_train/smoke
R=logs_system1/t13v4/runs/d_frontback_smoke
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "launch $2 avail $(avail) GB $(TZ=America/New_York date -Iseconds)"; }
echo "part2 git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
for lv in 2 8; do
  gate 17 finals_smoke_pm$lv
  python3 system1/t13v4/run_finals_isaac.py --run d_frontback_smoke --levels $lv > $SM/finals_pm$lv.log 2>&1; echo "finals pm$lv rc $? $(TZ=America/New_York date -Iseconds)"
done
gate 17 dump_smoke
python3 system1/t13v4/run_isaac_dump.py d_frontback_smoke > $SM/dump.log 2>&1; echo "dump rc $? $(TZ=America/New_York date -Iseconds)"
CK=$(ls $R/train/model_*.pt | tail -1)
CUDA_VISIBLE_DEVICES= $SIM -B system1/sim2sim/onnx_check.py --out logs_system1/t13v4/sim2sim/onnx_d_frontback_smoke --onnx-dir exports_system1/t13v4/onnx \
  --extra-checkpoint "D,table,721,d_frontback_smoke_t13v4,$CK,v1" > $SM/onnx.log 2>&1; echo "onnx rc $?"
CUDA_VISIBLE_DEVICES= $SIM -B system1/t13v4/check_sim2sim_v1.py --run d_frontback_smoke --task table --onnx exports_system1/t13v4/onnx/d_frontback_smoke_t13v4.onnx \
  --out logs_system1/t13v4/sim2sim/checks_d_frontback_smoke > $SM/checks_sim2sim.log 2>&1; echo "sim2sim checks rc $? $(TZ=America/New_York date -Iseconds)"
tail -12 $SM/checks_sim2sim.log
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 0-3 $SIM -B system1/sim2sim/mj_eval.py --task table --scheme D --seed 721 --level 2 \
  --dump logs_system1/t13v4/isaac_dump/d_frontback_smoke/dump --states system1/t14/frontback_train/states/table_pm2_states.json --obs-v1 \
  --onnx exports_system1/t13v4/onnx/d_frontback_smoke_t13v4.onnx --checkpoint $CK --out logs_system1/t13v4/mj_finals/d_frontback_smoke_pm2 > $SM/mujoco_pm2.log 2>&1
echo "mujoco pm2 rc $? end $(TZ=America/New_York date -Iseconds)"
