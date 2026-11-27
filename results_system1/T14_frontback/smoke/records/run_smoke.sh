#!/bin/bash
# T14 front-back training task (Kevin 2026-11-26), local smoke on the laptop 5070 before "可以开卡" (as the T13 v4 reference
# change, results_system1/T13v4_prep/t14ref): 1. in-simulation checks, 8 environments (observation v1 vs definitions, contact
# replay vs definitions; noise off twice, noise on); 2. 200-iteration training with the exact d_frontback cloud command lines
# (cloud_smoke.py; 2048 environments: the 8 GB 5070 cannot hold 4096) and the cloud validation command on its last checkpoint;
# 3. T10 gate before training, on the smoke checkpoint: run dir logs_system1/t13v4/runs/d_frontback_smoke, then post_run.py
# steps 2, 3, 5, 6 unchanged (Isaac +/-2 / +/-8 finals for the reset states, Isaac scene export of the nominal episode, actor
# ONNX, check_sim2sim_v1.py: initial states and observation v1 per dimension <= 1e-4); 4. MuJoCo closed loop +/-2 (pipeline
# health only, a 200-iteration policy). Every Isaac launch waits until MemAvailable >= 17 GB (the T13 v4 checks' gate).
set -u
cd /home/kevin/duet-system1
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
SIM=/home/kevin/.holosoma_deps/miniconda3/envs/sim2sim_s1/bin/python
SM=logs_system1/t14/frontback_train/smoke
FB=src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/system1_t14fb/table/t14_reference.json
LAB=system1/t14/frontback_train/labels/frontback_m1p00_labels.npz
avail() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
gate() { until [ "$(avail)" -ge "$1" ]; do sleep 30; done; echo "launch $2 avail $(avail) GB $(TZ=America/New_York date -Iseconds)"; }
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
git status --short system1 scripts src | head
mkdir -p $SM/checks
for x in noise_off:off noise_off_r2:off noise_on:on; do
  n=${x%%:*}; z=${x##*:}
  gate 17 check_$n
  $PY -B scripts/train_core4d_smalltable.py --experiment smalltable5kg_A --reward-variant bucket_A_no_rel_no_height --obs-v1 --critic-v1 \
    --contact-v1 $LAB --t14-reference $FB --num-envs 8 --seed 721 --obs-v1-noise $z --t13v4-check $SM/checks/$n --output-dir $SM/checks/${n}_rundir \
    > $SM/checks/$n.log 2>&1; echo "check $n rc $? $(TZ=America/New_York date -Iseconds)"
done
gate 17 train_smoke
S1_PY=$PY python3 system1/t13v4/cloud_smoke.py --run d_frontback --num-envs 2048 --iterations 200 --out $SM/train > $SM/train_smoke.log 2>&1
echo "train smoke rc $? $(TZ=America/New_York date -Iseconds)"; tail -1 $SM/train_smoke.log
R=logs_system1/t13v4/runs/d_frontback_smoke
python3 - <<'PY'
import json, shutil
from pathlib import Path
sm = Path("logs_system1/t14/frontback_train/smoke/train/d_frontback/smoke.json")
rec = json.loads(sm.read_text())
assert rec["passed"], "train smoke failed: no T10 gate"
R = Path("logs_system1/t13v4/runs/d_frontback_smoke")
(R / "train").mkdir(parents=True, exist_ok=False)
(R / "validation").mkdir()
ck = Path(rec["stages"][-1]["last_checkpoint"])
it = int(ck.stem.split("_")[1])
shutil.copy2(ck, R / "train" / ck.name)
(R / "validation" / f"it{it:05d}.launch.json").write_text(json.dumps(dict(argv=rec["validation"]["argv"], note="cloud_smoke validation argv (d_frontback smoke)"), indent=1))
(R / "launch.json").write_text(json.dumps(dict(stages=rec["plan"]["stages"], note="d_frontback cloud plan (2048 environments) of the smoke"), indent=1))
(R / "decision.json").write_text(json.dumps(dict(run="d_frontback_smoke", status="smoke", final_checkpoint_iteration=it, final_checkpoint_counted_iteration=it,
                                                 note=f"pre-training T10 gate on the 200-iteration smoke checkpoint {ck}"), indent=1))
print("smoke run dir", R, it)
PY
gate 22 finals_smoke
python3 system1/t13v4/run_finals_isaac.py --run d_frontback_smoke > $SM/finals.log 2>&1; echo "finals rc $? $(TZ=America/New_York date -Iseconds)"
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
