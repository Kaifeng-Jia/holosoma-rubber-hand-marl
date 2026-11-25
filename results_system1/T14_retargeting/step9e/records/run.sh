#!/bin/bash
# T14 step 9e (Kevin 2026-11-25 15:10): frozen construction (commit bba7053) on all 22 straight clips, main protocol each
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9e --fb-turn world --fb-legs full_scopy \
  --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
  --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src --parallel 6 --deadline 23:59
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
