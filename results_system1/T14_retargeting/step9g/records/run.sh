#!/bin/bash
# T14 step 9g (Kevin 2026-11-26 01:37; task book section 12 step 9g): the three debug clips, main protocol (a, 1.0, 1.25, 1.5,
# first free above 1.5), mirror construction (frontback --fb-legs mirror), no extension (Kevin judges from the videos first)
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9g --fb-turn world --fb-legs mirror \
  --fb-human-frame target --skip-failed-up --no-extension --r1-source construction \
  --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src --parallel 6 --deadline 23:59
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
