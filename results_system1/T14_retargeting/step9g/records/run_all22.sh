#!/bin/bash
# T14 step 9g (Kevin 2026-11-26 item 2): construction frozen as is (solver code unchanged since aec09c8), all 22 straight clips
# once with the main protocol (a, 1.0, 1.25, 1.5, first free above 1.5); the three debug clips' finished levels (same frozen
# code) are reused, as in step 9f. --parallel 5: one CPU process is kept for the re-judge and the smallbox frame-105 check
# (at most 6 processes in all).
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9g --fb-turn world --fb-legs mirror \
  --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
  --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src --parallel 5 --deadline 23:59
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
