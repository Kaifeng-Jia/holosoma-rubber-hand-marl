#!/bin/bash
# T14 final test after step 9g2 (Kevin 2026-11-26, received 2026-11-27 00:10): the frozen 9g2 construction (three rules + walked
# distance = mean of the pelvis and both ankles along the walking direction) on every straight clip of plasticbox and smalltable
# (class_a_turns.csv of step9_turns.py --objects plasticbox smalltable: |heading - walking direction| < 45 deg at every frame).
# Same levels (a, 1.0, 1.25, 1.5, first free above 1.5), same arguments as step 9g2's run_all22.sh except: --turns (the clip list),
# no --source-root (new clips: source solves under logs_system1/t14/step9_final/src), no --t13-yield (no T13 local evaluation).
# One run; CPU only (CUDA_VISIBLE_DEVICES=""), at most 6 processes (memory gate: available - need >= 8 GB), cores 16-31.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
git status --short system1 src | head -20
taskset -c 16-31 /home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9_final --fb-turn world --fb-legs mirror \
  --fb-walk-distance pelvis_feet --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
  --turns results_system1/T14_retargeting/step9_final/class_a_turns.csv --parallel 6 --deadline 23:59 --no-render
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
