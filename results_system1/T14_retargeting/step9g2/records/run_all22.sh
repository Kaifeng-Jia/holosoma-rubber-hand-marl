#!/bin/bash
# T14 step 9g second fix (Kevin 2026-11-26, received 10:29), item 3: the 22 straight debug clips with the 9g construction and the
# walked distance = mean of the pelvis and both ankles along the walking direction (--fb-walk-distance pelvis_feet; nothing else
# changed); main protocol as step 9g (a, 1.0, 1.25, 1.5, first free above 1.5). No level reused from 9g (the partner's legs
# change). Source solves reused as 9g (--source-root). CPU only, at most 6 processes (memory gate: available - need >= 8 GB);
# --t13-yield 2: while a T13 local evaluation is running, at most 2 of these workers. Cores 16-31 (T13 MuJoCo steps use 0-15).
# No rendering here (--no-render): step9g_render_both.py afterwards, at most 2 renders.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
git status --short system1 src | head -20
taskset -c 16-31 /home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9g2 --fb-turn world --fb-legs mirror \
  --fb-walk-distance pelvis_feet --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
  --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src --parallel 6 --deadline 23:59 --no-render --t13-yield 2
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
