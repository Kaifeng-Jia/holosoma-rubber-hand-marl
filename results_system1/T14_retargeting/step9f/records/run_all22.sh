#!/bin/bash
# T14 step 9f: >= 1 debug clip applicable (largeboxa 1.25 / 1.5, suitcasea 1.0 / 1.25 / 1.5) -> construction frozen at 3453e96,
# all 22 straight clips at once with the main protocol; the three debug clips' finished levels (same frozen code) are reused.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES=""
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python -B system1/t14/objects/run_step9.py --step step9f --fb-turn world --fb-legs reverse_warp \
  --fb-human-frame target --skip-failed-up --all-main --r1-source construction \
  --source-root logs_system1/t14/step9b/src logs_system1/t14/step9d/src --parallel 6 --deadline 23:59
echo "rc $? end $(TZ=America/New_York date -Iseconds)"
