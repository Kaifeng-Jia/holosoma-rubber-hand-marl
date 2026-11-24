#!/usr/bin/env bash
# T14 step 8: after the main driver: redo the crashed suitcase-c solve (PyTorch clock assert at import, rc -6) and that
# sequence's evaluations / verdict / levels with its own log, then the pictures and tables.
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
until python3 -c "import json,sys; sys.exit(0 if 'done' in json.load(open('logs_system1/t14/step8/driver.json')) else 1)"; do sleep 30; done
$SO -B system1/t14/objects/run_step8.py --parallel 6 --only suitcasec --log-name driver_rerun_suitcasec.json > logs_system1/t14/step8/driver_rerun_suitcasec.log 2>&1
echo "rerun rc $?"
python3 -B system1/t14/objects/post_step8.py > logs_system1/t14/step8/post.log 2>&1
echo "post rc $?"
echo AFTER_DONE
