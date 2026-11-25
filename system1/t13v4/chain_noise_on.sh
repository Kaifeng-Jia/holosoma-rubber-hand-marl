#!/bin/bash
# start the noise-on group of d_push only after both noise-off finals have written their exit records (GPU / CPU slots shared)
cd /home/kevin/duet-system1
F=logs_system1/t13v4/finals
until [ -f $F/d_push_pm2.exit.json ] && [ -f $F/d_push_pm8.exit.json ]; do sleep 60; done
echo "noise-off finals done $(date -Is); starting noise on"
python3 system1/t13v4/run_finals_isaac.py --run d_push --noise on --out $F
