#!/usr/bin/env bash
# Restart the laptop side after the Ubuntu shutdown of 2026-11-12 (Kevin: ~4 h off). Steps in system1/t03/RESUME_20261112.md.
#   bash system1/t03/resume_laptop.sh
# Starts the post-training pipeline (2 evaluations in parallel, finals first, --cloud-transfer --no-d-curve) only if the
# PID recorded in pipeline.pid is not a running pipeline; records the new PID. The watcher and the catch-up report are
# started separately (see the .md). Never kills anything.
set -u
cd /home/kevin/duet-system1
PT=logs_system1/post_train
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
if [ -f $PT/pipeline.pid ]; then
  P=$(cat $PT/pipeline.pid)
  if [ -d /proc/$P ] && tr '\0' ' ' < /proc/$P/cmdline | grep -q 'system1/t03/post_train.py run'; then
    echo "pipeline already running (pid $P), not starting another"; exit 1
  fi
fi
N=$(ps -eo args | grep -c '^[^ ]*python[^ ]* -B scripts/evaluate_paper_supplement.py')
[ "$N" = "0" ] || { echo "$N evaluator processes already running; check before starting the pipeline"; exit 1; }
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks_throttle_reasons.active,memory.used --format=csv
if [ -f $PT/PAUSE ]; then mv $PT/PAUSE $PT/PAUSE.released_$(date +%Y%m%dT%H%M%S); fi
echo "$(TZ=America/New_York date -Iseconds) restart after laptop shutdown (resume_laptop.sh): --slots 2 --cloud-transfer --no-d-curve" >> $PT/pipeline.log
setsid nohup $PY -B system1/t03/post_train.py run --slots 2 --poll-seconds 600 --cloud-transfer --no-d-curve >> $PT/pipeline.log 2>&1 < /dev/null &
echo $! > $PT/pipeline.pid
sleep 3
echo "pipeline pid $(cat $PT/pipeline.pid): $(tr '\0' ' ' < /proc/$(cat $PT/pipeline.pid)/cmdline)"
