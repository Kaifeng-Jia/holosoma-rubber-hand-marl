#!/bin/bash
# T03 C v2 launch on the cloud (after every pre-training check passed; Kevin 2026-11-11 14:41 relayed message):
#  1. six training queues, seed 721: C-in and C-out x Pull (20 kg) / Table (5 kg) / Bin (1 kg); production commands of
#     system1/cloud/run_train_queue.py (cin_/cout_ ids: --sonic-residual-kind in|out --c-train-diag);
#  2. the freed slots evaluate first (system1/cloud/run_eval_jobs.py, GPUs 1,3,4): C v1 at stop +/-2, constant-dq check;
#  3. the C v2 scheduler (iteration-1000 early stop, iteration-2000 kind choice, C mass sweep).
set -u
source /root/autodl-tmp/system1/s1_env.sh
cd $S1_ROOT/duet-system1
T=$S1_ROOT/logs/train; EV=$S1_ROOT/logs/evals; mkdir -p $EV
log(){ echo "$(TZ=America/New_York date -Iseconds) $*" | tee -a $T/c2_launch.log; }
log "launch, git $(git rev-parse HEAD)"
r=700
for spec in cin_pull:20@0 cout_pull:20@0 cin_table:5@0 cout_table:5@1 cin_bucket:1@3 cout_bucket:1@4; do
  run=${spec%@*}; gpu=${spec#*@}; r=$((r+1))
  nohup $S1_PY -B system1/cloud/run_train_queue.py $run --out $T --python $S1_PY --gpu $gpu --local-rank $r \
     > $T/queue_${run/:/_}_seed721.log 2>&1 < /dev/null &
  log "started $run on GPU $gpu (LOCAL_RANK $r, pid $!)"
  python3 -c "import json,os;f='$T/c2_pids.json';d=json.load(open(f)) if os.path.exists(f) else {};d['$run']=dict(pid=$!,gpu=$gpu,by='c2_launch.sh');json.dump(d,open(f,'w'),indent=1)"
  sleep 20
done
$S1_PY -B system1/cloud/make_c2_eval_jobs.py --train $T --out $EV --scales-from $S1_ROOT/logs/c2_smoke/cin_table_m5kg_seed721_fresh12000/run_config.json > $EV/jobs.json
log "eval jobs: $(python3 -c "import json;print(len(json.load(open('$EV/jobs.json'))))")"
nohup $S1_PY -B system1/cloud/run_eval_jobs.py --jobs $EV/jobs.json --out $EV --gpu-slots 1,3,4 > $EV/run_eval_jobs.log 2>&1 < /dev/null &
log "eval runner pid $!"; echo $! > $EV/run_eval_jobs.pid
nohup $S1_PY -B system1/cloud/c2_scheduler.py --train $T > $T/c2_scheduler.log 2>&1 < /dev/null &
log "scheduler pid $!"; echo $! > $T/c2_scheduler.launch.pid
