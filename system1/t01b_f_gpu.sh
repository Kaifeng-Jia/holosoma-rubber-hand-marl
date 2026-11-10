#!/bin/bash
# Remaining F-item GPU work (run after the drift single runs and the D regression finished).
set -uo pipefail
cd /home/kevin/duet-system1
P=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/T01b
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*"; }

# F1 online: dual env, object removed, both robots on their role references
for c in push pull kick chair bucket table; do
  [ -f $L/f1_online/$c/result.json ] && continue
  mkdir -p $L/f1_online
  log "F1 online $c"
  LOCAL_RANK=62 taskset -c 0-15 $P -B system1/sonic/f1_dual_online.py --case $c --out $L/f1_online/$c > $L/f1_online/$c.log 2>&1
done

# F2 acceptance on GPU: scheme A manifest (Kp, frozen scale) + PD target == q*
log "F2 smoke (scheme A, pull, 3 episodes)"
LOCAL_RANK=63 taskset -c 0-15 $P -B scripts/evaluate_paper_supplement.py --case pull --controller sonic \
  --states /home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/pull_states.json \
  --model-id F2_smoke_schemeA_pull --output-dir $L/f2_smoke_schemeA_pull --limit 3 > $L/f2_smoke_schemeA_pull.log 2>&1

# F6 preflights: exact cloud commands for 10 iterations (run_config + seconds/iteration)
log "F6 preflights"
$P -B system1/cloud/run_train_queue.py d_pull:40 d_table:10 d_table:20 d_bucket:2 d_bucket:4 d_bucket:8 \
   c_pull:20 c_table:5 c_bucket:1 --out $L/preflight --python $P --gpu 0 --local-rank 64 --cpus 0-15 --preflight 10 > $L/preflight.log 2>&1

# T03 smoke: C checkpoint at iteration 0 (zero output layer) == scheme A on the same states
for spec in "pull:c_pull_m20kg_seed721_critic50" "table:c_table_m5kg_seed721_fresh12000" "bucket:c_bucket_m1kg_seed721_fresh12000"; do
  c=${spec%%:*}; d=${spec##*:}
  st=/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003/${c}_states.json
  for ctl in sonic sonic_residual; do
    extra=""; [ $ctl = sonic_residual ] && extra="--checkpoint $L/preflight/$d/model_00000.pt"
    log "T03 iteration-0 check $c $ctl"
    LOCAL_RANK=65 taskset -c 0-15 $P -B scripts/evaluate_paper_supplement.py --case $c --controller $ctl $extra --states $st \
       --model-id C0check_${c}_${ctl} --output-dir $L/c0check/${c}_${ctl} --limit 3 > $L/c0check_${c}_${ctl}.log 2>&1
  done
done
log "done"
