#!/bin/bash
# T03 C v2 pre-training checks on the cloud ("开训前检查"), outputs under $S1_ROOT/logs/c2_smoke (not the production dir).
#  A. smoke: the exact production queue commands (run_train_queue.py --preflight 10) for cin/cout x pull/table/bucket:
#     10 iterations, F6 run_config diff, training diagnostics every 5 iterations (SYSTEM1_C_TRAIN_DIAG_EVERY, smoke only).
#  B. Pull joint (actor-updating) segment: production joint argv, resumed from the smoke critic checkpoint, 10 iterations.
#  C. iteration 0 vs scheme A: evaluator, TF32 off, 3 frozen states per task: --controller sonic vs sonic_residual with the
#     smoke model_00000.pt of each kind; compared array by array (system1/tests/t03_iter0_compare.py).
set -u
source /root/autodl-tmp/system1/s1_env.sh
cd $S1_ROOT/duet-system1
L=$S1_ROOT/logs/c2_smoke; mkdir -p $L
log(){ echo "$(TZ=America/New_York date -Iseconds) $*" | tee -a $L/checks.log; }
export SYSTEM1_C_TRAIN_DIAG_EVERY=5
log "start, git $(git rev-parse HEAD)"
# A
i=0
for spec in cin_pull:20@0 cin_table:5@0 cin_bucket:1@0 cout_pull:20@3 cout_table:5@3 cout_bucket:1@4; do
  run=${spec%@*}; gpu=${spec#*@}; i=$((i+1))
  $S1_PY -B system1/cloud/run_train_queue.py $run --out $L --python $S1_PY --gpu $gpu --local-rank $((400+i)) --preflight 10 \
     > $L/queue_${run/:/_}.log 2>&1 &
done
wait
log "A done"
# B
for k in cin cout; do
  P=$L/${k}_pull_m20kg_seed721_critic50/model_00000.pt
  argv=$(python3 -c "import json;print(' '.join(json.load(open('$L/ops/${k}_pull_m20kg_seed721_critic50.launch.json'))['argv'][2:]))")
  argv=$(echo "$argv" | sed -e "s# --iterations 50 # --iterations 10 #" -e "s# --critic-only##" -e "s#--output-dir [^ ]*#--output-dir $L/${k}_pull_joint_smoke --resume $P#")
  log "B $k: $argv"
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=41$([ $k = cin ] && echo 1 || echo 2) WORLD_SIZE=1 $S1_PY -B $argv > $L/${k}_pull_joint_smoke.log 2>&1 &
done
wait
log "B done"
# C
for c in pull table bucket; do
  st=$S1_INPUTS/states/${c}_states.json
  CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=420 $S1_PY -B scripts/evaluate_paper_supplement.py --case $c --controller sonic --states $st \
     --model-id C2iter0_${c}_A --output-dir $L/iter0/A/${c}_sonic --limit 3 > $L/iter0_${c}_A.log 2>&1
  for k in cin cout; do
    d=$( [ $c = pull ] && echo ${k}_pull_m20kg_seed721_critic50 || echo ${k}_${c}_m$([ $c = table ] && echo 5 || echo 1)kg_seed721_fresh12000 )
    mkdir -p $L/iter0/$k; ln -sfn ../A/${c}_sonic $L/iter0/$k/${c}_sonic
    CUDA_VISIBLE_DEVICES=0 LOCAL_RANK=421 $S1_PY -B scripts/evaluate_paper_supplement.py --case $c --controller sonic_residual \
       --checkpoint $L/$d/model_00000.pt --states $st --model-id C2iter0_${c}_${k} --output-dir $L/iter0/$k/${c}_sonic_residual --limit 3 \
       > $L/iter0_${c}_${k}.log 2>&1
  done
done
for k in cin cout; do
  $S1_PY -B system1/tests/t03_iter0_compare.py --root $L/iter0/$k --out $L/iter0_${k}_vs_A.json \
     --checkpoints $(for c in pull table bucket; do d=$( [ $c = pull ] && echo ${k}_pull_m20kg_seed721_critic50 || echo ${k}_${c}_m$([ $c = table ] && echo 5 || echo 1)kg_seed721_fresh12000 ); echo $c=$L/$d/model_00000.pt; done) \
     > $L/iter0_${k}_compare.log 2>&1
done
log "C done"
log "all done"
