#!/bin/bash
# T04: read the object mass / inertia / COM / material back from PhysX for every scheme A mass level,
# with the same evaluator, controller and asset path as the scheme A evaluations (no episodes are run).
cd /home/kevin/duet-system1
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
ST=/home/kevin/holosoma-core4d-base/logs/PaperSupplement20260929/unified_evaluation_20261003
OUT=logs_system1/T04/A_readback; mkdir -p $OUT
for cm in pull:5 pull:10 pull:20 pull:40 table:2.5 table:5 table:10 table:20 bucket:1 bucket:2 bucket:4 bucket:8; do
  c=${cm%%:*}; m=${cm##*:}; tag=${c}_m${m/./p}kg
  extra=(--object-mass-kg $m)
  [[ ($c == pull && $m == 20) || ($c == table && $m == 5) || ($c == bucket && $m == 1) ]] && extra=()   # original mass: as in scheme A main
  echo "$(TZ=America/New_York date '+%F %T %Z') $tag ${extra[*]}"
  LOCAL_RANK=45 WORLD_SIZE=1 $PY -B scripts/evaluate_paper_supplement.py --case $c --controller sonic --device cuda:0 \
     --states $ST/${c}_states.json --model-id ReadbackA_$tag --output-dir $OUT/$tag --readback-only "${extra[@]}" > $OUT/$tag.log 2>&1
  echo "rc=$? $(test -f $OUT/$tag/object_physics_readback.json && echo ok || echo MISSING)"
done
echo "$(TZ=America/New_York date '+%F %T %Z') done"
