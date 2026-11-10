#!/bin/bash
# Cloud reproduction check (T01b decision 1): Table Pulling paper model seed721, before any training.
#  (1) +/-2: 51 episodes vs V3 episodes.csv (that table was produced on the Ubuntu laptop GPU);
#  (2) +/-8: 51 episodes vs V3 episodes.csv (produced on the V3 cloud RTX 4090s).
# Gate (preregistered in T01b): episode-by-episode identical. If (1) differs only because the
# GPU model differs while (2) matches, report both and wait for Kevin before training.
set -uo pipefail
source "${S1_ROOT:-/root/autodl-tmp/system1}/s1_env.sh"
cd "$S1_ROOT/duet-system1"
OUT=$S1_ROOT/logs/repro_pull_$(date +%Y%m%dT%H%M%S)
mkdir -p "$OUT"
CK=$S1_INPUTS/repro/Pull-baseline_seed721/model_08050.pt
for lvl in 2 8; do
  st=$S1_INPUTS/states/pull_states.json; [[ $lvl == 8 ]] && st=$S1_INPUTS/states/pull_pm8.json
  LOCAL_RANK=71 WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=${GPU:-0} $S1_PY -B scripts/evaluate_paper_supplement.py --case pull \
     --model-id Pull-baseline_seed721 --checkpoint "$CK" --states "$st" --output-dir "$OUT/pm$lvl" > "$OUT/pm$lvl.log" 2>&1
  $S1_PY -B system1/repro/compare_eval_repro.py --new-dir "$OUT/pm$lvl" --orig-dir "$OUT/pm$lvl" \
     --v3-episodes "$S1_INPUTS/v3/episodes.csv" --case pull --seed 721 --level $lvl \
     --out-csv "$OUT/compare_pm$lvl.csv" --out-json "$OUT/compare_pm$lvl.json" | tail -20
done
nvidia-smi --query-gpu=name,driver_version --format=csv > "$OUT/gpu.txt"
echo "results in $OUT"
