#!/usr/bin/env bash
# T14 item 2: Bin / Chair size levels, k = 0.8 0.9 1.0 1.1 1.2, both persons; at most 8 solves at once (~1 GB each).
# Run inside tmux from the repository root.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
jobs_file=logs_system1/t14/core4d/jobs.txt
: > "$jobs_file"
for task in bin chair; do for k in 0.8 0.9 1.0 1.1 1.2; do for p in 1 2; do echo "$task $k $p" >> "$jobs_file"; done; done; done
run_one() {
  task=$1; k=$2; p=$3; kk=${k/./p}
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -B system1/t14/table_shared.py --task "$task" --variant new --person "$p" --k "$k" \
    --out "logs_system1/t14/core4d/${task}_k${kk}" > "logs_system1/t14/core4d/${task}_k${kk}_p${p}.log" 2>&1
  echo "$task $k $p exit $?"
}
export -f run_one
export PY
xargs -P 8 -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo ALL_DONE
