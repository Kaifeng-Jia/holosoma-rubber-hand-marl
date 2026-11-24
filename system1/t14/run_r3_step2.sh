#!/usr/bin/env bash
# T14 round 3, step 2 (section 8 item 2): levels whose round-3 contact retention is below 0.99 are re-solved with the CoCarry
# anchor term (weight 10), everything else as round 3. Args: "task:k" ...; waits for the round-3 queue (run.log ALL_DONE).
# At most $P solves at once (default 6: leaves RAM for the C1 Isaac dumps). Run inside tmux.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
D=logs_system1/t14/r3
P=${P:-6}
until grep -q ALL_DONE $D/run.log; do sleep 10; done
jobs_file=$D/jobs_a10_$(date +%H%M%S).txt
: > "$jobs_file"
for tk in "$@"; do task=${tk%%:*}; k=${tk##*:}
  if [ "$k" = single ]; then echo "$task single 0" >> "$jobs_file"; else for p in 1 2; do echo "$task $k $p" >> "$jobs_file"; done; fi
done
run_one() {
  task=$1; k=$2; p=$3; kk=${k/./p}
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  if [ "$k" = single ]; then
    $PY -B system1/t14/single_shared.py --task $task --sim-scene --prior body --anchor-weight 10 --out $D/single_${task}_a10 > $D/single_${task}_a10.log 2>&1
  else
    $PY -B system1/t14/table_shared.py --task $task --variant new --person $p --k $k --sim-scene --prior body \
      --anchor-weight 10 --out $D/${task}_k${kk}_a10 > $D/${task}_k${kk}_a10_p${p}.log 2>&1
  fi
  echo "$task $k $p a10 exit $? $(date +%T)"
}
export -f run_one
export PY D
xargs -P $P -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo A10_DONE $(date +%T)
