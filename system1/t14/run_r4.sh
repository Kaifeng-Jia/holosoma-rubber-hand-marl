#!/usr/bin/env bash
# T14 round 4 (section 9): every task and level, versions A (hand landmarks) and B (A + palm toward the surface); the
# anchor term (step 2, weight 10) is part of the method for all tasks. Simulator scene, prior on root / legs / waist.
# Interpreter hsretargeting_s1 (mujoco 4.0.0 + the int joint-type fix). At most $P solves at once. Run inside tmux.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
D=logs_system1/t14/r4
mkdir -p $D
jobs_file=$D/jobs.txt
: > "$jobs_file"
for V in A B; do
  for task in table bin chair; do for k in 0.8 0.9 1.0 1.1 1.2; do for p in 1 2; do echo "core4d $V $task $k $p" >> "$jobs_file"; done; done; done
  for task in push pull; do echo "single $V $task" >> "$jobs_file"; done
done
run_one() {
  kind=$1; V=$2; task=$3
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  flags="--sim-scene --prior body --anchor-weight 10 --landmarks"
  [ "$V" = B ] && flags="$flags --palm-orient-weight 5"
  case $kind in
    core4d) k=$4; p=$5; kk=${k/./p}
      $PY -B system1/t14/table_shared.py --task $task --variant new --person $p --k $k $flags --out $D/${task}_k${kk}_$V > $D/${task}_k${kk}_${V}_p${p}.log 2>&1 ;;
    single) $PY -B system1/t14/single_shared.py --task $task $flags --out $D/single_${task}_$V > $D/single_${task}_$V.log 2>&1 ;;
  esac
  echo "$* exit $? $(date +%T)"
}
export -f run_one
export PY D
xargs -P ${P:-6} -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo ALL_DONE $(date +%T)
