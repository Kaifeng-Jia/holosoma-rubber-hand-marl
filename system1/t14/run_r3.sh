#!/usr/bin/env bash
# T14 round 3 (section 8 item 1): all solves with the simulator scene (kinematics, robot / object collision shapes) and the
# prior on root / legs / waist only; plus OmniRetarget as is at 1.0x for all five tasks (same MuJoCo 4.0 environment).
# Interpreter: hsretargeting_s1 = clone of hsretargeting with mujoco 4.0.0 (3.10.0's mj_geomDistance is wrong for some pairs).
# At most 8 at once (Kevin's memory rule: 1.6 GB each). Run inside tmux from the repository root.
set -u
PY=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
D=logs_system1/t14/r3
mkdir -p $D
jobs_file=$D/jobs.txt
: > "$jobs_file"
for task in table bin chair; do for k in 0.8 0.9 1.0 1.1 1.2; do for p in 1 2; do echo "core4d $task $k $p" >> "$jobs_file"; done; done; done
for task in push pull; do echo "single $task" >> "$jobs_file"; done
for task in table bin chair; do for p in 1 2; do echo "asis $task 1.0 $p" >> "$jobs_file"; done; done
for task in push pull; do echo "single_asis $task" >> "$jobs_file"; done
run_one() {
  kind=$1; task=$2
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  case $kind in
    core4d) k=$3; p=$4; kk=${k/./p}
      $PY -B system1/t14/table_shared.py --task $task --variant new --person $p --k $k --sim-scene --prior body \
        --out $D/${task}_k${kk} > $D/${task}_k${kk}_p${p}.log 2>&1 ;;
    asis) k=$3; p=$4; kk=${k/./p}
      $PY -B system1/t14/table_shared.py --task $task --variant omni_asis --person $p --k $k \
        --out $D/asis_${task}_k${kk} > $D/asis_${task}_k${kk}_p${p}.log 2>&1 ;;
    single) $PY -B system1/t14/single_shared.py --task $task --sim-scene --prior body --out $D/single_$task > $D/single_$task.log 2>&1 ;;
    single_asis) $PY -B system1/t14/single_shared.py --task $task --variant omni_asis --out $D/single_asis_$task > $D/single_asis_$task.log 2>&1 ;;
  esac
  echo "$* exit $? $(date +%T)"
}
export -f run_one
export PY D
xargs -P ${P:-8} -L 1 bash -c 'run_one "$@"' _ < "$jobs_file"
echo ALL_DONE $(date +%T)
