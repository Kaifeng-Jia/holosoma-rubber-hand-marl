#!/bin/bash
# T14 step 9e side check: 9d gates regenerated with r1_source (walk_scopy, world turn; 9d's construction), then 9d's r1 support drift
# with the construction source mapping. At most 6 processes.
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
D=logs_system1/t14/step9d_r1check
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
echo "git $(git rev-parse HEAD) $(date -Iseconds)"
cat $D/gates_jobs.txt | xargs -P 6 -L 1 bash -c '$0 -B system1/t14/objects/step9_gates.py --object $2 --sequence $3 --frames $4 --levels $5 --out '$D'/$1 --fb-turn world --fb-legs walk_scopy > '$D'/gates_$1.log 2>&1; echo "$1 rc $?" >> '$D'/gates_rc.txt' $HS
$SO -B system1/t14/objects/step9d_drift_recheck.py --gates-root $D --out $D/drift_recheck.json --csv $D/drift_recheck.csv > $D/drift.log 2>&1
echo "drift rc $? $(date -Iseconds)"
echo R1CHECK_DONE
