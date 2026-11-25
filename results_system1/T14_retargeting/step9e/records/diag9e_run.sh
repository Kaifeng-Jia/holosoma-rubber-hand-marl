#!/bin/bash
# T14 step 9e diagnostics: jump sizes (1 process) + per-level phase diagnosis (5 processes); 6 processes in all
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
D=logs_system1/t14/step9e/diag9e
HS=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
$HS -B system1/t14/objects/step9e_jump.py --out $D/jump.json > $D/jump.log 2>&1 &
J=$!
cat $D/jobs.txt | xargs -P 5 -L 1 bash -c '$0 -B system1/t14/objects/step9e_diag.py --level-dir $1 --source $2 --gates $3 >> '$D'/diag.log 2>&1; echo "$1 rc $?" >> '$D'/diag_rc.txt' $SO
wait $J; echo "jump rc $?" >> $D/diag_rc.txt
echo DIAG_DONE
