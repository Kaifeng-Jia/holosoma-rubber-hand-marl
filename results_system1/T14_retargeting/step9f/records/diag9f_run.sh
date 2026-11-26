#!/bin/bash
# T14 step 9f diagnostics: per-level phase diagnosis of every evaluated level that is not applicable (6 processes, CPU)
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
D=logs_system1/t14/step9f/diag9f
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
cat $D/jobs.txt | xargs -P 6 -L 1 bash -c '$0 -B system1/t14/objects/step9e_diag.py --level-dir $1 --source $2 --gates $3 >> '$D'/diag.log 2>&1; echo "$1 rc $?" >> '$D'/diag_rc.txt' $SO
echo DIAG_DONE
