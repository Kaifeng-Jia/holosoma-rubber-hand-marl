#!/bin/bash
# T14 step 9g item 2 (Kevin 2026-11-26): after the 22-clip run. CPU only, at most 6 processes.
#  1. re-judge every evaluated level with the non-hand penetration on the actual collision shapes (nonhand_exact.py), 6 workers;
#  2. per-level phase diagnosis (step9e_diag.py, 6 processes) of every evaluated level that is not applicable (old verdict), as in 9f;
#  3. tables (step9e_tables.py, step9g_prediag_table.py, step9g_feet_ik_table.py).
cd /home/kevin/duet-system1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
SO=/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python
HR=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python
L=logs_system1/t14/step9g
R=results_system1/T14_retargeting/step9g
O=logs_system1/t14/rejudge_exact
D=$L/diag9g_all22
echo "git $(git rev-parse HEAD) start $(TZ=America/New_York date -Iseconds)"
$SO -B system1/t14/objects/rejudge_nonhand_exact.py --set step9g --driver $L/driver.json --out $O/step9g --workers 6 \
  --results $O/tables --name step9g_all22_nonhand_exact > $O/step9g.log 2>&1; echo "rejudge rc $? $(TZ=America/New_York date -Iseconds)"
$SO -B - > $D/jobs.txt <<'PY'
import json, shlex
from pathlib import Path
L = Path("logs_system1/t14/step9g")
for clip in json.loads((L / "driver.json").read_text())["summary"]:
    for ev in sorted((L / clip).glob("m*/eval.json")):
        e = json.loads(ev.read_text())
        if e["applicable"]:
            continue
        t = shlex.split(e["command"]); src = t[t.index("--source") + 1]
        print(ev.parent, src, L / clip / f"gates_{ev.parent.name}.json")
PY
echo "diag jobs $(wc -l < $D/jobs.txt)"
cat $D/jobs.txt | xargs -P 6 -L 1 bash -c '$0 -B system1/t14/objects/step9e_diag.py --level-dir $1 --source $2 --gates $3 >> '$D'/diag.log 2>&1; echo "$1 rc $?" >> '$D'/diag_rc.txt' $SO
echo "diag done $(TZ=America/New_York date -Iseconds)"
$SO -B system1/t14/objects/step9g_prediag_table.py --logdir $L --out $L/prediag_all22.csv > $L/prediag_all22.log 2>&1; echo "prediag rc $?"
$HR -B system1/t14/objects/step9g_feet_ik_table.py --logdir $L --out $L/feet_ik_all22.csv > $L/feet_ik_all22.log 2>&1; echo "feet_ik rc $?"
echo "end $(TZ=America/New_York date -Iseconds)"
