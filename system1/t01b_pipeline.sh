#!/bin/bash
# T01b local GPU pipeline after F1-F6: drift analysis -> (v) gate -> scheme A formal -> hip_pitch
# sensitivity -> T04 scheme A mass levels. GPU PhysX, TF32 off, frozen scales (defaults of the code).
set -uo pipefail
cd /home/kevin/duet-system1
P=/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python
L=logs_system1/T01b
log() { echo "$(TZ=America/New_York date '+%F %T %Z') $*"; }

# 1. drift analysis (i)(ii)(iv)(v) on the 14 single-robot runs
SEGS="sonic_sample_A001=logs_system1/T01/sonic/refs/sonic_sample_A001 sonic_sample_A001_M=logs_system1/T01/sonic/refs/sonic_sample_A001_M"
for t in push pull kick chair bucket table; do for k in 0 1; do SEGS="$SEGS ${t}_agent${k}=$L/refs/${t}_agent${k}"; done; done
CTRL=""
for t in push pull kick; do for k in 0 1; do d=$L/drift/control_${t}_agent${k}; [ -f $d/result.json ] && CTRL="$CTRL ${t}_agent${k}=$d=$L/refs/${t}_agent${k}"; done; done
log "drift analysis"
$P -B system1/sonic/drift_analysis.py --runs $L/drift --segments $SEGS ${CTRL:+--control $CTRL} --out results_system1/T01b/drift > $L/drift_analysis.log 2>&1 || { log "drift analysis failed"; exit 1; }
if $P -c "import json,sys;sys.exit(0 if json.load(open('results_system1/T01b/drift/checks_v.json'))['integration_problem_found'] else 1)"; then
  log "STOP: (v) flagged a possible integration problem; scheme A not started"; exit 2
fi

# 2. scheme A formal (6 tasks x +/-2, +/-8, nominal included)
log "scheme A main"
LOCAL_RANK=41 $P -B system1/scheme_a_queue.py --device cuda:0 --cpus 0-15 --out $L/schemeA_main > $L/schemeA_main.queue.log 2>&1 || { log "scheme A main incomplete"; exit 3; }
# 3. hip_pitch sensitivity (SONIC actuator on hip_pitch), Push/Pull/Kick, both levels
log "scheme A hip_pitch sensitivity"
LOCAL_RANK=42 $P -B system1/scheme_a_queue.py --device cuda:0 --cpus 0-15 --cases push,pull,kick --out $L/schemeA_hip_pitch_sonic \
   --extra="--sonic-hip-pitch-actuator sonic" > $L/schemeA_hip.queue.log 2>&1 || { log "sensitivity incomplete"; exit 4; }
# 4. T04 scheme A: every non-original mass level (original masses are the main runs above)
for spec in pull:5 pull:10 pull:40 table:2.5 table:10 table:20 bucket:2 bucket:4 bucket:8; do
  c=${spec%%:*}; m=${spec##*:}
  log "T04 scheme A $c ${m} kg"
  LOCAL_RANK=43 $P -B system1/scheme_a_queue.py --device cuda:0 --cpus 0-15 --cases $c --out logs_system1/T04/A/${c}_m${m/./p}kg \
     --extra="--object-mass-kg $m" > logs_system1/T04/A_${c}_${m}.queue.log 2>&1 || { log "T04 A $c $m incomplete"; exit 5; }
done
log "pipeline done"
