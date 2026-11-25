#!/usr/bin/env python3
"""T13 v4 section 5 item 3 (MuJoCo scene): Isaac scene export of one finished run, as system1/sim2sim/run_isaac_dumps.py:
exactly the argv of the run's +/-2 Isaac final (logs_system1/t13v4/finals/<run>_pm2.launch.json) with --model-id / --output-dir
replaced and ``--limit 1 --sim2sim-dump <dir>`` added (nominal episode). Records launch.json / exit.json."""
import json, subprocess, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path("/home/kevin/duet-system1")
run = sys.argv[1]
out = ROOT / "logs_system1/t13v4/isaac_dump" / run
assert not out.exists(), out
out.mkdir(parents=True)
src = ROOT / f"logs_system1/t13v4/finals/{run}_pm2.launch.json"
argv = list(json.loads(src.read_text())["argv"])
argv[2] = "8-15"
argv[argv.index("--model-id") + 1] = f"t13v4_isaac_dump__{run}"
argv[argv.index("--output-dir") + 1] = str(out / "eval")
argv += ["--limit", "1", "--sim2sim-dump", str(out / "dump")]
now = lambda: datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
(out / "launch.json").write_text(json.dumps(dict(run=run, source_final_launch=str(src.relative_to(ROOT)), argv=argv, started=now()), indent=1) + "\n")
with open(out / "stdout.log", "w") as f:
    rc = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
(out / "exit.json").write_text(json.dumps(dict(rc=rc, ended=now()), indent=1) + "\n")
print("DUMP_DONE", run, rc)
