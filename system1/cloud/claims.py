"""Evaluation job claims shared by the cloud slot filler and the laptop pipeline (Kevin 2026-11-11: no idle GPU; the
laptop takes what the cloud cannot). A job is run by whoever creates its claim directory first (mkdir is atomic):
  /root/autodl-tmp/system1/logs/evals/claims/<job name with '/' -> '__'>/owner.json
Claims are never removed; a job whose run fails stays with its owner (the owner's retry rule applies).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

CLAIMS = "/root/autodl-tmp/system1/logs/evals/claims"
NY = ZoneInfo("America/New_York")


def safe(name: str) -> str:
    return name.replace("/", "__")


def _owner(owner, extra):
    return json.dumps(dict(owner=owner, time=datetime.now(NY).isoformat(timespec="seconds"), **(extra or {})))


def claim_local_fs(name: str, owner: str, extra: dict | None = None) -> bool:
    """On the cloud machine itself."""
    d = Path(CLAIMS) / safe(name)
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        try:
            return json.loads((d / "owner.json").read_text()).get("owner") == owner
        except (OSError, json.JSONDecodeError):
            return False
    (d / "owner.json").write_text(_owner(owner, extra))
    return True


def claimed_local_fs() -> set[str]:
    p = Path(CLAIMS)
    return {x.name for x in p.iterdir()} if p.exists() else set()


def claim_via_ssh(ssh_cmd: list[str], name: str, owner: str, extra: dict | None = None) -> bool | None:
    """From the laptop. True = claimed, False = someone else has it, None = cloud unreachable (do not run)."""
    d = f"{CLAIMS}/{safe(name)}"
    payload = _owner(owner, extra).replace("'", "")
    cmd = (f"mkdir -p {CLAIMS} && if mkdir {d} 2>/dev/null; then echo '{payload}' > {d}/owner.json; echo CLAIMED; "
           f"else sleep 1; echo TAKEN; cat {d}/owner.json; fi")
    try:
        r = subprocess.run(ssh_cmd + [cmd], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    if "CLAIMED" in r.stdout:
        return True
    try:  # an earlier claim of the same owner (e.g. registered before a restart) is ours
        return json.loads(r.stdout.split("TAKEN", 1)[1].strip()).get("owner") == owner
    except (IndexError, json.JSONDecodeError):
        return False
