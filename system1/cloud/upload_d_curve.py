#!/usr/bin/env python3
"""Make the D "completion vs iteration" curve checkpoints available to the cloud slot filler (Kevin 2026-11-11).

For every checkpoint in logs_system1/post_train/d_curve_catalog.json whose laptop evaluation is not complete:
  * V3-25-27 (from scratch): the original files on the cloud (where they were trained), sha256 checked against the catalog;
  * paper main results: copied read-only from the laptop to $S1_INPUTS/d_curve/<label>/model_<it>.pt, sha256 checked.
Writes $S1_INPUTS/d_curve/catalog.json on the cloud (label -> case, seed, iteration -> cloud path, sha256) and a local copy
logs_system1/post_train/d_curve_catalog_cloud.json. Never deletes; files already present with the right hash are kept.

    hssim_s1/bin/python -B system1/cloud/upload_d_curve.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.t03.post_train import OUT, REMOTE_V3, SCP, ssh, now  # noqa: E402

INPUTS = "/root/autodl-tmp/system1/inputs/d_curve"
HOST = "root@connect.nmb1.seetacloud.com"


def done(name):
    s = OUT / name / "summary.json"
    try:
        return bool(json.loads(s.read_text()).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def main():
    cat = json.loads((OUT / "d_curve_catalog.json").read_text())["runs"]
    out = {}
    for label, r in cat.items():
        rows = {}
        for it, ck in r["checkpoints"].items():
            name = f"d_curve/{label}/it{int(it):05d}_pm2"
            if done(name):
                continue
            src = Path(ck["path"])
            if "/exports_system1/d_curve/V3-" in str(src):
                remote = f"{REMOTE_V3}/{src.parent.name}/{src.name}"
            else:
                remote = f"{INPUTS}/{label}/{src.name}"
            rows[it] = dict(name=name, path=remote, sha256=ck["sha256"], source=str(src))
        if rows:
            out[label] = dict(case=r["case"], seed=r["seed"], checkpoints=rows)
    # what is already on the cloud
    allpaths = [x["path"] for v in out.values() for x in v["checkpoints"].values()]
    rc, txt = ssh("sha256sum " + " ".join(allpaths) + " 2>/dev/null; true", timeout=600)
    have = {p: h for h, p in (l.split() for l in txt.strip().splitlines() if len(l.split()) == 2)}
    copied = 0
    for label, v in out.items():
        need = [x for x in v["checkpoints"].values() if have.get(x["path"]) != x["sha256"]]
        if not need:
            continue
        assert all(x["path"].startswith(INPUTS) for x in need), [x["path"] for x in need if not x["path"].startswith(INPUTS)]
        ssh(f"mkdir -p {INPUTS}/{label}")
        subprocess.run(SCP + [x["source"] for x in need] + [f"{HOST}:{INPUTS}/{label}/"], check=True, timeout=3600)
        copied += len(need)
    rc, txt = ssh("sha256sum " + " ".join(allpaths), timeout=600)
    have = {p: h for h, p in (l.split() for l in txt.strip().splitlines() if len(l.split()) == 2)}
    bad = [p for v in out.values() for x in v["checkpoints"].values() for p in [x["path"]] if have.get(p) != x["sha256"]]
    if bad:
        raise RuntimeError(f"sha256 mismatch on the cloud: {bad[:5]}")
    doc = dict(created=now(), source="logs_system1/post_train/d_curve_catalog.json (laptop)", runs=out,
               n_checkpoints=len(allpaths), copied_now=copied)
    (OUT / "d_curve_catalog_cloud.json").write_text(json.dumps(doc, indent=1) + "\n")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    subprocess.run(SCP + [f.name, f"{HOST}:{INPUTS}/catalog.json"], check=True, timeout=300)
    print(now(), "checkpoints", len(allpaths), "copied", copied, "all sha256 ok", flush=True)


if __name__ == "__main__":
    main()
