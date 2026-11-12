#!/usr/bin/env python3
"""Smoke test for the cloud -> laptop hand-over (Kevin 2026-11-12: cloud evaluations end 2 h after the last training,
unfinished ones go to the laptop): c2_scheduler eval_minutes / est_minutes / transfer_list, post_train transfer_jobs /
handover_names. No GPU, no ssh.

    hssim_s1/bin/python -B system1/tests/test_cloud_handover.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import system1.cloud.c2_scheduler as cs  # noqa: E402
import system1.t03.post_train as pt  # noqa: E402


def h(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        cs.EVALS = t / "evals"
        ops = cs.EVALS / "ops2"
        ops.mkdir(parents=True)
        # durations: only completed evaluations launched on a GPU without training count; last 5 per case
        def rec(name, case, start, end, before, completed=True):
            (ops / f"{name}.launch.json").write_text(json.dumps(dict(argv=["py", "e.py", "--case", case], started=start, gpu_procs_before=before)))
            (ops / f"{name}.exit.json").write_text(json.dumps(dict(completed=completed, ended=end)))
        rec("a", "table", "2026-11-13T01:00:00-05:00", "2026-11-13T02:00:00-05:00", ["eval x"])
        rec("b", "table", "2026-11-13T01:00:00-05:00", "2026-11-13T01:40:00-05:00", [])
        rec("c", "table", "2026-11-13T01:00:00-05:00", "2026-11-13T03:00:00-05:00", ["train cin_table"])  # next to a training
        rec("d", "table", "2026-11-13T01:00:00-05:00", "2026-11-13T05:00:00-05:00", [], completed=False)
        m = cs.eval_minutes(ops)
        assert m == {"table": 50.0}, m
        job = lambda case: dict(argv=["e.py", "--case", case])  # noqa: E731
        assert cs.est_minutes(job("table"), m) == 50.0 and cs.est_minutes(job("pull"), m) == cs.DEFAULT_EVAL_MIN["pull"]

        # transfer list: not complete and not owned by the laptop
        ck, st = t / "model_5000.pt", t / "table_states.json"
        ck.write_bytes(b"ck")
        st.write_text("{}")
        def j(name, kind, owner=None, complete=False, sha=None):
            if owner:
                (cs.EVALS / "claims" / cs.safe(name)).mkdir(parents=True)
                (cs.EVALS / "claims" / cs.safe(name) / "owner.json").write_text(json.dumps(dict(owner=owner)))
            if complete:
                (cs.EVALS / name).mkdir(parents=True)
                (cs.EVALS / name / "summary.json").write_text(json.dumps(dict(complete=True)))
            argv = ["scripts/evaluate_paper_supplement.py", "--case", "table", "--controller", "sonic_residual", "--checkpoint", str(ck),
                    "--states", str(st), "--model-id", "x", "--output-dir", str(cs.EVALS / name), "--object-mass-kg", "10"]
            return dict(name=name, argv=argv, sha256=sha, meta=dict(kind=kind, run="cin_table_m10kg_seed721_fresh12000", iteration=5000, level=2))
        jobs = [j("c2_curve/r/it05000_pm2", "c2"), j("c2_curve/r/it06000_pm2", "c2", owner="cloud"),
                j("c2_curve/r/it12000_pm2", "c2", owner="laptop"), j("c2_curve/r/it04000_pm2", "c2", owner="cloud", complete=True),
                j("d_curve/D/it02000_pm2", "d_curve", owner="cloud", sha="abc")]
        tl = cs.transfer_list(jobs, {"c2_curve/r/it06000_pm2": 1})
        names = [r["name"] for r in tl]
        assert names == ["c2_curve/r/it05000_pm2", "c2_curve/r/it06000_pm2", "d_curve/D/it02000_pm2"], names
        r0 = tl[0]
        assert r0["checkpoint_sha256"] == h(ck) and r0["states_sha256"] == h(st) and r0["object_mass_kg"] == 10.0 and r0["case"] == "table"
        assert tl[1]["cloud_owner"] == "cloud" and tl[1]["cloud_failures"] == 1 and tl[2]["checkpoint_sha256"] == "abc"
        assert isinstance(cs.training_alive(), bool)

        # laptop side
        pt.TRANSFER, pt.TRANSFER_FETCHED = t / "cloud_transfer.json", t / "cloud_transfer_fetched.json"
        assert pt.transfer_jobs() == [] and pt.handover_names() == {}
        pt.TRANSFER.write_text(json.dumps(dict(jobs=tl)))
        assert pt.transfer_jobs() == []  # nothing fetched yet
        pt.TRANSFER_FETCHED.write_text(json.dumps({"c2_curve/r/it05000_pm2": dict(checkpoint="/local/model_5000.pt", sha256=h(ck))}))
        tj = pt.transfer_jobs()
        assert len(tj) == 1 and tj[0]["name"] == "c2_curve/r/it05000_pm2" and tj[0]["scheme"] == "cin" and tj[0]["mass"] == 10.0
        assert tj[0]["priority"] == 2 and tj[0]["checkpoint"] == "/local/model_5000.pt" and tj[0]["states_sha256_required"] == h(st)
        assert tj[0]["scheme"] in pt.C_SCHEMES
        assert pt.handover_names() == {"d_curve/D/it02000_pm2": "abc"}
    print("ok")


if __name__ == "__main__":
    raise SystemExit(main())
