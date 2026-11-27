#!/usr/bin/env python3
"""T14 step 9e: return package (small files only) from logs_system1/t14/step9e (copy of step9d_package.py with the 9e paths;
the 9d nonhand files are replaced by the 9e diagnostics: diag9e/<clip>_<level>.json per evaluated level, jump.json, and the 9d
support-drift side check logs_system1/t14/step9d_r1check). Read only on the logs.

Copies into results_system1/T14_retargeting/step9d/: evals/<clip>_<level>.json (every evaluated level), levels/<clip>/levels.json
(the 3 main clips, render input), diag/<clip>.json (pre-solve gates diagnostics, all 22 clips), records/ (driver.json, driver.log,
the tail of every failed solve log, every run_record.json, gates logs), nonhand.json / nonhand_phase.csv (diagnostic), and writes
inputs_outputs_sha256.csv (sequences, OMOMO release, clip list, scene template, robot URDF, source solves, target URDFs, pairs,
videos).

    python3 system1/t14/objects/step9e_package.py
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / "logs_system1/t14/step9e"
OUT = ROOT / "results_system1/T14_retargeting/step9e"
VID = ROOT / "exports_system1/T14_videos/step9e"
R1C = ROOT / "logs_system1/t14/step9d_r1check"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import argparse
    global L, OUT, VID
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="step9e", help="step 9f: step9f (same layout; 9e-only files are copied when present)")
    ap.add_argument("--turns", type=Path, default=ROOT / "results_system1/T14_retargeting/step9/class_a_turns.csv",
                    help="clip list recorded in the inventory (final test after 9g2: results_system1/T14_retargeting/step9_final/class_a_turns.csv)")
    a = ap.parse_args()
    L, OUT, VID = ROOT / f"logs_system1/t14/{a.step}", ROOT / f"results_system1/T14_retargeting/{a.step}", ROOT / f"exports_system1/T14_videos/{a.step}"
    d = json.loads((L / "driver.json").read_text())
    DG = "diag" + a.step[len("step"):]  # per-level phase diagnosis (diag9e, diag9f, diag9g) (step9e_diag.py; the file in each level dir is diag9e.json)
    for sub in ("evals", "levels", "diag", DG, "side_check_9d_drift", "records/run_records", "records/failed_logs", "records/gates_logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    inv = []

    def add(kind, p):
        p = Path(p)
        inv.append(dict(kind=kind, path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), bytes=p.stat().st_size, sha256=sha(p)))

    shutil.copy2(L / "driver.json", OUT / "records/driver.json")
    if (L / "driver.log").exists() or not (L / "run_all22.log").exists():
        shutil.copy2(L / "driver.log", OUT / "records/driver.log")
    for n in ("run.sh", "run_all22.sh", "run_all22.log", "post22.sh", "driver_debug3.json", "driver_debug3.log", "post.sh", "post.log"):  # step 9g2: driver log = run_all22.log
        if (L / n).exists():
            shutil.copy2(L / n, OUT / "records" / n)
    for n in ("run.sh", "jobs.txt", "diag_rc.txt"):
        if (L / DG / n).exists():
            shutil.copy2(L / DG / n, OUT / "records" / f"{DG}_{n}")
    if (L / "diag9e/jump.json").exists():
        shutil.copy2(L / "diag9e/jump.json", OUT / "jump.json")
    if a.step == "step9e":
        for n in ("drift_recheck.json", "drift_recheck.csv", "run.sh", "run.log"):
            shutil.copy2(R1C / n, OUT / "side_check_9d_drift" / n)
    else:
        (OUT / "side_check_9d_drift").rmdir()
    for g in sorted((L / "records").glob("gates_*.log")):
        shutil.copy2(g, OUT / "records/gates_logs" / g.name)
    seqs = set()
    for tag, s in d["summary"].items():
        cd = L / tag
        if (cd / "diag.json").exists():
            shutil.copy2(cd / "diag.json", OUT / "diag" / f"{tag}.json")
        if (cd / "gates.log").exists():
            shutil.copy2(cd / "gates.log", OUT / "records/gates_logs" / f"{tag}_gates.log")
        if s["main"] and (cd / "levels.json").exists():
            (OUT / "levels" / tag).mkdir(exist_ok=True)
            shutil.copy2(cd / "levels.json", OUT / "levels" / tag / "levels.json")
        seqs.add(s["sequence"])
        src = next((Path(r) / tag for r in d["source_root"] if (Path(r) / tag / "run_record.json").exists()), L / "src" / tag)
        srec = json.loads((src / "run_record.json").read_text())
        shutil.copy2(src / "run_record.json", OUT / "records/run_records" / f"{tag}_src.json")
        q = sorted(src.glob("*_new_qpos.npz"))
        if q:
            add(f"source solve {tag} (reused from {src.parent.parent.name})", q[0])
        for lv in sorted(cd.glob("m*p*")):
            if not lv.is_dir():
                continue
            m = lv.name
            for r in (0, 1):
                rr = lv / f"r{r}" / "run_record.json"
                if rr.exists():
                    shutil.copy2(rr, OUT / "records/run_records" / f"{tag}_{m}_r{r}.json")
                lg = lv / f"r{r}.log"
                if not rr.exists() and lg.exists():
                    txt = lg.read_text(errors="replace").replace("\r", "\n").splitlines()
                    (OUT / "records/failed_logs" / f"{tag}_{m}_r{r}.log").write_text("\n".join(txt[-40:]) + "\n")
            if (lv / "diag9e.json").exists():
                shutil.copy2(lv / "diag9e.json", OUT / DG / f"{tag}_{m}.json")
            if (lv / "eval.json").exists():
                shutil.copy2(lv / "eval.json", OUT / "evals" / f"{tag}_{m}.json")
                e = json.loads((lv / "eval.json").read_text())
                add(f"target URDF {tag}_{m}", e["target_urdf"])
                if e.get("pair") and Path(e["pair"]).exists():
                    add(f"pair {tag}_{m}", e["pair"])
        del srec
    head = []
    for s in sorted(seqs):
        head.append(("OMOMO_new sequence", Path("/home/kevin/holosoma/OMOMO_new") / f"{s}.pt"))
    head += [("OMOMO original release (swing segments, person render)", Path("/home/kevin/holosoma/data/train_diffusion_manip_seq_joints24.p")),
             ("clip list", a.turns if a.turns.is_absolute() else ROOT / a.turns),
             ("scene template", ROOT / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"),
             ("robot URDF (render / evaluation)", ROOT / "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf")]
    body = inv
    inv = []
    for k, p in head:
        if p.exists():
            add(k, p)
    inv += body
    for v in sorted(VID.glob("*.mp4")):
        add("video", v)
    with open(OUT / "inputs_outputs_sha256.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(inv)
    print(len(inv), "inventory rows")


if __name__ == "__main__":
    main()
