#!/usr/bin/env python3
"""T14 step 8 driver (tmux, sonic_s1 python): for the 6 step-7 box sequences (after their source solves) and Push / Pull:
candidates + gates -> role solves at 1.0 for every candidate that passes the gates (both roles) -> evaluation (shared_eval.py)
-> verdict per sequence (applicable candidates; the existing selection rule among them: smallest joint weighted fit error,
ties by ID) -> levels a (0.05 steps down from 1.0, not below 0.55), 1.25, 1.5 for the selected candidate (new solves per level)
-> levels.json (step-7 format) and pictures. At most --parallel processes, each started only when MemAvailable - need >= 8 GB.
Return codes in logs_system1/t14/step8/driver.json."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"
L = ROOT / "logs_system1/t14/step8"
OUT = ROOT / "results_system1/T14_retargeting/step8"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
WBT = ROOT / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking"
BOX = [("smallbox", "sub10_smallbox_007", "smallboxb", "b"), ("largebox", "sub3_largebox_024", "largeboxb", "b"), ("suitcase", "sub1_suitcase_065", "suitcaseb", "b"),
       ("smallbox", "sub7_smallbox_027", "smallboxc", "c"), ("largebox", "sub7_largebox_027", "largeboxc", "c"), ("suitcase", "sub14_suitcase_016", "suitcasec", "c")]
TABLE = {"push": WBT / "objects_widetable_plan5_training.urdf", "pull": WBT / "objects_widetable_plan5_pull_training.urdf"}
log = dict(started=time.strftime("%F %T"), steps=[])
running = []
PAR = 6


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


LOGNAME = "driver.json"


def save():
    (L / LOGNAME).write_text(json.dumps(log, indent=1) + "\n")


def reap():
    for j in list(running):
        if j["p"].poll() is not None:
            j["rec"].update(rc=j["p"].returncode, done=time.strftime("%T"))
            running.remove(j)
            save()


def submit(cmd, name, need, logf):
    while True:
        reap()
        if len(running) < PAR and avail() - need >= 8.0:
            break
        time.sleep(5)
    rec = dict(name=name, cmd=" ".join(map(str, cmd)), started=time.strftime("%T"), avail_gb=round(avail(), 1))
    log["steps"].append(rec)
    f = open(logf, "w")
    running.append(dict(p=subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV), rec=rec))
    save()
    time.sleep(3)  # let the process allocate before the next memory reading


def wait_all():
    while running:
        reap()
        time.sleep(5)


def run(cmd, name, need, logf):
    submit(cmd, name, need, logf)
    wait_all()
    return log["steps"][-1].get("rc")


def seqs():
    out = [dict(family="box", obj=o, seq=s, tag=t, cls=c, source=L / f"src/{t}") for o, s, t, c in BOX]
    out += [dict(family="table", task=t, tag=t, cls="b", source=ROOT / f"logs_system1/t14/r6/single_{t}_C") for t in ("push", "pull")]
    return out


def solve_cmd(sq, cdir, role, m, odir):
    cmd = [HS, "-B", "system1/t14/objects/role_solve.py", "--family", sq["family"], "--mode", "role", "--tag", sq["tag"], "--cand", cdir / "cand.json",
           "--role", role, "--m", m, "--out", odir]
    if sq["family"] == "box":
        cmd += ["--object", sq["obj"], "--sequence", sq["seq"], "--source", sq["source"]]
    else:
        g = json.loads((cdir / f"gates_m{m:.2f}".replace(".", "p")).with_suffix(".json").read_text()) if m != 1.0 else {}
        cmd += ["--task", sq["task"], "--target-urdf", g.get("target_urdf", TABLE[sq["task"]])]
    return cmd


def level_dir(cdir, m):
    return cdir / f"m{m:.2f}".replace(".", "p")


def solve_level(sq, cdir, m, wait=True):
    d = level_dir(cdir, m)
    for r in (0, 1):
        if not (d / f"r{r}" / "run_record.json").exists():
            if (d / f"r{r}").exists():
                (d / f"r{r}").rename(ROOT / f"_to_delete/t14_step8/{sq['tag']}_{cdir.name}_{d.name}_r{r}_{time.strftime('%H%M%S')}")
            submit(solve_cmd(sq, cdir, r, m, d / f"r{r}"), f"solve_{sq['tag']}_{cdir.name}_{d.name}_r{r}", 2.0, d.parent / f"{d.name}_r{r}.log")
    if wait:
        wait_all()


def eval_level(sq, cdir, m, wait=True):
    d = level_dir(cdir, m)
    tag = f"m{m:.2f}".replace(".", "p")
    if not all((d / f"r{r}" / "run_record.json").exists() for r in (0, 1)):
        return
    submit([SO, "-B", "system1/t14/objects/shared_eval.py", "--cand-dir", d, "--source", sq["source"], "--gates", cdir / f"gates_{tag}.json",
            "--out", d / "eval.json", "--pair-out", ROOT / f"exports_system1/T14_references/step8/{sq['tag']}_{cdir.name}_{tag}/pair_fps30.npz"],
           f"eval_{sq['tag']}_{cdir.name}_{tag}", 2.5, d.parent / f"eval_{tag}.log")
    if wait:
        wait_all()


def main():
    global PAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--log-name", default="driver.json", help="re-runs (e.g. a crashed solve) write their own log; missing solves are redone, "
                    "existing ones kept; evaluations and the verdict of the given sequences are redone")
    a = ap.parse_args()
    global LOGNAME
    PAR, LOGNAME = a.parallel, a.log_name
    (ROOT / "_to_delete/t14_step8").mkdir(parents=True, exist_ok=True)
    S = [x for x in seqs() if a.only is None or x["tag"] in a.only]
    # sources must exist (src/run.sh)
    while not all((x["source"] / "run_record.json").exists() for x in S if x["family"] == "box"):
        time.sleep(15)
    # 1. candidates + gates
    for sq in S:
        cd = L / f"cands/{sq['tag']}"
        if not (cd / "candidates.json").exists():
            cmd = [SO, "-B", "system1/t14/objects/step8_candidates.py", "--family", sq["family"], "--cls", sq["cls"], "--out", cd]
            cmd += ["--source", sq["source"]] if sq["family"] == "box" else ["--task", sq["task"]]
            run(cmd, f"cands_{sq['tag']}", 1.0, L / f"cands/{sq['tag']}.log")
    # 2. role solves at 1.0 for every gate-passing candidate (all sequences interleaved)
    for sq in S:
        cs = json.loads((L / f"cands/{sq['tag']}/candidates.json").read_text())
        for c in cs["candidates"]:
            if c["passed"]:
                solve_level(sq, L / f"cands/{sq['tag']}/c{c['index']:02d}", 1.0, wait=False)
    wait_all()
    # 3. evaluation at 1.0
    for sq in S:
        cs = json.loads((L / f"cands/{sq['tag']}/candidates.json").read_text())
        for c in cs["candidates"]:
            if c["passed"]:
                eval_level(sq, L / f"cands/{sq['tag']}/c{c['index']:02d}", 1.0, wait=False)
    wait_all()
    # 4. verdict and levels
    summary = {}
    for sq in S:
        cd = L / f"cands/{sq['tag']}"
        cs = json.loads((cd / "candidates.json").read_text())
        ok = []
        for c in cs["candidates"]:
            e = level_dir(cd / f"c{c['index']:02d}", 1.0) / "eval.json"
            if c["passed"] and e.exists() and json.loads(e.read_text())["applicable"]:
                ok.append(c)
        sel = sorted(ok, key=lambda c: (round(c["fit_rmse_m"] / 1e-10), c["id"]))[0] if ok else None
        summary[sq["tag"]] = dict(applicable=bool(ok), applicable_candidates=[c["index"] for c in ok], selected=None if sel is None else sel["index"])
        log["summary"] = summary
        save()
        if sel is None:
            continue
        cdir = cd / f"c{sel['index']:02d}"
        search, a_level = [], None
        e1 = json.loads((level_dir(cdir, 1.0) / "eval.json").read_text())
        search.append(dict(m=1.0, intersect_frames=e1["robot_robot_intersect_frames"], rr_cm=e1["robot_robot_min_distance_cm"]))
        if e1["robot_robot_intersect_frames"] == 0:
            m = 0.95
            while m > 0.549:
                mm = round(m, 2)
                run([SO, "-B", "system1/t14/objects/step8_gates.py", "--cdir", cdir, "--cands", cd / "candidates.json", "--m", mm], f"gates_{sq['tag']}_{mm}", 1.0, cdir / f"gates_{mm}.log")
                solve_level(sq, cdir, mm)
                eval_level(sq, cdir, mm)
                ef = level_dir(cdir, mm) / "eval.json"
                if not ef.exists():
                    search.append(dict(m=mm, failed=True))
                    break
                e = json.loads(ef.read_text())
                search.append(dict(m=mm, intersect_frames=e["robot_robot_intersect_frames"], rr_cm=e["robot_robot_min_distance_cm"]))
                if e["robot_robot_intersect_frames"]:
                    break
                m -= 0.05
            last = search[-1]
            a_level = search[-2]["m"] if (last.get("intersect_frames") or last.get("failed")) else last["m"]
        for mm in (1.25, 1.5):
            run([SO, "-B", "system1/t14/objects/step8_gates.py", "--cdir", cdir, "--cands", cd / "candidates.json", "--m", mm], f"gates_{sq['tag']}_{mm}", 1.0, cdir / f"gates_{mm}.log")
            solve_level(sq, cdir, mm, wait=False)
        wait_all()
        for mm in (1.25, 1.5):
            eval_level(sq, cdir, mm, wait=False)
        wait_all()
        summary[sq["tag"]].update(a=a_level, search=search)
        save()
    log["done"] = time.strftime("%F %T")
    save()
    print("STEP8_DONE", flush=True)


if __name__ == "__main__":
    main()
