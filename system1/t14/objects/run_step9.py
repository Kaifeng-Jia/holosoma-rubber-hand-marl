#!/usr/bin/env python3
"""T14 step 9 driver (Kevin 2026-11-25 00:30; tmux, CPU only, at most --parallel worker processes, each started only when
MemAvailable - need >= 8 GB; no new start after --deadline, America/New_York).

Sequences: class-a clips with |heading - walking direction| < 45 deg at every frame (class_a_turns.csv "straight"); first the
three of Kevin's rule (per object the smallest |box yaw change|): smallbox sub4_smallbox_047, largebox sub7_largebox_045,
suitcase sub8_suitcase_006 ("main"). Per clip: source solve (role_solve --mode source --frames), pre-solve gates on the level grid
(step9_gates.py), role solves (role_solve --construction frontback, both roles) and the verdict (step9_eval.py) per level.
Every clip is solved whatever its gates say (the verdict lists the failed gate).

Main clips (Kevin item 2: levels a, 1.0, 1.25, 1.5; crowded -> lengthen along the walking axis only):
  * 1.0, 1.25, 1.5;
  * a (step-8 rule): only when 1.0 has no robot-robot intersection frame: 0.95, 0.90, ... (not below 0.55) until a level
    intersects; a = the last level without intersection;
  * when 1.5 still intersects: 1.55, 1.60, ... (at most 3.0) until the first level without intersection (fully evaluated);
  * levels.json (step-7 format) and pictures + video (render_box.py --cls a).
Extension (Kevin item 3, only if at least one main clip is applicable at one of its reported levels): every other straight
clip at 1.0 and at its smallest level without intersection on the same grid (1.0 clear -> the a search; else 1.25, 1.5, 1.55,
... <= 3.0). Record: logs_system1/t14/step9b/driver.json.

    sonic_s1/bin/python -B system1/t14/objects/run_step9.py --parallel 6 --deadline 08:15
"""
import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
SO = "/home/kevin/.holosoma_deps/miniconda3/envs/sonic_s1/bin/python"
HS = "/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting_s1/bin/python"
L = ROOT / "logs_system1/t14/step9b"
EXP = ROOT / "exports_system1/T14_references/step9"
VID = ROOT / "exports_system1/T14_videos/step9"
FIG = ROOT / "results_system1/T14_retargeting/step9b/figures"
TURNS = ROOT / "results_system1/T14_retargeting/step9/class_a_turns.csv"
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
           __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/50_mesa.json")
NY = ZoneInfo("America/New_York")
MAIN = {"sub4_smallbox_047": "smallboxa", "sub7_largebox_045": "largeboxa", "sub8_suitcase_006": "suitcasea"}
DOWN = [round(1.0 - 0.05 * i, 2) for i in range(1, 10)]  # 0.95 .. 0.55
UP = [round(1.5 + 0.05 * i, 2) for i in range(1, 31)]  # 1.55 .. 3.0
GRID = sorted(set(DOWN + [1.0, 1.25, 1.5] + UP))
log = dict(started=datetime.now(NY).isoformat(timespec="seconds"), steps=[], summary={})
lock = threading.Lock()
slots = None
DEADLINE = None


def now():
    return datetime.now(NY)


def avail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 ** 2


def save():
    with lock:
        (L / "driver.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n")


def tagm(m):
    return f"m{m:.2f}".replace(".", "p")


def run(cmd, name, need, logf):
    """one worker process; None = not started (deadline)."""
    slots.acquire()
    try:
        while True:
            if now() >= DEADLINE:
                with lock:
                    log["steps"].append(dict(name=name, skipped="deadline", time=now().strftime("%T")))
                save()
                return None
            with lock:
                ok = avail() - need >= 8.0
                if ok:
                    rec = dict(name=name, cmd=" ".join(map(str, cmd)), started=now().strftime("%T"), avail_gb=round(avail(), 1))
                    log["steps"].append(rec)
                    logf.parent.mkdir(parents=True, exist_ok=True)
                    f = open(logf, "w")
                    p = subprocess.Popen(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=ENV)
                    time.sleep(3)  # let it allocate before the next memory reading
            if ok:
                break
            time.sleep(10)
        save()
        rc = p.wait()
        f.close()
        rec.update(rc=rc, done=now().strftime("%T"))
        save()
        return rc
    finally:
        slots.release()


def parallel(jobs):
    """run (cmd, name, need, logf) jobs concurrently (each takes a slot); returns their return codes."""
    out = [None] * len(jobs)

    def go(i, j):
        out[i] = run(*j)
    ts = [threading.Thread(target=go, args=(i, j)) for i, j in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


class Clip:
    def __init__(self, row):
        self.obj, self.seq = row["object"], row["sequence"]
        self.frames = f"{row['start']}:{row['stop']}"
        self.tag = MAIN.get(self.seq, self.seq)
        self.d = L / self.tag
        self.src = L / "src" / self.tag
        self.main = self.seq in MAIN
        self.evals = {}
        self.sm = log["summary"].setdefault(self.tag, dict(object=self.obj, sequence=self.seq, frames=self.frames, main=self.main))

    def source(self):
        if (self.src / "run_record.json").exists():
            return True
        if self.src.exists():
            dst = ROOT / f"_to_delete/t14_step9b/src_{self.tag}_{time.strftime('%H%M%S')}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            self.src.rename(dst)
        rc = run([HS, "-B", "system1/t14/objects/role_solve.py", "--family", "box", "--object", self.obj, "--sequence", self.seq, "--mode", "source",
                  "--frames", self.frames, "--tag", self.tag, "--out", self.src], f"source_{self.tag}", 2.0, L / "src" / f"{self.tag}.log")
        return rc == 0 and (self.src / "run_record.json").exists()

    def gates(self):
        if all((self.d / f"gates_{tagm(m)}.json").exists() for m in GRID):
            return True
        rc = run([HS, "-B", "system1/t14/objects/step9_gates.py", "--object", self.obj, "--sequence", self.seq, "--frames", self.frames,
                  "--levels", ",".join(f"{m:.2f}" for m in GRID), "--out", self.d], f"gates_{self.tag}", 1.5, self.d / "gates.log")
        return rc == 0

    def solve_jobs(self, m):
        d = self.d / tagm(m)
        jobs = []
        for r in (0, 1):
            if (d / f"r{r}" / "run_record.json").exists():
                continue
            if (d / f"r{r}").exists():
                dst = ROOT / f"_to_delete/t14_step9b/{self.tag}_{tagm(m)}_r{r}_{time.strftime('%H%M%S')}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                (d / f"r{r}").rename(dst)
            cmd = [HS, "-B", "system1/t14/objects/role_solve.py", "--family", "box", "--object", self.obj, "--sequence", self.seq, "--mode", "role",
                   "--construction", "frontback", "--role", r, "--m", m, "--frames", self.frames, "--tag", self.tag, "--out", d / f"r{r}"]
            if r == 0:
                cmd += ["--source", self.src]
            jobs.append((cmd, f"solve_{self.tag}_{tagm(m)}_r{r}", 2.0, d / f"r{r}.log"))
        return jobs

    def eval_job(self, m):
        d = self.d / tagm(m)
        return ([SO, "-B", "system1/t14/objects/step9_eval.py", "--cand-dir", d, "--source", self.src, "--gates", self.d / f"gates_{tagm(m)}.json",
                 "--out", d / "eval.json", "--pair-out", EXP / f"{self.tag}_{tagm(m)}" / "pair_fps30.npz"], f"eval_{self.tag}_{tagm(m)}", 2.5, d / "eval.log")

    def levels(self, ms):
        """solve + evaluate the levels (concurrently); returns {m: eval or None}."""
        ms = [m for m in ms if m not in self.evals]
        parallel([j for m in ms for j in self.solve_jobs(m)])
        ok = [m for m in ms if all((self.d / tagm(m) / f"r{r}" / "run_record.json").exists() for r in (0, 1))]
        parallel([self.eval_job(m) for m in ok if not (self.d / tagm(m) / "eval.json").exists()])
        for m in ms:
            f = self.d / tagm(m) / "eval.json"
            self.evals[m] = json.loads(f.read_text()) if f.exists() else None
            e = self.evals[m]
            self.sm.setdefault("levels", {})[f"{m:.2f}"] = None if e is None else dict(applicable=e["applicable"], failed=e["failed"],
                                                                                    intersect_frames=e["robot_robot_intersect_frames"], rr_cm=e["robot_robot_min_distance_cm"])
        save()
        return {m: self.evals[m] for m in ms}

    def inter(self, m):
        e = self.evals.get(m)
        return None if e is None else e["robot_robot_intersect_frames"] > 0

    def search_down(self):
        a = 1.0
        for m in DOWN:
            self.levels([m])
            x = self.inter(m)
            if x is None or x:
                break
            a = m
        return a

    def search_up(self, grid):
        for m in grid:
            self.levels([m])
            x = self.inter(m)
            if x is None:
                return None
            if not x:
                return m
        return None

    def prepare(self):
        if not self.source():
            self.sm["error"] = "source solve failed or not started"
            save()
            return False
        if not self.gates():
            self.sm["error"] = "gates failed or not started"
            save()
            return False
        return True

    def run_main(self):
        try:
            self._run_main()
        except Exception:
            import traceback
            self.sm["exception"] = traceback.format_exc()[-2000:]
            save()
        finally:
            decide.update(self)

    def _run_main(self):
        if not self.prepare():
            return
        self.levels([1.0, 1.25, 1.5])
        decide.early(self)
        a = self.search_down() if self.inter(1.0) is False else None
        free = self.search_up(UP) if self.inter(1.5) else None
        rep = sorted({1.0, 1.25, 1.5} | ({a} if a is not None else set()) | ({free} if free is not None else set()))
        self.sm.update(a=a, first_free_above_1p5=free, report_levels=rep,
                       applicable_levels=[m for m in rep if (self.evals.get(m) or {}).get("applicable")])
        self.sm["min_free"] = min((m for m in self.evals if self.inter(m) is False), default=None)
        save()
        decide.update(self)
        self.render(rep, a, free)
        self.done = True

    def run_ext(self):
        try:
            self._run_ext()
        except Exception:
            import traceback
            self.sm["exception"] = traceback.format_exc()[-2000:]
            save()

    def _run_ext(self):
        if not self.prepare():
            return
        self.levels([1.0])
        if self.inter(1.0) is False:
            mf = self.search_down()
        elif self.inter(1.0):
            mf = self.search_up([1.25, 1.5] + UP)
        else:
            mf = None
        self.sm.update(min_free=mf, applicable_levels=[m for m in sorted({1.0} | ({mf} if mf else set())) if (self.evals.get(m) or {}).get("applicable")])
        save()

    def render(self, rep, a, free):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from step8_levels_json import conv
        out = {}
        for m in rep:
            e = self.evals.get(m)
            if e is not None:
                out[f"{m:.2f}"] = conv(e, m, self.d / tagm(m), json.loads((self.d / tagm(m) / "r0/run_record.json").read_text()))
        if "1.00" not in out:
            return
        e1 = self.evals[1.0]
        status = "一前一后 1.0 倍：" + ("适用" if e1["applicable"] else "不适用（" + "、".join(e1["failed"]) + "）")
        rec0 = json.loads((self.d / "m1p00/r0/run_record.json").read_text())
        srec = json.loads((self.src / "run_record.json").read_text())
        lv = dict(tag=self.tag, role_status=status, applicable=bool(e1["applicable"]), selected=None, a=a, first_free_level_above_1=free,
                  report_levels=[float(k) for k in out if free is None or float(k) != free], levels=out,
                  source_box_m=srec["target"]["size"], solve_r0=str(self.d / "m1p00/r0"), clip_start=rec0["clip_start"])
        (self.d / "levels.json").write_text(json.dumps(lv, indent=1, ensure_ascii=False) + "\n")
        rc = run([SO, "-B", "system1/t14/objects/render_box.py", "--solve", self.d / "m1p00/r0", "--levels", self.d / "levels.json", "--cls", "a",
                  "--clip-start", rec0["clip_start"], "--out", FIG, "--video-out", VID], f"render_{self.tag}", 4.3, self.d / "render.log")
        self.sm["render_rc"] = rc
        save()


class Decide:
    def __init__(self):
        self.ev = threading.Event()
        self.n = 0
        self.yes = False
        self.counted = set()

    def early(self, clip):
        """after 1.0 / 1.25 / 1.5: a main clip already applicable there decides the extension (later levels only add)."""
        if any((clip.evals.get(m) or {}).get("applicable") for m in (1.0, 1.25, 1.5)):
            with lock:
                if not self.yes:
                    self.yes = True
                    log["extension_decision"] = dict(run=True, time=now().isoformat(timespec="seconds"), by=clip.tag, after="levels 1.0 / 1.25 / 1.5",
                                                     rule="at least one main clip applicable at one of its reported levels")
                self.ev.set()

    def update(self, clip):
        with lock:
            if clip.tag in self.counted:
                return
            self.counted.add(clip.tag)
            self.n += 1
            if clip.sm.get("applicable_levels"):
                self.yes = True
            if (self.yes or self.n == len(MAIN)) and not self.ev.is_set():
                log["extension_decision"] = dict(run=self.yes, time=now().isoformat(timespec="seconds"),
                                                 rule="at least one main clip applicable at one of its reported levels")
                self.ev.set()


decide = Decide()


def main():
    global slots, DEADLINE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--deadline", default="08:15", help="HH:MM America/New_York today: no new worker after it")
    a = ap.parse_args()
    slots = threading.BoundedSemaphore(a.parallel)
    hh, mm = (int(x) for x in a.deadline.split(":"))
    DEADLINE = now().replace(hour=hh, minute=mm, second=0, microsecond=0)
    log.update(parallel=a.parallel, deadline=DEADLINE.isoformat(timespec="seconds"), grid=GRID)
    rows = [r for r in csv.DictReader(open(TURNS)) if r["straight"] == "True"]
    main_rows = [r for r in rows if r["sequence"] in MAIN]
    ext_rows = [r for r in rows if r["sequence"] not in MAIN]
    assert len(main_rows) == 3 and len(rows) == 22, (len(main_rows), len(rows))
    log["clips"] = [dict(object=r["object"], sequence=r["sequence"], frames=f"{r['start']}:{r['stop']}", main=r["sequence"] in MAIN) for r in rows]
    save()
    ts = [threading.Thread(target=Clip(r).run_main) for r in main_rows]
    for t in ts:
        t.start()
    decide.ev.wait()
    ets = []
    if decide.yes:
        ets = [threading.Thread(target=Clip(r).run_ext) for r in ext_rows]
        for t in ets:
            t.start()
    for t in ts + ets:
        t.join()
    log["done"] = now().isoformat(timespec="seconds")
    save()
    print("STEP9B_DONE", flush=True)


if __name__ == "__main__":
    main()
