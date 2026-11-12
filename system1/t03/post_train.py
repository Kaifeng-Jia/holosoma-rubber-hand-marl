#!/usr/bin/env python3
"""Post-training evaluation pipeline (Kevin 2026-11-10: evaluate every cloud run as soon as it finishes).

Every finished cloud training run (ops/<final segment>.exit.json completed=true) is fetched to
exports_system1/train/<label>/ (checkpoints, run_config.json, metrics.jsonl, status.json, ops records; sha256 checked
against the cloud) and gets, on the local RTX 5070 Laptop (same GPU model for C and D, and for A/B/D +/-2):
  * final : final checkpoint, frozen +/-2 and +/-8 states (T03 / T04 protocol: GPU PhysX, TF32 off)       priority 0
  * curve : T03 training speed, every saved checkpoint, frozen +/-2 states (C main runs, all seeds)         priority 1
D same-protocol curves (T03 "D 的同口径曲线", existing checkpoints, no retraining) fill idle slots:
  seed 721 priority 2, seeds 722/723 priority 3 (C extra seeds deferred, T03 2026-11-10)
  Pull: V3-25-27 (D from scratch, 1000-iteration spacing) and the paper main result (D prior init);
  Table (T-07), Bin (B-RZ0): paper main result, 2000-iteration spacing.

At most --slots evaluator processes on the local GPU at a time (8 GB: two Isaac processes), counting evaluator
processes started by anyone. Launch / exit records per job (argv, git HEAD/status/diff, checkpoint, states and
GPU); success = complete summary.json (the evaluator exits 0 even after an exception). Never deletes; skips jobs
whose exit record says completed.

    python system1/t03/post_train.py catalog          # D curve checkpoint list + sha256 -> logs_system1/post_train/d_curve_catalog.json
    python system1/t03/post_train.py fetch-v3         # copy the V3-25-27 intermediate checkpoints from the cloud (read-only source)
    python system1/t03/post_train.py run [--dry-run]  # poll / fetch / evaluate loop

--cloud-transfer (Kevin 2026-11-12: the cloud evaluates for at most 2 h after its last training; unfinished evaluations
go to the laptop): at each poll, once the cloud scheduler has written logs/evals/ops2/transfer_to_laptop.json, it is copied
to logs_system1/post_train/cloud_transfer.json; C v2 curve checkpoints listed there are copied to
exports_system1/cloud_transfer/<run>/ (sha256 checked) and evaluated here with the same evaluator and frozen states
(states sha256 must equal the cloud's), priority 2, after every final evaluation; output names as on the cloud
(c2_curve/...). D curve jobs listed there were handed over by the cloud (claims/<name>/handover.json) and are run here
without a new claim. Off (default) = old behaviour.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.scheme_a_queue import PUSH_PROOF, STATES  # noqa: E402  frozen states + sha256 per (case, level)

NY = ZoneInfo("America/New_York")
PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
SSH = ["ssh", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-p", "32748", "root@connect.nmb1.seetacloud.com"]
SCP = ["scp", "-q", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", "-P", "32748"]
REMOTE_TRAIN = "/root/autodl-tmp/system1/logs/train"
REMOTE_EVALS = "/root/autodl-tmp/system1/logs/evals"
REMOTE_V3 = "/root/autodl-tmp/v3_20261103_pull/logs/V3_20261103_pull_retry01"
OUT = ROOT / "logs_system1/post_train"
EXPORT = ROOT / "exports_system1"
ORIGINAL_MASS = {"pull": 20.0, "table": 5.0, "bucket": 1.0}
LABEL = re.compile(r"^(c|d|cin|cout)_(pull|table|bucket)_m([0-9p.]+)kg_seed(\d+)_(joint8000|fresh12000)$")
# cloud runs, final segment labels: wave 1 (T03 C main 721, T04 D), wave 2 (T04 C, T03 C extra seeds)
RUNS = ["d_pull_m40kg_seed721_joint8000", "d_table_m10kg_seed721_fresh12000", "d_table_m20kg_seed721_fresh12000",
        "d_bucket_m2kg_seed721_fresh12000", "d_bucket_m4kg_seed721_fresh12000", "d_bucket_m8kg_seed721_fresh12000",
        "c_pull_m20kg_seed721_joint8000", "c_table_m5kg_seed721_fresh12000", "c_bucket_m1kg_seed721_fresh12000",
        "c_pull_m40kg_seed721_joint8000", "c_table_m10kg_seed721_fresh12000", "c_table_m20kg_seed721_fresh12000",
        "c_bucket_m2kg_seed721_fresh12000", "c_bucket_m4kg_seed721_fresh12000", "c_bucket_m8kg_seed721_fresh12000"]
# T03 C v2 (Kevin 2026-11-12): final-checkpoint +/-2 and +/-8 of the C v2 main runs and the C v2 mass sweep on this laptop
# (RTX 5070, the GPU of schemes A and B); the cloud skips these job names (claims owned by "laptop", made 2026-11-12 00:0x).
# Intermediate C v2 checkpoints are evaluated on the cloud (c2_curve), so no curve jobs here.
RUNS += [f"{k}_{t}_seed721_{s}" for k in ("cin", "cout") for t, s in (("pull_m20kg", "joint8000"), ("table_m5kg", "fresh12000"),
                                                                     ("bucket_m1kg", "fresh12000"))]
RUNS += ["cin_table_m10kg_seed721_fresh12000", "cin_table_m20kg_seed721_fresh12000", "cin_bucket_m2kg_seed721_fresh12000",
         "cin_bucket_m4kg_seed721_fresh12000", "cin_bucket_m8kg_seed721_fresh12000", "cin_pull_m40kg_seed721_joint8000"]
# C-in extra seeds 722 / 723 (Kevin 2026-11-12): finals on the laptop as well (claims owned by "laptop")
RUNS += [f"cin_{t}_seed{sd}_{s}" for sd in (722, 723) for t, s in (("pull_m20kg", "joint8000"), ("table_m5kg", "fresh12000"),
                                                                   ("bucket_m1kg", "fresh12000"))]
C_SCHEMES = ("c", "cin", "cout")  # all evaluated with --controller sonic_residual (kind read from the checkpoint)
# C extra seeds 722/723: deferred until after the RSS stage-1 submission (T03 / README, commit 0712082)
_B = "/home/kevin/holosoma-core4d-base/logs"
_R = f"{_B}/PaperSupplement20260929/autodl_received/runs"
# D same-protocol curves: label -> (case, [(dir, iterations or None = all model_*.pt)], expected final sha256 prefix)
D_CURVES = {
    **{f"D_pull_scratch_V3-{25 + i}_seed{s}": ("pull", [(str(EXPORT / f"d_curve/V3-{25 + i}_pull_scratch_seed{s}_joint8000"), None)], None)
       for i, s in enumerate((721, 722, 723))},
    "D_pull_paper_seed721": ("pull", [(f"/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_critic50_seed721_env2048", [50]),
                                      (f"/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048", None)], "727630e9"),
    **{f"D_pull_paper_seed{s}": ("pull", [(f"{_R}/Pull_seed{s}_stage0_from0_plus50_20261001", [50]),
                                         (f"{_R}/Pull_seed{s}_stage1_from50_plus8000_20261001", None)], p)
       for s, p in ((722, "d6e545e5"), (723, "e4e5eaed"))},
    "D_table_paper_T-07_seed721": ("table", [(f"{_B}/Core4DSmallTableA/A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920", None)], "f163d11a"),
    # 722/723: 0-10000 from the fresh run, 12000 from the resume run (their model_10000 files hold identical actor and
    # normalizer weights; checked 2026-11-10)
    **{f"D_table_paper_T-07_seed{s}": ("table", [(f"{_R}/T07_seed{s}_fresh12000_cache_isolated02", None),
                                                (f"{_R}/T07_seed{s}_resume10000_to12000_clone20260930", [12000])], p)
       for s, p in ((722, "007161dc"), (723, "df9bc014"))},
    "D_bucket_paper_B-RZ0_seed721": ("bucket", [(f"{_B}/Core4DBucket/A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919", None)], "7ba039fb"),
    **{f"D_bucket_paper_B-RZ0_seed{s}": ("bucket", [(f"{_R}/BRZ0_seed{s}_fresh12000_concurrent_20260930", None)], p)
       for s, p in ((722, "05850d5b"), (723, "f936aa17"))},
}
EVAL_PATTERN = "scripts/evaluate_paper_supplement.py"
# Scheme B formal evaluations (T02 v2 + preregistration section 8), enabled by B_ENABLED.json once the revised planner
# gate has passed and the chosen LAST_WAYPOINT_FRAC is committed; PAUSE stops new launches (running jobs continue).
B_ENABLE = OUT / "B_ENABLED.json"
PAUSE = OUT / "PAUSE"
B_TASKS = ("push", "pull", "chair", "bucket", "table")    # Kick not applicable (T02 v2)
B_MAIN = ("B0", "B1-0", "B1-2", "B1-4")
B_MASS_VARIANTS = ("B0", "B1-2")                           # T02: mass sweep with B0 and B1-2 (fixed in advance)
B_MASS = {"pull": (5.0, 10.0, 40.0), "table": (2.5, 10.0, 20.0), "bucket": (2.0, 4.0, 8.0)}  # T04 non-original levels


def now():
    return datetime.now(NY).isoformat(timespec="seconds")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def ssh(cmd, timeout=120):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout


def it_of(p):
    return int(re.search(r"model_(\d+)\.pt$", str(p)).group(1))


def parse(label):
    m = LABEL.match(label)
    scheme, case, mass, seed, seg = m.groups()
    return dict(scheme=scheme, case=case, mass=float(mass.replace("p", ".")), seed=int(seed), segment=seg)


def catalog(skip_missing=False):
    out, skipped = {}, []
    for label, (case, parts, final_prefix) in D_CURVES.items():
        ck = {}
        if skip_missing and any(Path(d).name.startswith("V3-") and not (Path(d) / "SHA256SUMS.remote").exists() for d, _ in parts):
            skipped.append(label)  # V3-25-27 copy from the cloud not finished yet
            continue
        for d, its in parts:
            files = sorted(Path(d).glob("model_*.pt")) if its is None else [Path(d) / f"model_{i:05d}.pt" for i in its]
            for f in files:
                if not f.exists():
                    raise FileNotFoundError(f)
                ck[it_of(f)] = str(f)  # later parts override (resume run supplies the final iterations)
        its = sorted(ck)
        final = ck[its[-1]]
        fs = sha(final)
        if final_prefix and not fs.startswith(final_prefix):
            raise RuntimeError(f"{label}: final checkpoint sha {fs[:8]} != paper {final_prefix}")
        out[label] = dict(case=case, scheme="d", seed=int(label.rsplit("seed", 1)[1]), iterations=its,
                          spacing=sorted({b - a for a, b in zip(its[1:], its[2:])}) if len(its) > 2 else None,
                          checkpoints={str(i): dict(path=ck[i], sha256=sha(ck[i])) for i in its},
                          final_matches_paper_registry=bool(final_prefix), retrained=False)
        print(label, its, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "d_curve_catalog.json.tmp"
    tmp.write_text(json.dumps(dict(created=now(), runs=out, skipped_not_yet_copied=skipped), indent=1) + "\n")
    tmp.replace(OUT / "d_curve_catalog.json")


def fetch_v3():
    for i, s in enumerate((721, 722, 723)):
        name = f"V3-{25 + i}_pull_scratch_seed{s}_joint8000"
        dst = EXPORT / "d_curve" / name
        dst.mkdir(parents=True, exist_ok=True)
        rc, out = ssh(f"cd {REMOTE_V3}/{name} && sha256sum model_*.pt run_config.json")
        remote = dict(reversed(l.split()) for l in out.strip().splitlines())
        for f, h in remote.items():
            if (dst / f).exists() and sha(dst / f) == h:
                continue
            subprocess.run(SCP + [f"root@connect.nmb1.seetacloud.com:{REMOTE_V3}/{name}/{f}", str(dst / f)], check=True, timeout=900)
            if sha(dst / f) != h:
                raise RuntimeError(f"sha mismatch after copy: {dst / f}")
        (dst / "SHA256SUMS.remote").write_text("".join(f"{h}  {f}\n" for f, h in sorted(remote.items())))
        print(name, len(remote), "files", flush=True)


def fetch_run(label):
    """Copy a finished run (all segments of its queue) to exports_system1/train/<label>/; returns local dir."""
    meta = parse(label)
    dst = EXPORT / "train" / label
    dst.mkdir(parents=True, exist_ok=True)
    rc, out = ssh(f"cd {REMOTE_TRAIN}/{label} && sha256sum model_*.pt run_config.json metrics.jsonl status.json 2>/dev/null; "
                  f"cd {REMOTE_TRAIN}/ops && ls {label}.* " + (label.replace("joint8000", "critic50") + ".* " if meta["segment"] == "joint8000" else ""))
    files = {}
    ops = []
    for l in out.strip().splitlines():
        parts = l.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            files[parts[1]] = parts[0]
        elif l.strip():
            ops.append(l.strip())
    for f, h in files.items():
        if (dst / f).exists() and sha(dst / f) == h:
            continue
        subprocess.run(SCP + [f"root@connect.nmb1.seetacloud.com:{REMOTE_TRAIN}/{label}/{f}", str(dst / f)], check=True, timeout=1800)
        if sha(dst / f) != h:
            raise RuntimeError(f"sha mismatch after copy: {dst / f}")
    (dst / "ops").mkdir(exist_ok=True)
    for f in ops:
        subprocess.run(SCP + [f"root@connect.nmb1.seetacloud.com:{REMOTE_TRAIN}/ops/{f}", str(dst / "ops" / f)], check=True, timeout=300)
    (dst / "SHA256SUMS.remote").write_text("".join(f"{h}  {f}\n" for f, h in sorted(files.items())))
    (dst / "fetched.json").write_text(json.dumps(dict(label=label, fetched=now(), files=files, ops=ops), indent=1))
    return dst


def finished_remote():
    rc, out = ssh(f"cd {REMOTE_TRAIN}/ops && for f in " + " ".join(f"{r}.exit.json" for r in RUNS) +
                  "; do [ -f $f ] && echo $f $(python3 -c \"import json;print(json.load(open('$f')).get('completed'))\"); done")
    done, failed = [], []
    for l in out.strip().splitlines():
        f, c = l.split()
        (done if c == "True" else failed).append(f[: -len(".exit.json")])
    return done, failed


def jobs_for_run(label, d):
    meta = parse(label)
    cks = sorted(d.glob("model_*.pt"), key=it_of)
    final = cks[-1]
    target = 8050 if meta["segment"] == "joint8000" else 12000
    if it_of(final) != target:
        raise RuntimeError(f"{label}: final checkpoint {final.name}, expected {target}")
    kind = "T04" if meta["mass"] != ORIGINAL_MASS[meta["case"]] else "T03"
    out = []
    for lvl in (2, 8):
        out.append(dict(priority=0, name=f"final/{label}_pm{lvl}", case=meta["case"], scheme=meta["scheme"], mass=meta["mass"],
                        checkpoint=str(final), level=lvl, task_book=kind, run=label))
    if kind == "T03" and meta["scheme"] == "c":
        for c in cks:
            out.append(dict(priority=1, name=f"curve/{label}/it{it_of(c):05d}_pm2", case=meta["case"], scheme="c", mass=meta["mass"],
                            checkpoint=str(c), level=2, task_book="T03", run=label))
    return out


def d_curve_jobs():
    f = OUT / "d_curve_catalog.json"
    if not f.exists():
        return []
    try:
        runs = json.loads(f.read_text())["runs"]
    except (json.JSONDecodeError, KeyError):  # being rewritten: try again next loop
        return []
    out = []
    for label, r in runs.items():
        for i, c in r["checkpoints"].items():
            # seed 721 first (C runs only 721 this round, T03 2026-11-10); 722/723 fill idle time afterwards
            out.append(dict(priority=3 if r["seed"] == 721 else 4, name=f"d_curve/{label}/it{int(i):05d}_pm2", case=r["case"], scheme="d",
                            mass=ORIGINAL_MASS[r["case"]], checkpoint=c["path"], checkpoint_sha256=c["sha256"], level=2,
                            task_book="T03", run=label))
    return out


def early_curve_jobs():
    """T03 curve points from checkpoints of runs still training (Kevin 2026-11-11: latest C Pull checkpoint, +/-2, before
    the B mass sweep). early_curve.json lists {label, checkpoint, sha256} fetched to exports_system1/train_partial/; the job
    name equals the curve job the finished run would get, so it is not re-run after the full fetch."""
    f = OUT / "early_curve.json"
    if not f.exists():
        return []
    out = []
    for e in json.loads(f.read_text())["jobs"]:
        meta = parse(e["label"])
        out.append(dict(priority=1, name=f"curve/{e['label']}/it{it_of(Path(e['checkpoint'])):05d}_pm2", case=meta["case"], scheme=meta["scheme"],
                        mass=meta["mass"], checkpoint=e["checkpoint"], checkpoint_sha256=e["sha256"], level=2, task_book="T03",
                        run=e["label"], early=True))
    return out


def d_paper_pm8_jobs():
    """Kevin 2026-11-12 item 2: the D paper main result, Pull / Table / Bin x seeds 721-723, final checkpoint, +/-8 on this
    laptop (the earlier +/-8 ran on 4090s); the scheme comparison table uses laptop 5070 numbers only. The +/-2 of the
    722 / 723 finals are the d_curve jobs of the same checkpoints, claimed for the laptop and raised to priority 0
    (priority_overrides.json)."""
    f = OUT / "d_curve_catalog.json"
    if not f.exists():
        return []
    runs = json.loads(f.read_text())["runs"]
    out = []
    for label, r in runs.items():
        if "_paper_" not in label:
            continue
        it = max(int(i) for i in r["checkpoints"])
        c = r["checkpoints"][str(it)]
        prefix = D_CURVES[label][2]
        if prefix and not c["sha256"].startswith(prefix):
            raise RuntimeError(f"{label}: final checkpoint sha {c['sha256'][:8]} != paper {prefix}")
        out.append(dict(priority=0, name=f"d_paper_final/{label}/it{it:05d}_pm8", case=r["case"], scheme="d",
                        mass=ORIGINAL_MASS[r["case"]], checkpoint=c["path"], checkpoint_sha256=c["sha256"], level=8,
                        task_book="T03", run=label))
    return out


def recheck_jobs():
    """T03 C v2 same-GPU recheck (Kevin 2026-11-12, item 3a): recheck.json {"jobs": [{name, label, checkpoint, sha256, priority}]}
    lists cloud-evaluated C v2 checkpoints, fetched to exports_system1/train_partial/ with sha256 checked, re-evaluated here with
    the same evaluator and states (+/-2)."""
    f = OUT / "recheck.json"
    if not f.exists():
        return []
    out = []
    for e in json.loads(f.read_text())["jobs"]:
        meta = parse(e["label"])
        out.append(dict(priority=e.get("priority", -1), name=e["name"], case=meta["case"], scheme=meta["scheme"], mass=meta["mass"],
                        checkpoint=e["checkpoint"], checkpoint_sha256=e["sha256"], level=2, task_book="T03", run=e["label"], recheck=True))
    return out


SHARED_PREFIXES = ("d_curve/",)  # Kevin 2026-11-11: the cloud slot filler may take these too
TRANSFER = OUT / "cloud_transfer.json"
TRANSFER_FETCHED = OUT / "cloud_transfer_fetched.json"


def fetch_transfer():
    """Copy the cloud's hand-over list once it exists, then the C v2 checkpoints it names (sha256 checked)."""
    if not TRANSFER.exists():
        rc, out = ssh(f"test -f {REMOTE_EVALS}/ops2/transfer_to_laptop.json && cat {REMOTE_EVALS}/ops2/transfer_to_laptop.json", timeout=300)
        if rc != 0 or not out.strip():
            return
        json.loads(out)  # complete JSON only
        TRANSFER.write_text(out)
        print(now(), "cloud hand-over list copied:", len(json.loads(out)["jobs"]), "jobs", flush=True)
    got = json.loads(TRANSFER_FETCHED.read_text()) if TRANSFER_FETCHED.exists() else {}
    for r in json.loads(TRANSFER.read_text())["jobs"]:
        if r["kind"] != "c2" or r["name"] in got:
            continue
        dst = EXPORT / "cloud_transfer" / r["run"] / Path(r["checkpoint"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.exists() and sha(dst) == r["checkpoint_sha256"]):
            subprocess.run(SCP + [f"root@connect.nmb1.seetacloud.com:{r['checkpoint']}", str(dst)], check=True, timeout=1800)
        if sha(dst) != r["checkpoint_sha256"]:
            raise RuntimeError(f"sha mismatch after copy: {dst}")
        got[r["name"]] = dict(checkpoint=str(dst), sha256=r["checkpoint_sha256"], fetched=now())
        TRANSFER_FETCHED.write_text(json.dumps(got, indent=1))


def transfer_jobs():
    """C v2 curve evaluations handed over by the cloud (fetched checkpoints only)."""
    if not (TRANSFER.exists() and TRANSFER_FETCHED.exists()):
        return []
    got = json.loads(TRANSFER_FETCHED.read_text())
    out = []
    for i, r in enumerate(json.loads(TRANSFER.read_text())["jobs"]):
        if r["kind"] != "c2" or r["name"] not in got:
            continue
        scheme = r["run"].split("_")[0]
        out.append(dict(priority=2, order=i, name=r["name"], case=r["case"], scheme=scheme,
                        mass=r["object_mass_kg"] if r["object_mass_kg"] is not None else ORIGINAL_MASS[r["case"]],
                        checkpoint=got[r["name"]]["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"], level=r["level"],
                        states_sha256_required=r["states_sha256"], task_book="T03", run=r["run"], cloud_transfer=True))
    return out


def handover_names():
    """D curve jobs the cloud handed over: {name: checkpoint sha256 on the cloud}."""
    if not TRANSFER.exists():
        return {}
    return {r["name"]: r["checkpoint_sha256"] for r in json.loads(TRANSFER.read_text())["jobs"] if r["kind"] == "d_curve"}


def apply_priority_overrides(jobs):
    """priority_overrides.json {job name: priority} (Kevin 2026-11-11: D it04050 of V3-25-27 as soon as the laptop is free).
    Absent file = unchanged order."""
    f = OUT / "priority_overrides.json"
    if not f.exists():
        return jobs
    try:
        ov = json.loads(f.read_text())
    except json.JSONDecodeError:
        return jobs
    return [dict(j, priority=ov[j["name"]], priority_overridden=True) if j["name"] in ov else j for j in jobs]


def b_jobs():
    if not B_ENABLE.exists():
        return []
    frac = json.loads(B_ENABLE.read_text())["last_waypoint_frac"]
    spec = [(c, None, v, lvl) for lvl in (2, 8) for v in B_MAIN for c in B_TASKS]
    spec += [(c, None, "B-open", 2) for c in B_TASKS]                       # section 7.1: +/-2 only
    spec += [(c, m, v, lvl) for c, ms in B_MASS.items() for m in ms for v in B_MASS_VARIANTS for lvl in (2, 8)]
    out = []
    for i, (c, m, v, lvl) in enumerate(spec):
        tag = f"{c}_{v}" + ("" if m is None else f"_m{m:g}kg".replace(".", "p"))
        out.append(dict(priority=2, order=i, kind="b", name=f"B/{'main' if m is None else 'mass'}/{tag}_pm{lvl}", case=c,
                        scheme="b", variant=v, mass=ORIGINAL_MASS.get(c) if m is None else m, mass_override=m, level=lvl,
                        last_waypoint_frac=frac, task_book="T02" if m is None else "T04", run=f"scheme_b_{v}"))
    return out


def b_frac_committed(frac):
    """The evaluator imports system1/scheme_b.py at launch: it must hold the enabled value, committed."""
    src = (ROOT / "system1/scheme_b.py").read_text()
    m = re.search(r"^LAST_WAYPOINT_FRAC = ([0-9.]+)", src, re.M)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "system1/scheme_b.py"], cwd=ROOT).returncode != 0
    return m is not None and float(m.group(1)) == float(frac) and not dirty


def evaluator_count():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    n = 0
    for l in out.splitlines():
        tok = l.split()
        if tok and "python" in os.path.basename(tok[0]) and any(
                k in t for t in tok[1:4] for k in (EVAL_PATTERN, "validate_single.py", "validate_planner_single.py",
                                                "train_plan5_push.py", "train_core4d_smalltable.py")):
            n += 1
    return n


def done(job):
    ex = OUT / f"{job['name']}.exit.json"
    return ex.exists() and json.loads(ex.read_text()).get("completed")


def launch(job, slot):
    out_dir = OUT / job["name"]
    rec = OUT / job["name"]
    rec.parent.mkdir(parents=True, exist_ok=True)
    states, expect = STATES[(job["case"], job["level"])]
    if sha(states) != expect:
        raise RuntimeError(f"frozen state hash mismatch {states}")
    if job.get("states_sha256_required") and job["states_sha256_required"] != expect:
        raise RuntimeError(f"frozen states differ from the cloud's: {states}")
    if job.get("checkpoint_sha256") and sha(job["checkpoint"]) != job["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch {job['checkpoint']}")
    if out_dir.exists():  # earlier attempt without a completed exit record: keep it, renamed
        k = 1
        while out_dir.with_name(f"{out_dir.name}.attempt{k}").exists():
            k += 1
        out_dir.rename(out_dir.with_name(f"{out_dir.name}.attempt{k}"))
        for s in (".launch.json", ".stdout.log", ".exit.json"):
            p = Path(f"{rec}{s}")
            if p.exists():
                p.rename(Path(f"{rec}.attempt{k}{s}"))
    if job["scheme"] == "b":
        if not b_frac_committed(job["last_waypoint_frac"]):
            raise RuntimeError("scheme_b.LAST_WAYPOINT_FRAC on disk differs from B_ENABLED.json or is uncommitted")
        argv = [PY, "-B", EVAL_PATTERN, "--case", job["case"], "--controller", "sonic_planner", "--b-variant", job["variant"],
                "--device", "cuda:0", "--states", str(states), "--model-id", "SchemeB_" + job["name"].split("/")[-1],
                "--output-dir", str(out_dir)]
        if job["case"] == "push":
            argv += ["--push-placement-proof", str(PUSH_PROOF)]
        if job["mass_override"] is not None:
            argv += ["--object-mass-kg", f"{job['mass_override']:g}"]
    else:
        argv = [PY, "-B", EVAL_PATTERN, "--case", job["case"], "--controller", "sonic_residual" if job["scheme"] in C_SCHEMES else "policy",
                "--checkpoint", job["checkpoint"], "--device", "cuda:0", "--states", str(states),
                "--model-id", job["name"].replace("/", "__"), "--output-dir", str(out_dir)]
        if job["mass"] != ORIGINAL_MASS[job["case"]]:
            argv += ["--object-mass-kg", f"{job['mass']:g}"]  # retrained at this mass: contract checked at the evaluated mass
    cpus = ("16-23", "24-31")[slot]
    argv = ["taskset", "-c", cpus] + argv
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    Path(f"{rec}.launch.json").write_text(json.dumps(dict(
        job=job, argv=argv, started=now(), checkpoint_sha256=sha(job["checkpoint"]) if job.get("checkpoint") else None,
        states=str(states), states_sha256=expect,
        local_rank=81 + slot, gpu_model=gpu, git_head=git("rev-parse", "HEAD").strip(), git_status=git("status", "--porcelain"),
        git_diff=git("diff", "HEAD")), indent=1))
    env = dict(os.environ, LOCAL_RANK=str(81 + slot), WORLD_SIZE="1")
    log = open(f"{rec}.stdout.log", "w")
    return subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env), log


def finish(job, proc):
    summ = OUT / job["name"] / "summary.json"
    completed = proc.returncode == 0 and summ.exists() and bool(json.loads(summ.read_text()).get("complete"))
    Path(f"{OUT / job['name']}.exit.json").write_text(json.dumps(dict(name=job["name"], returncode=proc.returncode,
                                                                      completed=completed, ended=now()), indent=1))
    print(now(), "end", job["name"], "rc", proc.returncode, "completed", completed, flush=True)
    return completed


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    fetched = {}
    run_failures, job_failures = {}, {}
    running = {}  # slot -> (job, proc, log)
    claimed_elsewhere = set()
    last_poll = 0.0
    while True:
        if time.time() - last_poll > a.poll_seconds:
            last_poll = time.time()
            try:
                dn, bad = finished_remote()
                for label in dn:
                    if label not in fetched:
                        d = fetch_run(label)
                        fetched[label] = jobs_for_run(label, d)
                        print(now(), "fetched", label, len(fetched[label]), "jobs", flush=True)
                for label in bad:
                    if label not in run_failures:
                        run_failures[label] = now()
                        print(now(), "cloud run ended without completion:", label, flush=True)
            except Exception as e:  # network trouble: retry at the next poll
                print(now(), "poll error", repr(e)[:300], flush=True)
            if a.cloud_transfer:
                try:
                    fetch_transfer()
                except Exception as e:  # retry at the next poll
                    print(now(), "hand-over fetch error", repr(e)[:300], flush=True)
        handed = handover_names() if a.cloud_transfer else {}
        pending = ([j for js in fetched.values() for j in js] + recheck_jobs() + d_paper_pm8_jobs() + early_curve_jobs() + b_jobs()
                   + d_curve_jobs() + (transfer_jobs() if a.cloud_transfer else []))
        pending = apply_priority_overrides(pending)
        busy = {r[0]["name"] for r in running.values()}
        pending = [j for j in pending if j["name"] not in claimed_elsewhere or j["name"] in handed]
        pending = sorted((j for j in pending if not done(j) and j["name"] not in busy and job_failures.get(j["name"], 0) < 2),
                         key=lambda j: (j["priority"], j.get("order", 0), j["name"]))
        if a.dry_run:
            for j in pending:
                print(j["priority"], j["name"], j.get("checkpoint", ""))
            return 0
        for s, (job, proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                if not finish(job, proc):
                    job_failures[job["name"]] = job_failures.get(job["name"], 0) + 1
                del running[s]
        free = [s for s in range(a.slots) if s not in running]
        if pending and free and evaluator_count() < a.slots and not PAUSE.exists():
            job = pending[0]
            if job["name"] in handed:  # handed over by the cloud: no new claim; same checkpoint required
                if job.get("checkpoint_sha256") != handed[job["name"]]:
                    job_failures[job["name"]] = 99
                    print(now(), "hand-over checkpoint sha256 differs, skip", job["name"], flush=True)
                    continue
            elif job["name"].startswith(SHARED_PREFIXES):  # shared with the cloud slot filler: claim first (system1/cloud/claims.py)
                from system1.cloud.claims import claim_via_ssh
                c = claim_via_ssh(SSH, job["name"], "laptop", dict(gpu="NVIDIA GeForce RTX 5070 Laptop GPU"))
                if c is False:
                    claimed_elsewhere.add(job["name"])
                    print(now(), "claimed by the cloud, skip", job["name"], flush=True)
                    continue
                if c is None:
                    time.sleep(60)
                    continue
            proc, log = launch(job, free[0])
            running[free[0]] = (job, proc, log)
            print(now(), "start", job["name"], "slot", free[0], flush=True)
            time.sleep(20)
            continue
        time.sleep(15)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("catalog", "fetch-v3", "run"))
    p.add_argument("--slots", type=int, default=2)
    p.add_argument("--poll-seconds", type=int, default=600)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cloud-transfer", action="store_true", help="take over the cloud's unfinished evaluations (transfer_to_laptop.json)")
    p.add_argument("--skip-missing", action="store_true", help="catalog: leave out V3-25-27 runs whose copy is unfinished")
    a = p.parse_args()
    if a.mode == "catalog":
        catalog(a.skip_missing)
    elif a.mode == "fetch-v3":
        fetch_v3()
    else:
        return run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
