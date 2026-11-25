#!/usr/bin/env python3
"""T13 v4 round 2 (task book section 8 item 2): local smoke of C-out-Table / C-out-Push on the laptop 5070 before "可以开卡".

The command lines come from cloud_runs.py --dry-run (the cloud plan, 4096 environments); only the environment count, the
iteration counts, the save interval, the output directories and the interpreter are replaced (an 8 GB laptop GPU cannot hold
4096 environments). Per run:
  1. training stages as planned (c_out_push: critic-only warm-up, then actor + critic with --resume), shortened;
  2. checkpoint checks: C-out kind and round-2 marker in the checkpoint, MAPPO sizes (actor 159, critic 585), the actor's output
     layer is zero at iteration 0 (and still zero after the critic-only warm-up);
  3. iteration-0 check (as T03 C v2): the zero-output C-out checkpoint reproduces scheme A (--controller sonic) on the first 3
     validation states, every trajectory array identical (system1/tests/t03_iter0_compare.py);
  4. the cloud validation command (--controller sonic_residual --t13v4-cout, observation v1, noise off) on the last smoke
     checkpoint, first 3 validation states: complete summary;
  5. metrics: finite, contact v1 and scheme C diagnostics present; s/it; peak GPU memory (nvidia-smi, 1 s polling).
Writes logs_system1/t13v4/round2/smoke_local/<run>/smoke.json.

    python3 system1/t13v4/round2_smoke_local.py --run c_out_table --num-envs 512 --iterations 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "system1/t13v4"))
from cloud_smoke import metrics_summary, overflow_lines, sha  # noqa: E402

PY = "/home/kevin/.holosoma_deps/miniconda3/envs/hssim_s1/bin/python"
OUT = ROOT / "logs_system1/t13v4/round2/smoke_local"


class GpuPeak:
    def __init__(self):
        self.peak, self.stop = 0, False
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self.stop:
            try:
                v = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                self.peak = max(self.peak, int(v.split()[0]))
            except Exception:
                pass
            time.sleep(1)

    def done(self):
        self.stop = True
        self.t.join()
        return self.peak


def setv(argv, k, v):
    argv = list(argv)
    argv[argv.index(k) + 1] = str(v)
    return argv


def ev(argv, log, env):
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, round(time.time() - t0, 1)


def ckpt_checks(path, warm_or_zero):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    actor = st["actor_model_state_dict"]
    lin = [k for k, v in actor.items() if k.endswith("weight") and v.ndim == 2]
    last_w, last_b = actor[lin[-1]], actor[lin[-1].replace("weight", "bias")]
    first = [v for k, v in actor.items() if k.endswith("weight") and v.ndim == 2][0]
    meta = st.get("plan5_mappo") or st.get("core4d_smalltable_mappo") or {}
    out = dict(file=str(path), sha256=sha(path), sonic_residual=st.get("system1_sonic_residual"), kind=st.get("system1_sonic_residual_kind"),
               t13v4_cout=st.get("system1_t13v4_cout"), actor_input_dim=int(first.shape[1]), output_layer=lin[-1], output_shape=list(last_w.shape),
               output_layer_abs_max=float(max(last_w.abs().max(), last_b.abs().max())), mappo_meta={k: meta.get(k) for k in ("actor_obs_dim", "critic_obs_dim", "num_agents", "action_dim")},
               t14_reference=(st.get("system1_t14_reference") or {}).get("manifest_sha256"))
    out["passed"] = bool(out["sonic_residual"] and out["kind"] == "out" and out["t13v4_cout"] and out["actor_input_dim"] == 159
                         and out["output_shape"][0] == 29 and (out["output_layer_abs_max"] == 0.0 if warm_or_zero else True))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=("c_out_table", "c_out_push"), required=True)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup-iterations", type=int, default=3)
    ap.add_argument("--save-interval", type=int, default=10)
    a = ap.parse_args()
    plan = json.loads(subprocess.run([sys.executable, str(ROOT / "system1/t13v4/cloud_runs.py"), "--run", a.run, "--num-envs", "4096", "--out", "/cloud",
                                      "--python", PY, "--dry-run"], capture_output=True, text=True, check=True).stdout)
    d = OUT / a.run
    d.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", WORLD_SIZE="1")
    rec = dict(run=a.run, num_envs=a.num_envs, iterations=a.iterations, warmup_iterations=a.warmup_iterations, cloud_plan=plan, stages=[],
               git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               git_status=subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True).stdout)
    ok, warm_ck = True, None
    for st in plan["stages"]:
        argv = setv(setv(st["argv"], "--num-envs", a.num_envs), "--output-dir", d / st["name"])
        if st["name"] == "train_warmup":
            argv = setv(setv(argv, "--iterations", a.warmup_iterations), "--save-interval", a.warmup_iterations)
        else:
            argv = setv(setv(argv, "--iterations", a.iterations), "--save-interval", a.save_interval)
            if "--resume" in argv:
                argv = setv(argv, "--resume", warm_ck)
        gp = GpuPeak()
        rc, sec = ev(argv, d / f"{st['name']}.stdout.log", env)
        out_dir = d / st["name"]
        cks = sorted(out_dir.glob("model_*.pt"))
        ms = [json.loads(x) for x in (out_dir / "metrics.jsonl").read_text().splitlines() if x.strip()] if (out_dir / "metrics.jsonl").exists() else []
        keys = sorted({k for m in ms for k in m})
        s = dict(name=st["name"], argv=argv, returncode=rc, seconds=sec, gpu_peak_mib=gp.done(), checkpoints=[c.name for c in cks],
                 overflow_lines=overflow_lines(d / f"{st['name']}.stdout.log"), contact_v1_keys=[k for k in keys if k.startswith("ContactV1/")][:20],
                 scheme_c_keys=[k for k in keys if "SchemeC" in k or k.startswith("C/") or "sonic" in k.lower()][:20], **metrics_summary(out_dir / "metrics.jsonl"))
        if cks:
            s["first_checkpoint"] = ckpt_checks(cks[0], cks[0].name == "model_00000.pt" or st["name"] == "train_warmup")
            s["last_checkpoint"] = ckpt_checks(cks[-1], st["name"] == "train_warmup")
        rec["stages"].append(s)
        print(st["name"], rc, sec, s.get("s_per_it_median_after_3"), s["gpu_peak_mib"], s["nonfinite_keys"], flush=True)
        if rc != 0 or not cks or s["nonfinite_keys"] or not s["first_checkpoint"]["passed"] or not s["last_checkpoint"]["passed"]:
            ok = False
            break
        if st["name"] == "train_warmup":
            warm_ck = cks[-1]
    task = plan["task"]
    if ok:  # iteration-0 check: the zero-output checkpoint (table: model_00000.pt of the main stage; push: the warm-up's last) vs scheme A
        zero = d / "train" / "model_00000.pt" if task == "table" else Path(warm_ck)
        root = d / "iter0"
        root.mkdir()
        vf = [x for x in plan["validation_flags"] if x not in ("--obs-v1", "--critic-v1")]  # scheme A has no policy observation
        common = ["--case", task, "--device", "cuda:0", "--states", plan["validation_states"], "--limit", "3"]
        rA, _ = ev([PY, "-B", "scripts/evaluate_paper_supplement.py", *common, *vf, "--controller", "sonic", "--model-id", f"r2_iter0_{task}_A",
                    "--output-dir", str(root / f"{task}_sonic")], d / "iter0_A.log", env)
        rC, _ = ev([PY, "-B", "scripts/evaluate_paper_supplement.py", *common, *plan["validation_flags"], *plan["validation_extra"], "--controller",
                    "sonic_residual", "--checkpoint", str(zero), "--model-id", f"r2_iter0_{task}_cout", "--output-dir", str(root / f"{task}_sonic_residual")],
                   d / "iter0_C.log", env)
        rcmp, _ = ev([PY, "-B", "system1/tests/t03_iter0_compare.py", "--root", str(root), "--cases", task, "--checkpoints", f"{task}={zero}",
                      "--out", str(d / "iter0_vs_A.json")], d / "iter0_compare.log", env)
        cmp_ = json.loads((d / "iter0_vs_A.json").read_text()) if (d / "iter0_vs_A.json").exists() else {}
        rec["iter0"] = dict(checkpoint=str(zero), rc_A=rA, rc_C=rC, rc_compare=rcmp, passed=cmp_.get("passed"), result=cmp_.get("cases"))
        ok = ok and bool(cmp_.get("passed"))
        print("iter0", rA, rC, rcmp, cmp_.get("passed"), flush=True)
    if ok:  # the cloud validation command on the last smoke checkpoint
        ck = d / "train" / rec["stages"][-1]["checkpoints"][-1]
        argv = [PY, "-B", "scripts/evaluate_paper_supplement.py", "--case", task, "--controller", plan["validation_controller"], *plan["validation_flags"],
                *plan["validation_extra"], "--checkpoint", str(ck), "--device", "cuda:0", "--states", plan["validation_states"], "--limit", "3",
                "--model-id", f"t13v4_round2_smoke__{a.run}", "--output-dir", str(d / "validation")]
        gp = GpuPeak()
        rc, sec = ev(argv, d / "validation.stdout.log", env)
        sm = json.loads((d / "validation/summary.json").read_text()) if (d / "validation/summary.json").exists() else {}
        mf = json.loads((d / "validation/run_manifest.json").read_text()) if (d / "validation/run_manifest.json").exists() else {}
        rec["validation"] = dict(argv=argv, returncode=rc, seconds=sec, gpu_peak_mib=gp.done(), complete=sm.get("complete"), episodes=len(sm.get("episodes", [])),
                                 system1_t13v4=mf.get("system1_t13v4"), scheme_c_kind=mf.get("system1_scheme_c_kind"), controller=mf.get("controller"))
        ok = rc == 0 and sm.get("complete") is True and len(sm.get("episodes", [])) == 3
        print("validation", rc, sec, sm.get("complete"), flush=True)
    rec["passed"] = ok
    (d / "smoke.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("SMOKE", a.run, "passed" if ok else "FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
