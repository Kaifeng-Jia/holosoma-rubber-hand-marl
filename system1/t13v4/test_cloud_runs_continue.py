#!/usr/bin/env python3
"""Smoke test of cloud_runs.py's section 9 continuation (CONT runs) with a stub interpreter instead of Isaac (seconds, no GPU):
  1. d_table_to10k, stub metrics equal to the pre-stop window: no warm-up stage, training argv = --resume <ckpt> --iterations
     (max_it - 7500), resume check passed, validations only at counted 8000 / 9000 / 10000, status finished, final 10000;
  2. d_push_to10k (warm-up 25 at 4096, Push path: no resume sha256 in run_config, checkpoints counted from the resume): passed,
     validations at 7025 ... 10025 (counted 7000 ... 10000), final 10025;
  3. d_table_to10k with the stub reward x10: resume check failed -> SIGINT -> status "stopped: resume check failed";
  4. a wrong checkpoint file (sha256 differs from continue_ref): refused before any launch.
The stub (written to a temp dir) imitates the training scripts' outputs (run_config.json, metrics.jsonl, model_*.pt) and the
evaluator's summary.json. Read only on the repository; temp dirs only.

    python3 system1/t13v4/test_cloud_runs_continue.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUB = r'''#!/usr/bin/env python3
import json, os, signal, sys, time
from pathlib import Path
a = sys.argv[1:]
if a[0] == "-B":
    a = a[1:]
script, args = a[0], a[1:]
g = lambda f, d=None: args[args.index(f) + 1] if f in args else d
if script.endswith("evaluate_paper_supplement.py"):
    out = Path(g("--output-dir")); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(complete=True, episodes=[dict(kind="perturbed", completed_reference=True)] * 3)))
    sys.exit(0)
signal.signal(signal.SIGINT, lambda *x: sys.exit(130))
out = Path(g("--output-dir")); out.mkdir(parents=True, exist_ok=True)
ref = json.loads(Path(os.environ["STUB_REF"]).read_text())["before"]
it0 = int(Path(g("--resume")).stem.split("_")[1])
n, save, envs = int(g("--iterations")), int(g("--save-interval")), int(g("--num-envs"))
push = "train_plan5_push.py" in script
cfg = dict(num_envs=envs, start_iteration=it0)
if not push:
    cfg["resume_checkpoint_sha256"] = os.environ["STUB_SHA"]
(out / "run_config.json").write_text(json.dumps(cfg))
(out / f"model_{it0:05d}.pt").write_text("x")
k = float(os.environ.get("STUB_FACTOR", "1"))
with open(out / "metrics.jsonl", "w") as f:
    for it in range(it0 + 1, it0 + n + 1):
        f.write(json.dumps(dict(iteration=it, reward_mean=ref["reward_mean"] * k, value_loss=ref["value_loss"], surrogate_loss=-ref["surrogate_loss_abs"],
                                entropy=ref["entropy"], reset_count=ref["resets_per_env"] * envs, actor_learning_rate=ref["actor_learning_rate_mean"],
                                critic_learning_rate=ref["critic_learning_rate"])) + "\n")
        f.flush()
        if ((it - it0) if push else it) % save == 0 or it == it0 + n:
            (out / f"model_{it:05d}.pt").write_text("x")
        time.sleep(float(os.environ.get("STUB_SLEEP", "0.002")))
'''


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(tmp, name, run_name, factor=1.0, wrong_ck=False, sleep="0.002"):
    t = tmp / name
    t.mkdir()
    ref = json.loads((ROOT / "logs_system1/t13v4/continue" / run_name / "continue_ref.json").read_text())
    ck = t / f"model_{ref['from_iteration']:05d}.pt"
    ck.write_bytes(b"checkpoint bytes")
    ref["checkpoint_sha256"] = sha(ck) if not wrong_ck else "0" * 64
    (t / "continue_ref.json").write_text(json.dumps(ref))
    env = dict(os.environ, STUB_REF=str(t / "continue_ref.json"), STUB_SHA=sha(ck), STUB_FACTOR=str(factor), STUB_SLEEP=sleep)
    p = subprocess.run([sys.executable, str(ROOT / "system1/t13v4/cloud_runs.py"), "--run", run_name, "--num-envs", "4096", "--out", str(t / "runs"),
                        "--python", str(tmp / "stub.py"), "--poll", "0.2", "--continue-checkpoint", str(ck), "--continue-ref", str(t / "continue_ref.json")],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    d = t / "runs" / run_name
    st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else None
    dec = json.loads((d / "decision.json").read_text()) if (d / "decision.json").exists() else None
    launch = json.loads((d / "launch.json").read_text()) if (d / "launch.json").exists() else None
    return p, st, dec, launch


def main():
    with tempfile.TemporaryDirectory(dir=ROOT / "logs_system1/t13v4") as tmp:
        tmp = Path(tmp)
        (tmp / "stub.py").write_text(STUB)
        (tmp / "stub.py").chmod(0o755)
        # 1
        p, st, dec, L = run(tmp, "t1", "d_table_to10k")
        assert p.returncode == 0, p.stderr[-2000:]
        assert [s["name"] for s in L["stages"]] == ["train"] and L["stages"][0]["argv"][-2] == "--resume"
        assert L["stages"][0]["argv"][L["stages"][0]["argv"].index("--iterations") + 1] == "2500"
        assert st["resume_check"]["passed"], st["resume_check"]
        assert sorted(int(k) for k in st["curve"]) == [8000, 9000, 10000], st["curve"]
        assert dec["status"] == "finished" and dec["final_checkpoint_iteration"] == 10000, dec
        # 2
        p, st, dec, L = run(tmp, "t2", "d_push_to10k")
        assert p.returncode == 0, p.stderr[-2000:]
        assert [s["name"] for s in L["stages"]] == ["train"]
        assert st["resume_check"]["passed"] and st["resume_check"]["resume_sha256_source"] == "file passed to --resume", st["resume_check"]
        assert sorted(int(k) for k in st["curve"]) == [7000, 8000, 9000, 10000], st["curve"]
        assert dec["status"] == "finished" and dec["final_checkpoint_iteration"] == 10025 and dec["final_checkpoint_counted_iteration"] == 10000, dec
        # 3
        p, st, dec, L = run(tmp, "t3", "d_table_to10k", factor=10.0, sleep="0.05")
        assert st["resume_check"] is not None and not st["resume_check"]["passed"], st["resume_check"]
        assert not st["resume_check"]["items"]["ratio_reward_mean"] and dec["status"] == "stopped: resume check failed", dec["status"]
        assert dec["final_checkpoint_iteration"] < 10000
        # 4
        p, st, dec, L = run(tmp, "t4", "d_table_to10k", wrong_ck=True)
        assert p.returncode != 0 and "sha256" in p.stderr and L is None, (p.returncode, p.stderr[-300:])
    print("CLOUD_RUNS_CONTINUE_SMOKE_OK")


if __name__ == "__main__":
    main()
