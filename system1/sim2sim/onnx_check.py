#!/usr/bin/env python3
"""T10 step 2.1: ONNX export and element-wise checks (sim2sim environment, CPU).

Actors (D and C-in, Table and Bin, seeds 721-723 = the checkpoints of the laptop-5070 finals, results_system1/T03/c2_main/
per_seed_summary.csv, sha256 checked): actor + observation normalisation exported to one ONNX graph
(x -> (x - mean) / (std + eps) -> MLP -> action mean, exactly deterministic_core4d_smalltable_actions), compared with PyTorch on
1,000 random observations (seed 0): half N(0,1), half mean + std * N(0,1) of the checkpoint's own normaliser. Pass: max |diff| < 1e-5.

SONIC: the official deployment files (nvidia/GEAR-SONIC model_encoder.onnx + model_decoder.onnx, whose HF config.json lists
sonic_release/last.pt as their training file) against the PyTorch release checkpoint as loaded in Isaac (sonic_torch_ref.py
outputs). Encoder input = deploy layout: encoder_mode_4 = [mode id (g1 = 0), 0, 0, 0] followed by the Isaac tokenizer without its
3-wide one-hot (same field order and sizes); decoder input = [token 64, proprioception 930]. Pass: max |diff| < 1e-5.

    sim2sim_s1/bin/python -B system1/sim2sim/onnx_check.py --out results_system1/T10_sim2sim/consistency --onnx-dir exports_system1/t10/onnx
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/holosoma"))
HF = Path("/home/kevin/third_party/GR00T-WholeBodyControl/sonic_hf_release")
TOL = 1e-5


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checkpoints(tasks=("table", "bucket")):
    rows = list(csv.DictReader((ROOT / "results_system1/T03/c2_main/per_seed_summary.csv").open()))
    rows += [r for r in csv.DictReader((ROOT / "results_system1/T03/d_final_5070/per_seed_summary.csv").open())]  # D Push / Kick / Chair
    out = []
    for r in rows:
        if r["perturbation_level_cm"] != "2" or r["variant"] not in ("D", "C-in") or r["task"] not in tasks:
            continue
        s = json.loads((ROOT / r["eval_dir"] / "summary.json").read_text())
        assert sha(s["checkpoint"]) == r["checkpoint_sha256"] == s["checkpoint_sha256"]
        out.append(dict(scheme=r["variant"], task=r["task"], seed=int(r["training_seed"]), checkpoint=s["checkpoint"], checkpoint_sha256=r["checkpoint_sha256"]))
    return out


class ActorOnnx(torch.nn.Module):
    def __init__(self, actor, norm):
        super().__init__()
        self.net = actor.actor_module.module
        self.register_buffer("mean", norm._mean.detach().clone())
        self.register_buffer("den", (norm._std + norm.eps).detach().clone())

    def forward(self, obs):
        return self.net((obs - self.mean) / self.den)


def algo_config(task):
    """The algo config the evaluator builds the actor from (scripts/evaluate_paper_supplement.py)."""
    if task in ("push", "pull"):
        from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_pull_baseline, g1_29dof_plan5_push_baseline
        return (g1_29dof_plan5_push_baseline if task == "push" else g1_29dof_plan5_pull_baseline).algo.config
    if task == "kick":
        from holosoma.config_values.marl.g1.kick_experiment import g1_29dof_plan5_kick_baseline
        return g1_29dof_plan5_kick_baseline.algo.config
    from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_smoke
    return g1_29dof_core4d_smalltable_smoke.algo.config


def extra_checkpoints(specs):
    """--extra-checkpoint SCHEME,TASK,SEED,NAME,PATH[,v1] (T10b: a pilot's final checkpoint); NAME is the ONNX file stem.
    T13 v4: a sixth field "v1" = a checkpoint trained with observation v1 (actor input 154 + teammate 5 = 159)."""
    out = []
    for s in specs:
        f = s.split(",")
        scheme, task, seed, name, path = f[:5]
        v1 = len(f) > 5 and f[5] == "v1"
        out.append(dict(scheme=scheme, task=task, seed=int(seed), name=name, checkpoint=str(Path(path).resolve()), checkpoint_sha256=sha(path), obs_v1=v1))
    return out


def actors(onnx_dir, rows_out, tasks=("table", "bucket"), cks=None):
    import onnxruntime as ort
    from holosoma.agents.mappo.core4d_smalltable_initialization import initialize_core4d_smalltable_model_bundle
    for c in (checkpoints(tasks) if cks is None else cks):
        if c.get("obs_v1"):  # T13 v4 (off = unchanged): the evaluator's v1 bundle (teammate 5, critic 585)
            sys.path.insert(0, str(ROOT))
            from system1.t13v4.mappo_v1 import initialize_core4d_v1_model_bundle
            m = initialize_core4d_v1_model_bundle(algo_config(c["task"]), teammate=5, critic=585, device="cpu")
            dim = 159
        else:
            m = initialize_core4d_smalltable_model_bundle(algo_config(c["task"]), device="cpu")
            dim = 158
        ck = torch.load(c["checkpoint"], map_location="cpu", weights_only=False)
        m.actor.load_state_dict(ck["actor_model_state_dict"], strict=True)
        m.actor_obs_normalizer.load_state_dict(ck["actor_obs_normalizer_state_dict"], strict=True)
        m.actor.eval()
        m.actor_obs_normalizer.eval()
        wrap = ActorOnnx(m.actor, m.actor_obs_normalizer).eval()
        name = c.get("name") or f"{c['scheme'].replace('-', '').lower()}_{c['task']}_seed{c['seed']}"
        path = Path(onnx_dir) / f"{name}.onnx"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(wrap, torch.zeros(1, dim), str(path), input_names=["obs"], output_names=["action"],
                          dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
        rng = np.random.default_rng(0)
        mean = m.actor_obs_normalizer._mean.numpy()
        std = m.actor_obs_normalizer._std.numpy()
        x = np.concatenate([rng.standard_normal((500, dim)), mean + std * rng.standard_normal((500, dim))]).astype(np.float32)
        with torch.no_grad():  # the evaluator path: normaliser(update=False) then act_inference
            ref = m.actor.act_inference({"actor_obs": m.actor_obs_normalizer(torch.from_numpy(x), update=False)}).numpy()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"obs": x})[0]
        err = float(np.abs(got - ref).max())
        # float64 control (same graph, double precision): separates conversion errors from float32 rounding
        w64 = ActorOnnx(m.actor, m.actor_obs_normalizer).double().eval()
        p64 = Path(onnx_dir) / f"{name}_float64_check.onnx"
        torch.onnx.export(w64, torch.zeros(1, dim, dtype=torch.float64), str(p64), input_names=["obs"], output_names=["action"],
                          dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
        with torch.no_grad():
            ref64 = w64(torch.from_numpy(x.astype(np.float64))).numpy()
        from onnx.reference import ReferenceEvaluator  # onnxruntime CPU has no float64 Elu; the ONNX reference evaluator runs the graph
        got64 = ReferenceEvaluator(str(p64)).run(None, {"obs": x.astype(np.float64)})[0]
        half = len(x) // 2
        rows_out.append(dict(item="actor", name=name, scheme=c["scheme"], task=c["task"], seed=c["seed"], n=len(x), **({"input_dim": dim} if c.get("obs_v1") else {}), max_abs_diff=err, tol=TOL,
                             passed=err < TOL, max_abs_diff_N01_half=float(np.abs(got - ref)[:half].max()),
                             max_abs_diff_normaliser_half=float(np.abs(got - ref)[half:].max()), max_abs_action=float(np.abs(ref).max()),
                             max_rel_diff=float((np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3)).max()),
                             float64_onnx_vs_torch=float(np.abs(got64 - ref64).max()), torch32_vs_torch64=float(np.abs(ref - ref64).max()),
                             onnx32_vs_torch64=float(np.abs(got - ref64).max()),
                             checkpoint=c["checkpoint"], checkpoint_sha256=c["checkpoint_sha256"], onnx=str(path), onnx_sha256=sha(path)))
        print(name, err, rows_out[-1]["float64_onnx_vs_torch"], rows_out[-1]["torch32_vs_torch64"], rows_out[-1]["onnx32_vs_torch64"], flush=True)


def sonic(ref_path, rows_out, tag):
    import onnxruntime as ort
    z = np.load(ref_path)
    enc = ort.InferenceSession(str(HF / "model_encoder.onnx"), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(HF / "model_decoder.onnx"), providers=["CPUExecutionProvider"])
    layout = json.loads(str(z["layout"]))
    assert layout[0][0] == "encoder_index" and layout[0][1] == 3
    for k in [k[:-len("_tokenizer")] for k in z.files if k.endswith("_tokenizer")]:
        tz, ao = z[f"{k}_tokenizer"], z[f"{k}_actor_obs"]
        mode = np.argmax(tz[:, :3], axis=1).astype(np.float32)
        x = np.concatenate([mode[:, None], np.zeros((len(tz), 3), np.float32), tz[:, 3:]], axis=1).astype(np.float32)
        toks = np.concatenate([enc.run(None, {"obs_dict": x[i:i + 1]})[0] for i in range(len(x))])
        acts = np.concatenate([dec.run(None, {"obs_dict": np.concatenate([toks[i:i + 1], ao[i:i + 1]], axis=1).astype(np.float32)})[0]
                               for i in range(len(x))])
        et = float(np.abs(toks - z[f"{k}_token"]).max())
        ea = float(np.abs(acts - z[f"{k}_action"]).max())
        extra = {}
        if f"{k}_action64" in z.files:  # PyTorch float64 control of the same checkpoint and inputs
            extra = dict(torch32_vs_torch64=float(np.abs(z[f"{k}_action"] - z[f"{k}_action64"]).max()),
                         onnx32_vs_torch64=float(np.abs(acts - z[f"{k}_action64"]).max()), max_abs_action=float(np.abs(z[f"{k}_action"]).max()))
        n_tok_mismatch = int((np.abs(toks - z[f"{k}_token"]).max(axis=1) > TOL).sum())
        for item, err in (("sonic_token", et), ("sonic_action", ea)):
            rows_out.append(dict(item=item, name=f"official_onnx_vs_last.pt_{tag}_{k}", n=len(x), max_abs_diff=err, tol=TOL, passed=err < TOL,
                                 samples_token_mismatch=n_tok_mismatch, encoder_onnx_sha256=sha(HF / "model_encoder.onnx"),
                                 decoder_onnx_sha256=sha(HF / "model_decoder.onnx"), reference=str(ref_path), reference_sha256=sha(ref_path),
                                 **(extra if item == "sonic_action" else {})))
        print(tag, k, "token", et, "action", ea, "token mismatches", n_tok_mismatch, extra, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--sonic-ref", default="logs_system1/t10/onnx/sonic_torch_ref_random.npz")
    ap.add_argument("--skip-actors", action="store_true")
    ap.add_argument("--tasks", default="table,bucket")
    ap.add_argument("--extra-checkpoint", action="append", default=[],
                    help="SCHEME,TASK,SEED,NAME,PATH: export and check only these actors (no SONIC check, no default list); off = unchanged")
    a = ap.parse_args()
    torch.set_num_threads(4)
    rows = []
    if a.extra_checkpoint:
        actors(a.onnx_dir, rows, cks=extra_checkpoints(a.extra_checkpoint))
    else:
        for p in [x for x in a.sonic_ref.split(",") if x]:
            sonic(ROOT / p, rows, Path(p).stem)
        if not a.skip_actors:
            actors(a.onnx_dir, rows, tuple(a.tasks.split(",")))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (a.out / "onnx_check.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("all passed" if all(r["passed"] for r in rows) else "FAILED: " + ", ".join(r["name"] + "/" + r["item"] for r in rows if not r["passed"]))
    f64 = [r["float64_onnx_vs_torch"] for r in rows if r["item"] == "actor"]
    if f64:  # T10s pass rule: float64 graph vs PyTorch <= 1e-10 (float32 differences are reported only)
        print("float64 graph max", max(f64), "pass" if max(f64) <= 1e-10 else "FAIL")


if __name__ == "__main__":
    main()
