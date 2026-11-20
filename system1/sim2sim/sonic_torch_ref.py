#!/usr/bin/env python3
"""T10 step 2.1 (SONIC): reference outputs of the SONIC release checkpoint loaded exactly as in Isaac (system1/sonic_controller.py,
sonic_release/last.pt), on CPU, to compare with the ONNX files run in the sim2sim environment (system1/sim2sim/onnx_check.py).

Inputs: 1,000 random observations (seed 0; actor_obs ~ N(0,1) [930]; tokenizer ~ N(0,1) [1761] with encoder_index one-hot g1),
plus, with --recorded, the SONIC inputs recorded during the Isaac nominal C-in episodes (system1/sim2sim/isaac_dump.py).
Each sample is one independent call (init_rollout before every block, history length 1, as in Isaac's evaluation).
Writes actions [N,29] and tokens [N,64] (output of the actor module's quantizer, captured by a forward hook).

    hssim_s1/bin/python -B system1/sim2sim/sonic_torch_ref.py --out logs_system1/t10/onnx/sonic_torch_ref.npz [--recorded a.npz,b.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1 import sonic_controller as sc  # noqa: E402


def pol64_decoder(pol):
    """g1_dyn decoder of the release checkpoint in float64: action = decoder(token_flattened, proprioception)."""
    import copy
    d = copy.deepcopy(pol.actor_module.decoders["g1_dyn"]).double().eval()

    def run(tok, prop):
        x = torch.from_numpy(np.concatenate([tok, prop]).astype(np.float64))[None]
        return d.module(x).numpy()[0] if hasattr(d, "module") else d(x).numpy()[0]
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--recorded", default="")
    a = ap.parse_args()
    torch.set_num_threads(4)
    spec = json.loads(sc.DEFAULT_SPEC.read_text())
    pol = sc.load_sonic_policy(sc.DEFAULT_CHECKPOINT, spec, "cpu")
    layout = sc.TokenizerLayout.from_spec(spec)
    tok = {}
    q = pol.actor_module.quantizer  # FSQ; called once per encoder (inactive encoders with batch 0): keep the non-empty output

    def hook(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        if t.shape[0] > 0:
            tok["t"] = t.detach().clone()
    q.register_forward_hook(hook)
    rng = np.random.default_rng(0)
    sets = {}
    actor = rng.standard_normal((a.n, 930)).astype(np.float32)
    tz = rng.standard_normal((a.n, layout.total)).astype(np.float32)
    off = 0
    for name, d in zip(layout.names, layout.dims):
        if name == "encoder_index":
            tz[:, off:off + d] = 0.0
            tz[:, off] = 1.0
        off += d
    sets["random"] = (actor, tz)
    for i, p in enumerate([x for x in a.recorded.split(",") if x]):
        z = np.load(p)
        sets[f"recorded{i}"] = (z["sonic_actor_obs"].reshape(-1, 930).astype(np.float32), z["sonic_tokenizer"].reshape(-1, layout.total).astype(np.float32))
    out = dict(layout=np.array(json.dumps(list(zip(layout.names, layout.dims)))), quantizer_module=np.array(str(type(q).__name__) if q is not None else "none"))
    for k, (ao, to) in sets.items():
        acts, toks = [], []
        with torch.no_grad():
            for j in range(len(ao)):
                pol.init_rollout()
                obs = {"actor_obs": torch.from_numpy(ao[j:j + 1]), "tokenizer": torch.from_numpy(to[j:j + 1])}
                acts.append(pol.act_inference(obs_dict=obs, cur_dones=torch.zeros(1), skip_episode_attnmask=True).numpy()[0])
                toks.append(tok["t"].reshape(-1).numpy() if "t" in tok else np.zeros(64))
        out[f"{k}_actor_obs"], out[f"{k}_tokenizer"] = ao, to
        out[f"{k}_action"], out[f"{k}_token"] = np.stack(acts), np.stack(toks)
        # float64 control: decoder in double precision on the float32 tokens (separates rounding from conversion errors)
        dec = pol64_decoder(pol)
        with torch.no_grad():
            out[f"{k}_action64"] = np.stack([dec(np.stack(toks)[j], ao[j]) for j in range(len(ao))])
        print(k, len(ao), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
