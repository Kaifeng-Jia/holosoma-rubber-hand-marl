#!/usr/bin/env python3
"""T13 round 2, C-out: read-only diagnosis of a failed q* step check (check_sim2sim_v1.py --scheme C-out, T10c threshold 1e-4).
For every control step of the run's Isaac nominal export: SONIC ONNX on the MuJoCo-built SONIC inputs (as the check) and on the
Isaac-recorded SONIC inputs (sonic_tokenizer, sonic_actor_obs of the export); per step the largest action / q* difference to the
Isaac record; the encoder token (64 values, quantised) differences between the two inputs; at the worst step, the action change
under N(0, 1e-6) noise on the tokenizer input (200 draws). Nothing is changed; writes one json.

    sim2sim_s1/bin/python -B system1/t13v4/cout_qstar_diag.py --run c_out_push --task push --out results_system1/T13_runs/c_out_push/checks_qstar_diag.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.sim2sim.mj_env import PairEnv  # noqa: E402
from system1.sim2sim.sonic_mj import COutController, SonicOnnx  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base = ROOT / "logs_system1/t13v4/isaac_dump" / a.run
    env = PairEnv(base / "dump", a.task)
    env.obs_v1 = True
    s = SonicOnnx(env.dof_names)
    cout = COutController.__new__(COutController)
    cout.env, cout.sonic, cout.scale_d = env, s, np.array(env.ss.control["action_scales"])
    z = np.load(base / "dump/nominal.sim2sim.npz")
    tr = np.load(base / "eval/nominal.trajectory.npz")
    K = z["actor_obs"].shape[0]

    def enc(tn):
        x = np.concatenate([np.zeros((1, 4), np.float32), tn[None, 3:]], axis=1).astype(np.float32)
        return s.enc.run(None, {"obs_dict": x})[0]

    def dec(t, an):
        return s.dec.run(None, {"obs_dict": np.concatenate([t, an[None]], axis=1).astype(np.float32)})[0][0]

    def qstar(act, k):
        target = (s.default + s.scale * torch.as_tensor(act).clamp(-s.clip, s.clip)).numpy()
        th = target[:, s.s2h].astype(np.float64)
        return th + cout.residual(z["actions"][k, 0].reshape(2, 29).astype(np.float64))

    per = dict(mj_action=[], mj_qstar=[], isaac_in_action=[], isaac_in_qstar=[])
    flips, mj_in = [], []
    for k in range(K):
        env.set_state(z["root_states"][k, 0].astype(np.float64), z["dof_pos"][k, 0], z["dof_vel"][k, 0],
                      tr["object_pos_w"][k].reshape(3).astype(np.float64), tr["object_quat_xyzw"][k].reshape(4).astype(np.float64))
        st = env.state()
        env.t = k
        if k == 0:
            s.reset()
        else:
            s.last_action = torch.as_tensor(np.clip(z["sonic_action"][k - 1], -s.clip, s.clip), dtype=torch.float32)
        act_mj, _, sobs = s.act(*cout.sonic_inputs_out(st, k))
        mj_in.append(sobs["sonic_tokenizer"].copy())
        pd = z["pd_target"][k, 0].reshape(2, 29).astype(np.float64)
        act_is = np.stack([dec(enc(z["sonic_tokenizer"][k, r].astype(np.float32)), z["sonic_actor_obs"][k, r].astype(np.float32)) for r in range(2)])
        for r in range(2):
            t1, t2 = enc(z["sonic_tokenizer"][k, r].astype(np.float32)), enc(sobs["sonic_tokenizer"][r].astype(np.float32))
            if not np.array_equal(t1, t2):
                flips.append(dict(step=k, robot=r, components=int((t1 != t2).sum()), max_token_diff=float(np.abs(t1 - t2).max())))
        per["mj_action"].append(float(np.abs(act_mj - z["sonic_action"][k]).max()))
        per["mj_qstar"].append(float(np.abs(qstar(act_mj, k) - pd).max()))
        per["isaac_in_action"].append(float(np.abs(act_is - z["sonic_action"][k]).max()))
        per["isaac_in_qstar"].append(float(np.abs(qstar(act_is, k) - pd).max()))
    worst = int(np.argmax(per["mj_qstar"]))
    rw = int(np.argmax(np.abs(np.asarray(z["pd_target"][worst, 0]).reshape(2, 29) - qstar(np.stack([dec(enc(mj_in[worst][r].astype(np.float32)), z["sonic_actor_obs"][worst, r].astype(np.float32)) for r in range(2)]), worst)).max(1)))
    tn = z["sonic_tokenizer"][worst, rw].astype(np.float32)
    an = z["sonic_actor_obs"][worst, rw].astype(np.float32)
    a0 = dec(enc(tn), an)
    rng = np.random.default_rng(0)
    jumps = np.array([float(np.abs(dec(enc(tn + rng.normal(0, 1e-6, tn.shape).astype(np.float32)), an) - a0).max()) for _ in range(200)])
    out = dict(run=a.run, steps=K, threshold=1e-4,
               mujoco_built_inputs=dict(qstar_max=max(per["mj_qstar"]), action_max=max(per["mj_action"]), steps_qstar_over_threshold=[i for i, v in enumerate(per["mj_qstar"]) if v >= 1e-4],
                                        tokenizer_input_max_abs_diff=float(np.abs(np.asarray(mj_in) - z["sonic_tokenizer"]).max())),
               isaac_recorded_inputs=dict(qstar_max=max(per["isaac_in_qstar"]), action_max=max(per["isaac_in_action"]), steps_qstar_over_threshold=[i for i, v in enumerate(per["isaac_in_qstar"]) if v >= 1e-4]),
               encoder_token_differences=flips, token_values_unique_at_worst=int(len(np.unique(np.round(enc(tn), 6)))),
               worst_step=dict(step=worst, robot=rw, noise_1e6_action_change_max=float(jumps.max()), noise_1e6_share_over_1e3=float((jumps > 1e-3).mean())),
               command=" ".join(sys.argv), rules=__doc__)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rules",)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
