"""CPU smoke test for scheme C v2 (T03): SonicResidualEnv kinds v1 / in / out, SonicDualDriver clamp and output residual,
CTrainDiag window statistics. No simulator: the env, driver and SONIC controller are stand-ins.

    hssim_s1/bin/python -B -m pytest -q -p no:cacheprovider system1/tests/test_scheme_c_v2.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1.scheme_a import SonicDualDriver  # noqa: E402
from system1.scheme_c import C_IN_PPO_OVERRIDES, KINDS, SonicResidualEnv  # noqa: E402

E, A, J = 3, 2, 29


class FakeDriver:
    def __init__(self):
        self.kw = None

    def actions(self, command, dones=None, **kw):
        self.kw = kw
        return torch.zeros(E, A * J), None

    def reset(self, ids=None):
        pass


class FakeEnv:
    def __init__(self):
        self.action_manager = SimpleNamespace(policy_action=torch.zeros(E, A * J))
        self.got = None

    def step(self, actor_state):
        self.got = actor_state
        return None, torch.zeros(E), torch.zeros(E, dtype=torch.bool), {}


def make(kind, out_clip=100.0):
    w = object.__new__(SonicResidualEnv)
    sa = lambda k, v: object.__setattr__(w, k, v)  # noqa: E731
    sa("env", FakeEnv()); sa("driver", FakeDriver()); sa("E", E); sa("A", A)
    sa("scale_d", torch.linspace(0.1, 0.9, J))
    sa("command", SimpleNamespace(time_steps=torch.zeros(E, dtype=torch.long)))
    sa("_prev_t", None); sa("_dones", torch.zeros(E)); sa("sonic_resets", 0); sa("_diag", None); sa("train_diag", None)
    sa("kind", kind)
    sa("ref_limits_h", (torch.full((J,), -1.0), torch.full((J,), 1.0)) if kind == "in" else None)
    sa("out_clip", out_clip if kind == "out" else None)
    return w


def test_kinds_and_overrides():
    assert KINDS == ("v1", "in", "out")
    assert C_IN_PPO_OVERRIDES == dict(init_noise_std=0.3, entropy_coef=0.0)


def test_step_v1_unchanged():
    w = make("v1")
    a = torch.randn(E, A * J) * 3
    w.step({"actions": a})
    assert set(w.driver.kw) == {"delta_q_h"}
    assert torch.equal(w.driver.kw["delta_q_h"], a.reshape(E, A, J) * w.scale_d)
    assert torch.equal(w.env.action_manager.policy_action, a)


def test_step_c_in_clips_and_limits():
    w = make("in")
    a = torch.randn(E, A * J) * 3
    w.step({"actions": a})
    dq = w.driver.kw["delta_q_h"]
    ac = a.reshape(E, A, J).clamp(-1, 1)
    assert torch.equal(dq, ac * w.scale_d)
    assert bool(((dq.abs() / w.scale_d) <= 1 + 1e-6).all())
    assert w.driver.kw["ref_limits_h"] is w.ref_limits_h and "target_residual_h" not in w.driver.kw
    assert torch.equal(w.env.action_manager.policy_action, ac.reshape(E, -1))  # last-action obs = clipped a
    assert torch.equal(w.env.got["actions"], torch.zeros(E, A * J))           # env gets a_eff from the driver


def test_step_c_out_residual_on_output():
    w = make("out", out_clip=2.0)
    a = torch.randn(E, A * J) * 3
    w.step({"actions": a})
    assert set(w.driver.kw) == {"target_residual_h"}
    assert torch.equal(w.driver.kw["target_residual_h"], a.reshape(E, A, J).clamp(-2, 2) * w.scale_d)
    assert torch.equal(w.env.action_manager.policy_action, a)  # own raw a, as D


def _driver_stub():
    d = object.__new__(SonicDualDriver)
    d.E, d.A = E, A
    perm = torch.randperm(J)
    inv = torch.argsort(perm)
    seen = {}

    class Ctl:
        def to_sonic(self, x):
            return x[..., perm]

        def to_holosoma(self, x):
            return x[..., inv]

        def act(self, q, w_b, jq, jqd, jp, jv, rq, dn):
            seen["jp"] = jp.clone()
            return torch.zeros(E * A, J), torch.full((E * A, J), 0.5), None
    d.ctl = Ctl()
    ref = torch.zeros(E * A, 10, J)
    d.reference_window = lambda command: (ref.clone(), ref.clone(), torch.zeros(E * A, 10, 4), None)
    d.robot_state = lambda: (None, None, None, None)
    d.term = SimpleNamespace(joint_target_override=None)
    d.default_h = torch.zeros(J)
    d.scale_h = torch.ones(J)
    return d, seen


def test_driver_clamp_and_residual():
    d, seen = _driver_stub()
    dq = torch.full((E, A, J), 5.0)
    lim = (torch.full((J,), -0.3), torch.linspace(0.1, 0.4, J))
    d.actions(None, delta_q_h=dq, ref_limits_h=lim)
    hi_s = d.ctl.to_sonic(lim[1])
    assert torch.allclose(seen["jp"], hi_s.expand_as(seen["jp"]))            # clamped to the upper limit, SONIC order
    assert torch.allclose(d.term.joint_target_override, torch.full((E, A, J), 0.5))
    d, seen = _driver_stub()
    res = torch.randn(E, A, J)
    d.actions(None, target_residual_h=res)
    assert torch.equal(seen["jp"], torch.zeros(E * A, 10, J))                # input untouched
    assert torch.allclose(d.term.joint_target_override - d.last_sonic_target_h, res)
    d, seen = _driver_stub()
    d.actions(None)                                                           # scheme A / B path unchanged
    assert torch.equal(seen["jp"], torch.zeros(E * A, 10, J)) and torch.equal(d.term.joint_target_override, d.last_sonic_target_h)


def test_train_diag_window():
    from system1.c_train_diag import CTrainDiag
    g = object.__new__(CTrainDiag)
    g.w = SimpleNamespace(kind="in", scale_d=torch.ones(J))
    g.every, g.names = 100, [f"j{i}" for i in range(J)]
    g._steps, g._last_iter_corr = [], None
    g.checks = dict(c_in_max_dq_over_scale=0.0, c_in_max_ref_limit_violation_rad=0.0, c_out_max_qstar_minus_sonic_plus_c=0.0)
    g._reset_window()
    for it in range(2):
        for _ in range(4):
            a = torch.randn(E, A, J) * 2
            au = a.clamp(-1, 1)
            g.step(a, au, au * 0.5)
        g.end_iteration()
    out = g.summary(SimpleNamespace(std=torch.full((J,), 0.3)))
    assert out["window_iterations"] == 2 and 0 < out["correction_abs_mean_rad"] <= out["correction_abs_p95_rad"] <= out["correction_abs_max_rad"] <= 0.5
    assert 0 < out["frac_clipped"] == out["frac_abs_a_gt_1"] < 1
    assert abs(out["action_std_mean"] - 0.3) < 1e-6 and len(out["correction_last_iteration_per_joint"]) == J
    assert out["checks"]["c_in_max_dq_over_scale"] <= 0.5 + 1e-6
    assert g.win["n_iter"] == 0  # window reset after summary


def test_train_diag_termination_split():
    from system1.c_train_diag import CTrainDiag
    g = object.__new__(CTrainDiag)
    g._reset_window()
    g.term = SimpleNamespace(last_diagnostics=dict(bad_object_position=torch.tensor([True, False, True, False]),
                                                   bad_object_orientation=torch.tensor([False, False, True, False]),
                                                   bad_robot=torch.tensor([False, True, False, False])))
    dones = torch.tensor([True, True, True, True])
    g.post(dones, dict(time_outs=torch.tensor([False, False, False, True])))
    w = g.win
    assert (w["done"], w["timeout"], w["terminated"]) == (4, 1, 3)
    assert (w["object_position"], w["object_orientation"], w["robot_torso"]) == (2, 1, 1)
