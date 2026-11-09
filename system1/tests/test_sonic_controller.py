"""Smoke tests for system1.sonic_controller (CPU, no checkpoint, no simulator).

The end-to-end check against SONIC's own environment is system1/sonic/check_obs_equivalence.py.
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system1 import sonic_controller as sc  # noqa: E402


def test_quaternion_math():
    a = torch.tensor([[math.cos(0.3), 0, 0, math.sin(0.3)]])  # yaw 0.6 rad, wxyz
    v = torch.tensor([[1.0, 0, 0]])
    r = sc.quat_apply(a, v)
    assert torch.allclose(r, torch.tensor([[math.cos(0.6), math.sin(0.6), 0]]), atol=1e-6)
    m = sc.matrix_from_quat(a)[0]
    assert torch.allclose(m @ v[0], r[0], atol=1e-6)
    ident = sc.quat_mul(sc.quat_conj(a), a)
    assert torch.allclose(ident, torch.tensor([[1.0, 0, 0, 0]]), atol=1e-6)
    assert torch.equal(sc.xyzw_to_wxyz(torch.tensor([[1.0, 2, 3, 4]])), torch.tensor([[4.0, 1, 2, 3]]))


def test_future_indices_clamp():
    idx = sc.future_indices(torch.tensor([0, 95]), 100, 10, 5)
    assert idx[0].tolist() == list(range(0, 50, 5))
    assert idx[1].tolist() == [95] + [99] * 9


def test_layout_and_terms():
    spec = json.loads(sc.DEFAULT_SPEC.read_text())
    lay = sc.TokenizerLayout.from_spec(spec)
    assert lay.total == spec["env_config"]["obs"]["obs_dims"]["tokenizer"] == 1761
    assert sum(d for _, d in sc.PROPRIO_TERMS) * sc.HISTORY == spec["env_config"]["obs"]["obs_dims"]["actor_obs"] == 930
    cmd, ori = sc.g1_encoder_terms(torch.zeros(2, 10, 29), torch.ones(2, 10, 29),
                                   torch.tensor([1.0, 0, 0, 0]).expand(2, 10, 4), torch.tensor([1.0, 0, 0, 0]).expand(2, 4))
    assert cmd.shape == (2, 10, 58) and ori.shape == (2, 10, 6)
    assert torch.equal(cmd[:, :5], torch.zeros(2, 5, 58)) and torch.equal(cmd[:, 5:], torch.ones(2, 5, 58))
    assert torch.allclose(ori[0, 0], torch.tensor([1.0, 0, 0, 1, 0, 0]))


if __name__ == "__main__":
    test_quaternion_math(); test_future_indices_clamp(); test_layout_and_terms(); print("ok")
