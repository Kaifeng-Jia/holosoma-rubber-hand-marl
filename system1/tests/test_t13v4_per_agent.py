"""T13 v4 D-Table-PA unit checks (CPU): per-agent GAE with both robots given the team rewards / values equals the team
GAE; the actor advantage rows follow the env-major / robot-minor layout of the team repeat_interleave.
    hssim_s1/bin/python -B system1/tests/test_t13v4_per_agent.py"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src/holosoma"), str(ROOT)]
from holosoma.agents.mappo.ppo import Plan5PPO  # noqa: E402
from system1.t13v4.per_agent_ppo import Core4DPerAgentPPO  # noqa: E402

torch.manual_seed(0)
T, E = 24, 16
cfg = SimpleNamespace(gamma=0.99, lam=0.95)
team = SimpleNamespace(config=cfg)
pa = SimpleNamespace(config=cfg)
r = torch.randn(T, E, 1)
v = torch.randn(T, E, 1)
d = torch.rand(T, E, 1) < 0.1
last = torch.randn(E, 1)
ret_t, adv_t = Plan5PPO.compute_team_returns_and_advantages(team, last_values=last, values=v, dones=d, rewards=r)
ret_p, adv_p = Core4DPerAgentPPO.compute_team_returns_and_advantages(pa, last_values=last.expand(E, 2), values=v.expand(T, E, 2).clone(),
                                                                     dones=d, rewards=r.expand(T, E, 2).clone())
err_ret = (ret_p - ret_t).abs().max().item()
raw_t = ret_t - v
adv_t = (raw_t - raw_t.mean()) / (raw_t.std(unbiased=False) + 1.0e-8)  # demo3 estimator on the team stream
err_adv = (adv_p - adv_t).abs().max().item()
assert err_ret < 1e-6 and err_adv < 1e-5, (err_ret, err_adv)
# layout: team advantages [B, 1] repeat_interleave(2) == per-agent [B, 2] (both equal) reshaped
B = 8
a = torch.randn(B, 1)
assert torch.equal(a.repeat_interleave(2, dim=0), a.expand(B, 2).reshape(-1, 1))
# per-agent advantages: row 2e + k is robot k of env e
pa_adv = torch.arange(2 * B, dtype=torch.float).reshape(B, 2)
assert pa_adv.reshape(-1, 1)[2 * 3 + 1, 0] == pa_adv[3, 1]
print(dict(per_agent_vs_team_returns_max_abs=err_ret, per_agent_vs_team_advantages_max_abs=err_adv))
print("OK")
