#!/usr/bin/env python3
"""Offline smoke test of the T10b recipe-v2 randomisation config (no simulator): term names and order, every function path
resolves, and the ranges equal the table of docs/system1/T10b_dr_ranges_zh.md; off (no --system1-randomization) the base config
is not modified. Run with the training env (holosoma importable):

    hssim_s1/bin/python -B system1/t10b/test_randomization_config.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/holosoma"))

from holosoma.config_values.marl.g1.randomization import g1_29dof_plan5_fixed_object_material as BASE  # noqa: E402

from system1.t10b.randomization import randomization_config  # noqa: E402

TASK_BOOK = dict(robot_static_friction=(0.3, 1.6), robot_dynamic_friction=(0.3, 1.2), robot_restitution=(0.0, 0.5),
                 torso_com_offset_m=dict(x=(-0.025, 0.025), y=(-0.05, 0.05), z=(-0.075, 0.075)), dof_pos_bias_rad=(-0.01, 0.01),
                 push_interval_s=(1.0, 3.0), push_max_vel=(0.3, 0.3, 0.3, 0.78, 0.78, 0.78), kp_scale=(0.9, 1.1), kd_scale=(0.9, 1.1),
                 object_static_friction=(0.1, 0.6), object_dynamic_friction=(0.1, 0.6), object_restitution=(0.0, 0.0),
                 object_mass_scale=(0.8, 1.2), object_inertia_scale=(0.5, 1.5), object_com_offset_m=(-0.08, 0.08))


def main():
    before = repr(BASE)
    cfg, r = randomization_config(BASE)
    assert repr(BASE) == before, "base config modified"
    for k, v in TASK_BOOK.items():
        assert r[k] == v, (k, r[k], v)
    assert list(cfg.setup_terms)[0] == "set_object_rigid_body_material_startup"  # the fixed holosoma material first
    assert list(cfg.setup_terms)[-1] == "s1_readback_check"
    assert set(cfg.reset_terms) == {"push_randomizer_state", "randomize_push_schedule", "s1_pd_gain_reset"}
    assert set(cfg.step_terms) == {"push_randomizer_state", "s1_apply_pushes"}
    assert not any("delay" in n for stage in (cfg.setup_terms, cfg.reset_terms, cfg.step_terms) for n in stage), "action delay must be off"
    for stage in (cfg.setup_terms, cfg.reset_terms, cfg.step_terms):
        for t in stage.values():
            m, f = t.func.split(":")
            getattr(importlib.import_module(m), f)
    print("ok randomization_config v2:", len(cfg.setup_terms), "setup /", len(cfg.reset_terms), "reset /", len(cfg.step_terms), "step terms")


if __name__ == "__main__":
    main()
