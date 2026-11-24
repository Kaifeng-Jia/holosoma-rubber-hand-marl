"""System1 T13 v4 observation v1 terms (switch: the *_v1 observation presets; the legacy terms are untouched).

Per robot (actor, 159 = 154 legacy actor block + 5 below):
  * reference torso orientation 6: as paired_motion_ref_ori_b, but the actual torso_link orientation is first rotated about
    world z by this robot's heading bias (yaw only; roll / pitch unchanged);
  * partner relative position 2: partner neck point minus own neck point (world), horizontal part, expressed in the own
    heading frame (torso_link yaw + heading bias), clipped to +-1 m by the group config;
  * object relative position 3: object root origin minus own neck point, in the own heading frame (yaw only, with bias),
    clipped to +-2 m by the group config.
Neck point = a fixed point on torso_link (NECK_POINT_TORSO_M, system1 T13 v4 config). Noise v1 (only when the term's
``noise`` param is True; training on, validation / evaluation off): per-episode offsets and per-step uniform noise on the
neck point and the object point, per robot and independent between robots; per-episode heading bias per robot. All numbers
live in NOISE_V1. Every operation is batched over environments; nothing is moved to the CPU.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from holosoma.managers.observation.base import ObservationTermBase
from holosoma.managers.observation.terms.marl import _paired_motion_command
from holosoma.utils.rotations import quat_mul, quat_rotate_inverse, quaternion_to_matrix, subtract_frame_transforms

# --------------------------------------------------------------------------------------------- the one config place
# torso_link frame, metres: midpoint of the two factory lifting rings on top of the torso (torso_link_rev_1_0.STL
# components centred at y = +0.0722 / -0.0732, x = 0.003, z = 0.2898); see results_system1/T13v4_prep/README_CN.md
NECK_POINT_TORSO_M = (0.003, -0.0005, 0.2898)
NOISE_V1 = dict(
    neck_episode_offset_m=0.03,   # uniform +-, per axis, per episode, per robot
    neck_step_m=0.01,             # uniform +-, per axis, per step, per robot
    object_episode_offset_m=0.02,
    object_step_m=0.01,
    heading_bias_deg=5.0,         # uniform +-, per episode, per robot
)
PARTNER_CLIP_M = 1.0
OBJECT_CLIP_M = 2.0


def _uniform(shape, half, device):
    return (torch.rand(shape, device=device) * 2.0 - 1.0) * half


class ObsV1State:
    """Per-environment noise state shared by the v1 terms of one environment."""

    def __init__(self, env: Any):
        E, A, dev = env.num_envs, env.num_agents, env.device
        self.env = env
        self.neck_offset = torch.zeros(E, A, 3, device=dev)
        self.object_offset = torch.zeros(E, A, 3, device=dev)
        self.heading_bias = torch.zeros(E, A, device=dev)
        self.neck_step = torch.zeros(E, A, 3, device=dev)
        self.object_step = torch.zeros(E, A, 3, device=dev)
        self.noise_enabled: bool | None = None
        self.owner: Any = None
        self.initialized = False
        names = list(env.simulator._body_list)
        self.torso_index = names.index("torso_link")
        self.neck_local = torch.tensor(NECK_POINT_TORSO_M, device=dev, dtype=torch.float)
        # clip statistics (GPU counters; read only when logging)
        self.partner_clipped = torch.zeros((), device=dev, dtype=torch.long)
        self.object_clipped = torch.zeros((), device=dev, dtype=torch.long)
        self.samples = 0  # python int: E * A per partner-term call (a tensor += int would synchronise)

    @staticmethod
    def of(env: Any) -> "ObsV1State":
        state = getattr(env, "_system1_obs_v1", None)
        if state is None:
            state = ObsV1State(env)
            env._system1_obs_v1 = state
        return state

    def configure(self, noise: bool, term: Any) -> None:
        if self.noise_enabled is None:
            self.noise_enabled = bool(noise)
        elif self.noise_enabled != bool(noise):
            raise ValueError("All observation-v1 terms of one environment must share the same noise switch")
        if self.owner is None:
            self.owner = term

    def resample_episode(self, env_ids: torch.Tensor | None) -> None:
        if not self.noise_enabled:
            return
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        n, A, dev = env_ids.numel(), self.env.num_agents, self.env.device
        self.neck_offset[env_ids] = _uniform((n, A, 3), NOISE_V1["neck_episode_offset_m"], dev)
        self.object_offset[env_ids] = _uniform((n, A, 3), NOISE_V1["object_episode_offset_m"], dev)
        self.heading_bias[env_ids] = _uniform((n, A), math.radians(NOISE_V1["heading_bias_deg"]), dev)

    def advance(self, term: Any) -> None:
        """Draw this step's noise once per observation computation (called by the owner term)."""
        if term is not self.owner:
            return
        if not self.initialized:
            self.resample_episode(None)
            self.initialized = True
        if self.noise_enabled:
            E, A, dev = self.env.num_envs, self.env.num_agents, self.env.device
            self.neck_step = _uniform((E, A, 3), NOISE_V1["neck_step_m"], dev)
            self.object_step = _uniform((E, A, 3), NOISE_V1["object_step_m"], dev)

    # ------------------------------------------------------------------------------------------- geometry
    def torso_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        sim = self.env.simulator
        return sim.agent_rigid_body_pos[:, :, self.torso_index], sim.agent_rigid_body_rot[:, :, self.torso_index]

    def biased_heading_quat(self) -> torch.Tensor:
        """xyzw yaw-only quaternion: torso_link yaw + heading bias, [E, A, 4]."""
        _, quat = self.torso_pose()
        x, y, z, w = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) + self.heading_bias
        half = 0.5 * yaw
        zeros = torch.zeros_like(half)
        return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)

    def neck_points_w(self, noisy: bool) -> torch.Tensor:
        pos, quat = self.torso_pose()
        local = self.neck_local.expand(pos.shape[0], pos.shape[1], 3)
        from holosoma.utils.rotations import quat_apply
        neck = pos + quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3), w_last=True).reshape(pos.shape)
        if noisy and self.noise_enabled:
            neck = neck + self.neck_offset + self.neck_step
        return neck

    def object_point_w(self) -> torch.Tensor:
        command = _paired_motion_command(self.env)
        obj = command.simulator_object_pos_w[:, None, :].expand(-1, self.env.num_agents, -1)
        if self.noise_enabled:
            obj = obj + self.object_offset + self.object_step
        return obj


class _V1Term(ObservationTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.state = ObsV1State.of(env)
        self.state.configure(bool(cfg.params.get("noise", False)), self)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.state.owner is self:
            self.state.resample_episode(env_ids)


class MotionRefOriBV1(_V1Term):
    """[E, A, 6]: reference torso orientation relative to the heading-biased actual torso orientation."""

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        self.state.advance(self)
        command = _paired_motion_command(env)
        actual_pos = env.simulator.agent_rigid_body_pos[:, :, command.ref_body_index]
        actual_quat = env.simulator.agent_rigid_body_rot[:, :, command.ref_body_index]
        if self.state.noise_enabled:
            half = 0.5 * self.state.heading_bias
            zeros = torch.zeros_like(half)
            bias_quat = torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)
            actual_quat = quat_mul(bias_quat.reshape(-1, 4), actual_quat.reshape(-1, 4), w_last=True).reshape(actual_quat.shape)
        _, relative_quat = subtract_frame_transforms(
            actual_pos.flatten(0, 1), actual_quat.flatten(0, 1),
            command.agent_ref_pos_w.flatten(0, 1), command.agent_ref_quat_w.flatten(0, 1),
        )
        matrix = quaternion_to_matrix(relative_quat.reshape(env.num_envs, env.num_agents, 4), w_last=True)
        return matrix[..., :2].reshape(env.num_envs, env.num_agents, 6)


class NeckPartnerRelPosBV1(_V1Term):
    """[E, A, 2]: partner neck point minus own neck point, horizontal, in the own biased heading frame."""

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        self.state.advance(self)
        neck = self.state.neck_points_w(noisy=True)
        partner = neck.flip(1)  # the other robot (2 agents); avoids a host index tensor
        rel_w = partner - neck
        heading = self.state.biased_heading_quat()
        rel_b = quat_rotate_inverse(heading.reshape(-1, 4), rel_w.reshape(-1, 3), w_last=True).reshape(rel_w.shape)[..., :2]
        self.state.partner_clipped += (rel_b.abs() > PARTNER_CLIP_M).any(dim=-1).sum()
        self.state.samples += int(rel_b.shape[0] * rel_b.shape[1])
        return rel_b


class ObjectRelPosBV1(_V1Term):
    """[E, A, 3]: object root origin minus own neck point, in the own biased heading frame (yaw only)."""

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        self.state.advance(self)
        neck = self.state.neck_points_w(noisy=True)
        rel_w = self.state.object_point_w() - neck
        heading = self.state.biased_heading_quat()
        rel_b = quat_rotate_inverse(heading.reshape(-1, 4), rel_w.reshape(-1, 3), w_last=True).reshape(rel_w.shape)
        self.state.object_clipped += (rel_b.abs() > OBJECT_CLIP_M).any(dim=-1).sum()
        return rel_b


def centralized_both_motion_commands(env: Any) -> torch.Tensor:
    """Critic fix: both robots' reference joint commands, [E, 2 * 58] (legacy term: robot 0 only)."""
    return _paired_motion_command(env).command.reshape(env.num_envs, -1)


__all__ = [
    "MotionRefOriBV1", "NECK_POINT_TORSO_M", "NOISE_V1", "NeckPartnerRelPosBV1", "OBJECT_CLIP_M", "ObjectRelPosBV1",
    "ObsV1State", "PARTNER_CLIP_M", "centralized_both_motion_commands",
]
