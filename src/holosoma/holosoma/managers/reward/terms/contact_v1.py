"""System1 T13 v4 contact items (opt-in reward / termination terms; nothing else changes without them).

Per-frame labels come from a label file (system1/t13v4/contact_labels.py), aligned to the runtime reference frames
(command.time_steps):
  should_contact [F, 2, 2] bool   per robot, per hand (L, R): the person's hand within 1.5 cm of the object (T12 / T14 rule);
  contact_point  [F, 2, 2, 3]     the reference robot palm point projected to the object surface, object frame;
  human_far      [F, 2, L] bool   per robot, per non-hand link: the corresponding human body part farther than 0.1 m
                                  from the object surface; link_names [L];
  palm_offset    [2, 3]           palm point in the rubber-hand link frame (L, R);
  palm_offset_wrist [2, 3]        the same point in the wrist_yaw_link frame (used when the simulator does not track the
                                  rubber hands as separate bodies: they are fixed children of wrist_yaw_link).
Forces: the filtered object-robot contact sensor (simulator flag enable_object_link_contact), one row per robot body.

1. EefContactExpV1 (reward, HDMI eef_contact_exp): per hand r = gain * exp(-d / sigma) * exp(min(F - f0, 0) / fs) when the
   hand should be in contact, else 1; d = palm point to contact point, F = hand-object force. Mean over the two hands,
   per robot; times exp(-k * count) (InterMimic-style undesired-contact penalty: count = non-hand links with force > 1 N
   whose human part is far). Team value = mean over the two robots ([E]); per_agent=True returns [E, 2].
2. LostContactV1 (termination, HDMI cum_lost_contact_steps): a hand that should be in contact is lost when d > 0.2 m or
   F < 1 N; any hand of any robot lost for 25 consecutive steps ends the episode.
All operations are batched on the simulation device; there is no per-environment Python loop and no GPU->CPU copy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from holosoma.managers.reward.base import RewardTermBase
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import quat_apply

# HDMI hdmi-base defaults (eef_contact_exp, cum_lost_contact_steps) and the InterMimic penalty, T13 v4 section 2
CONTACT_V1 = dict(sigma_m=0.3, force_ref_n=10.0, force_scale_n=40.0, gain=5.0,
                  lost_distance_m=0.2, lost_force_n=1.0, lost_steps=25,
                  undesired_force_n=1.0, undesired_k=5.0)
HANDS = ("left_rubber_hand_link", "right_rubber_hand_link")
WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContactV1Labels:
    """Labels on the device plus the sensor / body bindings of one environment (shared by the reward and termination)."""

    def __init__(self, env: Any, labels_file: str):
        path = Path(labels_file).expanduser().resolve()
        z = np.load(path, allow_pickle=False)
        dev = env.device
        self.file, self.sha256 = str(path), _sha256(path)
        self.should = torch.as_tensor(z["should_contact"], device=dev, dtype=torch.bool)
        self.point = torch.as_tensor(z["contact_point"], device=dev, dtype=torch.float)
        self.far = torch.as_tensor(z["human_far"], device=dev, dtype=torch.bool)
        self.link_names = [str(x) for x in z["link_names"]]
        self.palm_offset = torch.as_tensor(z["palm_offset"], device=dev, dtype=torch.float)
        F = self.should.shape[0]
        if self.should.shape != (F, 2, 2) or self.point.shape != (F, 2, 2, 3) or self.far.shape != (F, 2, len(self.link_names)):
            raise ValueError("contact label arrays have inconsistent shapes")
        self.frames = F
        names = list(env.simulator._body_list)
        if all(h in names for h in HANDS):
            self.hand_body = torch.tensor([names.index(h) for h in HANDS], device=dev)
            self.palm_frame = "rubber_hand_link"
        else:  # the tracked body list stops at the wrists (rubber hands are fixed children): palm point in wrist_yaw_link
            self.hand_body = torch.tensor([names.index(w) for w in WRISTS], device=dev)
            self.palm_offset = torch.as_tensor(z["palm_offset_wrist"], device=dev, dtype=torch.float)
            self.palm_frame = "wrist_yaw_link"
        sensor = getattr(env.simulator, "object_robot_contact_sensor", None)
        if sensor is None:
            raise ValueError("contact v1 requires simulator.config.enable_object_link_contact (object_robot_contact_sensor)")
        filters = list(sensor.cfg.filter_prim_paths_expr)
        idx = {}
        for i, f in enumerate(filters):
            robot, body = f.rsplit("/", 2)[-2:]
            idx[(1 if robot.endswith("_1") else 0, body)] = i
        self.hand_filter = torch.tensor([[idx[(r, h)] for h in HANDS] for r in (0, 1)], device=dev)
        missing = [n for n in self.link_names if (0, n) not in idx or (1, n) not in idx]
        if missing:
            raise ValueError(f"label links without a sensor filter: {missing}")
        self.link_filter = torch.tensor([[idx[(r, n)] for n in self.link_names] for r in (0, 1)], device=dev)
        self.sensor = sensor
        self.num_filters = len(filters)

    @staticmethod
    def of(env: Any, labels_file: str) -> "ContactV1Labels":
        cache = getattr(env, "_system1_contact_v1", None)
        if cache is None or cache.file != str(Path(labels_file).expanduser().resolve()):
            cache = ContactV1Labels(env, labels_file)
            env._system1_contact_v1 = cache
        return cache

    def measure(self, env: Any) -> dict[str, torch.Tensor]:
        command = env.command_manager.get_state("paired_motion_command")
        phase = command.time_steps.clamp(0, self.frames - 1)
        should, point, far = self.should[phase], self.point[phase], self.far[phase]
        obj_pos, obj_quat = command.simulator_object_pos_w, command.simulator_object_quat_w  # [E, 3], xyzw
        E = obj_pos.shape[0]
        cp_w = obj_pos[:, None, None] + quat_apply(obj_quat[:, None, None].expand(E, 2, 2, 4).reshape(-1, 4),
                                                   point.reshape(-1, 3), w_last=True).reshape(E, 2, 2, 3)
        hand_pos = env.simulator.agent_rigid_body_pos[:, :, self.hand_body]   # [E, 2, 2, 3]
        hand_rot = env.simulator.agent_rigid_body_rot[:, :, self.hand_body]   # xyzw
        palm = hand_pos + quat_apply(hand_rot.reshape(-1, 4), self.palm_offset[None, None].expand(E, 2, 2, 3).reshape(-1, 3),
                                     w_last=True).reshape(E, 2, 2, 3)
        d = torch.linalg.vector_norm(palm - cp_w, dim=-1)
        forces = self.sensor.data.force_matrix_w
        if forces is None or forces.shape != (E, 1, self.num_filters, 3):
            raise ValueError("object_robot_contact_sensor force matrix must be [N, 1, filters, 3]")
        norms = torch.linalg.vector_norm(forces[:, 0], dim=-1)               # [E, filters]
        f_hand = norms[:, self.hand_filter]                                    # [E, 2, 2]
        f_link = norms[:, self.link_filter]                                    # [E, 2, L]
        return dict(should=should, far=far, d=d, f_hand=f_hand, f_link=f_link, palm=palm, contact_point_w=cp_w)


class EefContactExpV1(RewardTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        E = env.num_envs
        self.last_hand = torch.zeros(E, 2, 2, device=env.device)
        self.last_count = torch.zeros(E, 2, device=env.device)
        self.last_robot = torch.zeros(E, 2, device=env.device)

    def reset(self, env_ids=None) -> None:
        return

    def __call__(self, env: Any, labels_file: str, per_agent: bool = False, **kwargs) -> torch.Tensor:
        c = CONTACT_V1
        m = ContactV1Labels.of(env, labels_file).measure(env)
        dist_term = torch.exp(-m["d"] / c["sigma_m"])
        force_term = torch.exp(torch.clamp(m["f_hand"] - c["force_ref_n"], max=0.0) / c["force_scale_n"])
        hand = torch.where(m["should"], c["gain"] * dist_term * force_term, torch.ones_like(dist_term))
        count = ((m["f_link"] > c["undesired_force_n"]) & m["far"]).sum(dim=-1).float()
        robot = hand.mean(dim=-1) * torch.exp(-c["undesired_k"] * count)
        self.last_hand.copy_(hand)
        self.last_count.copy_(count)
        self.last_robot.copy_(robot)
        return robot if per_agent else robot.mean(dim=-1)


class LostContactV1(TerminationTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, 2, 2, device=env.device, dtype=torch.long)
        self.last_lost = torch.zeros(env.num_envs, 2, 2, device=env.device, dtype=torch.bool)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.counter.zero_()
        else:
            self.counter[env_ids] = 0

    def __call__(self, env: Any, labels_file: str, **kwargs) -> torch.Tensor:
        c = CONTACT_V1
        m = ContactV1Labels.of(env, labels_file).measure(env)
        started = env.episode_length_buf.gt(0)[:, None, None]   # the force of a freshly reset env is not yet valid
        lost = m["should"] & ((m["d"] > c["lost_distance_m"]) | (m["f_hand"] < c["lost_force_n"])) & started
        self.counter = torch.where(lost, self.counter + 1, torch.zeros_like(self.counter))
        self.last_lost.copy_(lost)
        return (self.counter >= c["lost_steps"]).flatten(1).any(dim=1)


__all__ = ["CONTACT_V1", "ContactV1Labels", "EefContactExpV1", "LostContactV1"]
