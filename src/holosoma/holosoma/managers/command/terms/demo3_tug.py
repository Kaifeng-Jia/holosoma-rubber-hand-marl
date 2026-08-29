"""Non-looping command state for the isolated Demo 3 tug-of-war task."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.command.terms.marl import PairedA1MotionCommand


class Demo3TugMotionCommand(PairedA1MotionCommand):
    """Track two Pull priors while leaving the shared table fully physical.

    The paired reference is written only during an environment reset.  Once an
    episode starts, the phase is clamped at its final frame instead of looping,
    so neither robot nor table can be teleported by the command term.
    """

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        if self.paired_reference_file is None:
            raise ValueError("Demo 3 tug requires an explicit full paired reference")

    def setup(self) -> None:
        super().setup()
        initial = self.reference.sample(
            torch.zeros(1, dtype=torch.long, device=self.device)
        )
        root_xy = initial["agent_body_pos_w"][0, :, 0, :2]
        object_xy = initial["object_pos_w"][0, :2]
        outward = root_xy - object_xy
        norm = torch.linalg.vector_norm(outward, dim=-1, keepdim=True)
        if torch.any(norm <= 1.0e-6):
            raise ValueError("Demo 3 robot root must not coincide with the table center")
        outward = outward / norm
        if not torch.allclose(outward[0], -outward[1], atol=1.0e-4, rtol=0.0):
            raise ValueError(
                "Demo 3 competitors must have opposite fixed Pull directions; "
                f"got {outward.detach().cpu().tolist()}"
            )
        self._pull_axis_template_w = outward
        self.agent_pull_axis_w = outward[None].expand(self.num_envs, -1, -1).clone()
        self.reset_object_pos_w = torch.zeros(
            self.num_envs,
            3,
            dtype=initial["object_pos_w"].dtype,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor | None) -> None:
        env_ids = self._ensure_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        super().reset(env_ids)
        # Demo 3 always starts at the reviewed contact geometry.  The object
        # reset position comes from the written reference, not stale sim data.
        sample = self.reference.sample(self.time_steps[env_ids])
        origins = self.env.simulator.scene.env_origins[env_ids]
        self.reset_object_pos_w[env_ids] = sample["object_pos_w"] + origins
        self.agent_pull_axis_w[env_ids] = self._pull_axis_template_w

    def step(self) -> None:
        self.time_steps.add_(1)
        self.time_steps.clamp_(max=self.reference.num_frames - 1)

    @property
    def object_displacement_w(self) -> torch.Tensor:
        return self.simulator_object_pos_w - self.reset_object_pos_w

    @property
    def signed_object_progress(self) -> torch.Tensor:
        """Displacement toward each competitor, shaped ``[E, 2]``."""
        return torch.sum(
            self.object_displacement_w[:, None, :2] * self.agent_pull_axis_w,
            dim=-1,
        )

    def update_metrics(self) -> None:
        agent_ref_error = torch.linalg.vector_norm(
            self.agent_ref_pos_w
            - self.env.simulator.agent_rigid_body_pos[:, :, self.ref_body_index],
            dim=-1,
        )
        progress = self.signed_object_progress
        self.metrics["motion/error_ref_pos_mean"] = agent_ref_error.mean(dim=1)
        self.metrics["motion/error_ref_pos_max"] = agent_ref_error.max(dim=1).values
        self.metrics["tug/progress_agent_a_m"] = progress[:, 0]
        self.metrics["tug/progress_agent_b_m"] = progress[:, 1]
        self.metrics["tug/table_displacement_m"] = torch.linalg.vector_norm(
            self.object_displacement_w[:, :2], dim=-1
        )


__all__ = ["Demo3TugMotionCommand"]
