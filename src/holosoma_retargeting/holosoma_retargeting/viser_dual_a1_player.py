#!/usr/bin/env python3
"""Preview two laterally shifted copies of the frozen A1 reference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting.dual_a1_layout import load_a1_motion, shifted_robot_positions


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_MOTION_NPZ = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "rubber_hand_largetable_v1"
    / "a1"
    / "sub6_largetable_033_a1_mj_fps50_w_obj.npz"
)
DEFAULT_ROBOT_URDF = PACKAGE_DIR / "models" / "g1" / "g1_29dof.urdf"
DEFAULT_OBJECT_URDF = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "objects_widetable.urdf"
)


@dataclass(frozen=True)
class DualA1ViserConfig:
    """Configuration for the read-only two-agent A1 geometry preview."""

    motion_npz: Path = DEFAULT_MOTION_NPZ
    robot_urdf: Path = DEFAULT_ROBOT_URDF
    object_urdf: Path = DEFAULT_OBJECT_URDF
    lateral_spacing: float = 0.8
    port: int = 8080
    loop: bool = False
    grid_size: float = 8.0


def make_player(config: DualA1ViserConfig) -> viser.ViserServer:
    """Create a Viser server for inspecting the shared-table A1 layout."""
    joint_pos, object_pos, object_quat, motion_fps = load_a1_motion(config.motion_npz)

    server = viser.ViserServer(port=config.port)
    agent_frames = (
        server.scene.add_frame("/agent_0", show_axes=False),
        server.scene.add_frame("/agent_1", show_axes=False),
    )
    object_frame = server.scene.add_frame("/object", show_axes=False)
    agents = (
        ViserUrdf(server, config.robot_urdf, root_node_name="/agent_0"),
        ViserUrdf(server, config.robot_urdf, root_node_name="/agent_1"),
    )
    table = ViserUrdf(server, config.object_urdf, root_node_name="/object")
    server.scene.add_grid(
        "/grid",
        width=config.grid_size,
        height=config.grid_size,
        position=(0.0, 0.0, 0.0),
    )

    robot_dof = len(agents[0].get_actuated_joint_limits())
    if robot_dof != 29:
        raise ValueError(f"Expected a 29-DoF rubber-hand robot URDF, got {robot_dof} DoF")
    if len(agents[1].get_actuated_joint_limits()) != robot_dof:
        raise ValueError("The two robot instances expose different joint layouts")

    with server.gui.add_folder("Layout"):
        spacing_input = server.gui.add_number(
            "Agent center spacing (m)",
            initial_value=config.lateral_spacing,
            min=0.4,
            max=1.2,
            step=0.02,
        )
        show_meshes = server.gui.add_checkbox("Show meshes", initial_value=True)
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=joint_pos.shape[0] - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play / Pause")
        fps_input = server.gui.add_number(
            "FPS",
            initial_value=motion_fps,
            min=1,
            max=240,
            step=1,
        )

    playback = {"playing": False, "frame": 0}

    def draw_frame(frame: int) -> None:
        index = int(np.clip(frame, 0, joint_pos.shape[0] - 1))
        qpos = joint_pos[index]
        positions = shifted_robot_positions(
            qpos[:3],
            object_quat[index],
            float(spacing_input.value),
        )
        with server.atomic():
            for agent, frame_handle, position in zip(
                agents,
                agent_frames,
                positions,
                strict=True,
            ):
                agent.update_cfg(qpos[7:])
                frame_handle.position = position
                frame_handle.wxyz = qpos[3:7]
            object_frame.position = object_pos[index]
            object_frame.wxyz = object_quat[index]

    @frame_slider.on_update
    def _(_) -> None:
        playback["frame"] = int(frame_slider.value)
        draw_frame(playback["frame"])

    @spacing_input.on_update
    def _(_) -> None:
        draw_frame(playback["frame"])

    @show_meshes.on_update
    def _(_) -> None:
        visible = bool(show_meshes.value)
        for agent in agents:
            agent.show_visual = visible
        table.show_visual = visible

    @play_button.on_click
    def _(_) -> None:
        playback["playing"] = not playback["playing"]

    def playback_loop() -> None:
        next_tick = time.perf_counter()
        while True:
            if not playback["playing"]:
                next_tick = time.perf_counter()
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.01))
                continue

            next_frame = playback["frame"] + 1
            if next_frame >= joint_pos.shape[0]:
                if config.loop:
                    next_frame = 0
                else:
                    next_frame = joint_pos.shape[0] - 1
                    playback["playing"] = False
            playback["frame"] = next_frame
            frame_slider.value = next_frame
            draw_frame(next_frame)
            next_tick = now + 1.0 / max(float(fps_input.value), 1.0)

    draw_frame(0)
    threading.Thread(target=playback_loop, daemon=True).start()
    print(
        f"[viser_dual_a1_player] Loaded {joint_pos.shape[0]} frames at {motion_fps} FPS | "
        f"spacing={config.lateral_spacing:.3f} m | rubber-hand agents=2 | shared table=1"
    )
    return server


def main(config: DualA1ViserConfig) -> None:
    """Run the Viser server until interrupted."""
    make_player(config)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(DualA1ViserConfig))
