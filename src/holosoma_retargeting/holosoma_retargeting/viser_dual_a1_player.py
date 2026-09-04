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
    rollout_npz: Path | None = None
    robot_urdf: Path = DEFAULT_ROBOT_URDF
    object_urdf: Path = DEFAULT_OBJECT_URDF
    lateral_spacing: float = 0.8
    host: str = "127.0.0.1"
    port: int = 8080
    loop: bool = False
    grid_size: float = 8.0


def _xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Convert Isaac Sim quaternion arrays to the convention used by ViSER."""
    return quaternion[..., [3, 0, 1, 2]]


def make_player(config: DualA1ViserConfig) -> viser.ViserServer:
    """Create a Viser server for inspecting the shared-table A1 layout."""
    if config.rollout_npz is None:
        joint_pos, object_pos, object_quat, motion_fps = load_a1_motion(config.motion_npz)
        frame_count = joint_pos.shape[0]
        rollout_root_pos = None
        rollout_root_quat = None
        rollout_joint_pos = None
        source_label = f"reference={config.motion_npz}"
    else:
        with np.load(config.rollout_npz, allow_pickle=False) as recording:
            required = {
                "root_pos",
                "root_quat_xyzw",
                "dof_pos",
                "object_pos_w",
                "object_quat_xyzw",
            }
            missing = sorted(required.difference(recording.files))
            if missing:
                raise ValueError(f"Rollout recording is missing required channels: {missing}")
            rollout_root_pos = np.asarray(recording["root_pos"])
            rollout_root_quat = _xyzw_to_wxyz(
                np.asarray(recording["root_quat_xyzw"])
            )
            rollout_joint_pos = np.asarray(recording["dof_pos"])
            object_pos = np.asarray(recording["object_pos_w"])
            object_quat = _xyzw_to_wxyz(np.asarray(recording["object_quat_xyzw"]))

        frame_count = rollout_joint_pos.shape[0]
        expected_shapes = {
            "root_pos": (frame_count, 2, 3),
            "root_quat_xyzw": (frame_count, 2, 4),
            "dof_pos": (frame_count, 2, 29),
            "object_pos_w": (frame_count, 3),
            "object_quat_xyzw": (frame_count, 4),
        }
        actual_shapes = {
            "root_pos": rollout_root_pos.shape,
            "root_quat_xyzw": rollout_root_quat.shape,
            "dof_pos": rollout_joint_pos.shape,
            "object_pos_w": object_pos.shape,
            "object_quat_xyzw": object_quat.shape,
        }
        mismatched = {
            name: {"actual": actual_shapes[name], "expected": expected}
            for name, expected in expected_shapes.items()
            if actual_shapes[name] != expected
        }
        if mismatched:
            raise ValueError(f"Unexpected rollout channel shapes: {mismatched}")
        if not all(
            np.isfinite(values).all()
            for values in (
                rollout_root_pos,
                rollout_root_quat,
                rollout_joint_pos,
                object_pos,
                object_quat,
            )
        ):
            raise ValueError("Rollout recording contains non-finite pose values")
        motion_fps = 50.0
        joint_pos = None
        source_label = f"rollout={config.rollout_npz}"

    server = viser.ViserServer(host=config.host, port=config.port)
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
        spacing_input = None
        if config.rollout_npz is None:
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
            max=frame_count - 1,
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
        index = int(np.clip(frame, 0, frame_count - 1))
        if rollout_joint_pos is None:
            assert joint_pos is not None and spacing_input is not None
            qpos = joint_pos[index]
            positions = shifted_robot_positions(
                qpos[:3],
                object_quat[index],
                float(spacing_input.value),
            )
            orientations = (qpos[3:7], qpos[3:7])
            joint_configs = (qpos[7:], qpos[7:])
        else:
            assert rollout_root_pos is not None and rollout_root_quat is not None
            positions = rollout_root_pos[index]
            orientations = rollout_root_quat[index]
            joint_configs = rollout_joint_pos[index]
        with server.atomic():
            for agent, frame_handle, position, orientation, joint_config in zip(
                agents,
                agent_frames,
                positions,
                orientations,
                joint_configs,
                strict=True,
            ):
                agent.update_cfg(joint_config)
                frame_handle.position = position
                frame_handle.wxyz = orientation
            object_frame.position = object_pos[index]
            object_frame.wxyz = object_quat[index]

    @frame_slider.on_update
    def _(_) -> None:
        playback["frame"] = int(frame_slider.value)
        draw_frame(playback["frame"])

    if spacing_input is not None:
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
            if next_frame >= frame_count:
                if config.loop:
                    next_frame = 0
                else:
                    next_frame = frame_count - 1
                    playback["playing"] = False
            playback["frame"] = next_frame
            frame_slider.value = next_frame
            draw_frame(next_frame)
            next_tick = now + 1.0 / max(float(fps_input.value), 1.0)

    draw_frame(0)
    threading.Thread(target=playback_loop, daemon=True).start()
    print(
        f"[viser_dual_a1_player] Loaded {frame_count} frames at {motion_fps} FPS | "
        f"{source_label} | rubber-hand agents=2 | shared table=1"
    )
    return server


def main(config: DualA1ViserConfig) -> None:
    """Run the Viser server until interrupted."""
    make_player(config)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(DualA1ViserConfig))
