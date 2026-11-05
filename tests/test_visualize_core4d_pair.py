from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize_core4d_pair.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("visualize_core4d_pair", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pair(path: Path, *, robot_shape: tuple[int, ...] = (3, 2, 36)) -> None:
    robot_qpos = np.zeros(robot_shape, dtype=np.float64)
    if robot_qpos.ndim == 3 and robot_qpos.shape[-1] >= 7:
        robot_qpos[..., 3] = 1.0
    object_qpos = np.zeros((robot_shape[0], 7), dtype=np.float64)
    object_qpos[:, 3] = 1.0
    np.savez_compressed(
        path,
        robot_qpos=robot_qpos,
        object_qpos=object_qpos,
        fps=np.asarray(30, dtype=np.int64),
        object_name=np.asarray("Box026"),
    )


def test_load_pair_reference_accepts_expected_pickle_free_schema(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "pair.npz"
    _write_pair(path)

    reference = module.load_pair_reference(path)

    assert reference.robot_qpos.shape == (3, 2, 36)
    assert reference.object_qpos.shape == (3, 7)
    assert reference.fps == 30
    assert reference.object_name == "Box026"


def test_load_pair_reference_rejects_wrong_robot_shape(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "pair.npz"
    _write_pair(path, robot_shape=(3, 1, 36))

    with pytest.raises(ValueError, match="robot_qpos must have shape"):
        module.load_pair_reference(path)


def test_load_pair_reference_requires_object_name(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "pair.npz"
    robot_qpos = np.zeros((1, 2, 36), dtype=np.float64)
    robot_qpos[..., 3] = 1.0
    object_qpos = np.zeros((1, 7), dtype=np.float64)
    object_qpos[..., 3] = 1.0
    np.savez_compressed(path, robot_qpos=robot_qpos, object_qpos=object_qpos, fps=30)

    with pytest.raises(ValueError, match="object_name"):
        module.load_pair_reference(path)


def test_load_pair_reference_rejects_non_unit_quaternion(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "pair.npz"
    _write_pair(path)
    with np.load(path, allow_pickle=False) as archive:
        robot_qpos = np.asarray(archive["robot_qpos"]).copy()
        object_qpos = np.asarray(archive["object_qpos"]).copy()
    robot_qpos[1, 0, 3:7] = 0.0
    np.savez_compressed(
        path,
        robot_qpos=robot_qpos,
        object_qpos=object_qpos,
        fps=np.asarray(30, dtype=np.int64),
        object_name=np.asarray("Box026"),
    )

    with pytest.raises(ValueError, match="non-unit quaternions"):
        module.load_pair_reference(path)


def test_load_pair_reference_rejects_fractional_fps(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "pair.npz"
    _write_pair(path)
    with np.load(path, allow_pickle=False) as archive:
        robot_qpos = np.asarray(archive["robot_qpos"]).copy()
        object_qpos = np.asarray(archive["object_qpos"]).copy()
    np.savez_compressed(
        path,
        robot_qpos=robot_qpos,
        object_qpos=object_qpos,
        fps=np.asarray(29.5),
        object_name=np.asarray("Box026"),
    )

    with pytest.raises(ValueError, match="finite integer"):
        module.load_pair_reference(path)


def test_resolve_object_urdf_uses_run_assets_by_default(tmp_path: Path) -> None:
    module = _load_script()
    pair_path = tmp_path / "core4d_pair_reference.npz"
    pair_path.touch()
    object_path = tmp_path / "assets" / "Box026.urdf"
    object_path.parent.mkdir()
    object_path.write_text("<robot/>", encoding="utf-8")

    assert module.resolve_object_urdf(pair_path, "Box026", None) == object_path.resolve()


def test_resolve_object_urdf_rejects_path_components(tmp_path: Path) -> None:
    module = _load_script()

    with pytest.raises(ValueError, match="plain filename stem"):
        module.resolve_object_urdf(tmp_path / "pair.npz", "../Box026", None)


def test_load_collision_box_proxies_uses_only_collision_geometry_and_full_sizes(
    tmp_path: Path,
) -> None:
    module = _load_script()
    object_urdf = tmp_path / "desk.urdf"
    object_urdf.write_text(
        """\
<robot name="desk">
  <link name="base">
    <visual name="must_not_be_shown">
      <geometry><box size="99 99 99"/></geometry>
    </visual>
    <collision name="tabletop">
      <origin xyz="1 2 3" rpy="0 0 0"/>
      <geometry><box size="0.6 0.05 0.8"/></geometry>
    </collision>
    <collision name="mesh_collision">
      <geometry><mesh filename="desk.obj"/></geometry>
    </collision>
  </link>
  <link name="child">
    <collision name="leg">
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry><box size="0.06 0.65 0.075"/></geometry>
    </collision>
  </link>
  <joint name="child_fixed" type="fixed">
    <parent link="base"/>
    <child link="child"/>
    <origin xyz="10 20 30" rpy="0 0 1.5707963267948966"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    proxies = module.load_collision_box_proxies(object_urdf)

    assert [proxy.name for proxy in proxies] == ["tabletop", "leg"]
    assert proxies[0].dimensions == (0.6, 0.05, 0.8)
    assert proxies[0].position == (1.0, 2.0, 3.0)
    assert proxies[0].wxyz == (1.0, 0.0, 0.0, 0.0)
    assert proxies[1].dimensions == (0.06, 0.65, 0.075)
    np.testing.assert_allclose(proxies[1].position, (10.0, 21.0, 30.0))
    np.testing.assert_allclose(
        proxies[1].wxyz,
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
    )


def test_collision_proxy_scene_and_checkbox_are_default_off() -> None:
    module = _load_script()

    class FakeScene:
        def __init__(self) -> None:
            self.frame_calls = []
            self.box_calls = []

        def add_frame(self, name, **kwargs):
            self.frame_calls.append((name, kwargs))
            return SimpleNamespace(name=name, **kwargs)

        def add_box(self, name, **kwargs):
            self.box_calls.append((name, kwargs))
            return SimpleNamespace(name=name, **kwargs)

    class FakeCheckbox:
        def __init__(self, value: bool) -> None:
            self.value = value
            self.callback = None

        def on_update(self, callback):
            self.callback = callback
            return callback

    class FakeGui:
        def __init__(self) -> None:
            self.calls = []

        def add_checkbox(self, label, *, initial_value):
            self.calls.append((label, initial_value))
            return FakeCheckbox(initial_value)

    proxy = module.CollisionBoxProxy(
        name="tabletop",
        position=(1.0, 2.0, 3.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        dimensions=(0.6, 0.05, 0.8),
    )
    scene = FakeScene()
    root_handle, box_handles = module.add_collision_box_proxies(scene, (proxy,))

    assert scene.frame_calls == [
        (
            "/object/collision_proxies",
            {"show_axes": False, "visible": False},
        )
    ]
    assert root_handle.visible is False
    assert len(box_handles) == 1
    box_name, box_kwargs = scene.box_calls[0]
    assert box_name.startswith("/object/collision_proxies/")
    assert box_kwargs["dimensions"] == proxy.dimensions
    assert box_kwargs["position"] == proxy.position
    assert box_kwargs["wxyz"] == proxy.wxyz
    assert box_kwargs["opacity"] == module.COLLISION_PROXY_OPACITY

    gui = FakeGui()
    checkbox = module.add_collision_proxy_visibility_control(gui, root_handle)
    assert gui.calls == [("Show collision proxies", False)]
    assert checkbox.callback is not None
    checkbox.value = True
    checkbox.callback(None)
    assert root_handle.visible is True


def test_validate_robot_urdf_contract_requires_rubber_hands_and_joint_order() -> None:
    module = _load_script()
    links = [SimpleNamespace(name=name) for name in sorted(module.RUBBER_HAND_LINKS)]
    joints = [
        SimpleNamespace(name=name, type="revolute")
        for name in module.EXPECTED_ROBOT_JOINT_NAMES
    ]
    model = SimpleNamespace(robot=SimpleNamespace(links=links, joints=joints))

    module.validate_robot_urdf_contract(model)

    model.robot.links.pop()
    with pytest.raises(ValueError, match="rubber-hand"):
        module.validate_robot_urdf_contract(model)
