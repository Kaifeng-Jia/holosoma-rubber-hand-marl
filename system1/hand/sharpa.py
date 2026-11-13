"""Sharpa Wave hand model (T06): URDF parsing, batched torch forward kinematics, collision surface samples, PD table,
palm frame and the mount on the G1 wrist (CPU only).

Source: assets/sharpa (sharpa-robotics/sharpa-urdf-usd-xml, Apache 2.0), wave_01/<side>_sharpa_wave/<side>_sharpa_wave_with_flange.*

Palm frame (same construction as the human palm frame of the existing A1 wrist solver,
holosoma_retargeting/src/utils.py: build_pt_wrist_palm_orientation_targets): forward = wrist point -> middle MCP,
across = sign * (index MCP - pinky MCP) with sign +1 left / -1 right, normal = forward x across (points out of the palm),
across re-orthogonalised = normal x forward; basis columns [forward, across, normal]. Sharpa landmarks at q = 0: wrist
point = origin of <side>_hand_wrist (the hand's own wrist frame, 0.5 mm above the flange), MCP points = origins of
<side>_{index,middle,pinky}_MCP_FE (pinky at pinky_CMC = 0).

Mount (T06 "flange 坐标系对齐橡胶手原来的安装位置，掌心朝向和橡胶手的掌面一致"): flange origin at the rubber hand's
fixed palm joint origin in <side>_wrist_yaw_link (xyz = 0.0415, +-0.003, 0); flange +z (the hand's long axis) along the
wrist_yaw_link +x axis (the rubber hand's finger direction); roll about that axis chosen so the Sharpa palm normal is
closest to the rubber-hand palm normal of the A1 solver (interaction_mesh_retargeter._init_hand_orientation).
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SHARPA = ROOT / "assets/sharpa/wave_01"
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# rubber hand, A1 solver (interaction_mesh_retargeter._init_hand_orientation): palm normal in <side>_wrist_yaw_link
RUBBER_PALM_NORMAL = {"left": np.array([-0.07513681, -0.99540367, -0.05938011]),
                      "right": np.array([-0.07514936, 0.99540878, -0.05927846])}
RUBBER_MOUNT_XYZ = {"left": np.array([0.0415, 0.003, 0.0]), "right": np.array([0.0415, -0.003, 0.0])}  # <side>_hand_palm_joint


def urdf_path(side: str) -> Path:
    return SHARPA / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_with_flange.urdf"


def usd_path(side: str) -> Path:
    return SHARPA / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_with_flange.usda"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4
    axis: np.ndarray
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass
class Collision:
    link: str
    mesh: Path
    origin: np.ndarray  # 4x4 in the link frame


class SharpaHand:
    """One Sharpa Wave hand (flange version), root = <side>_hand_flange."""

    def __init__(self, side: str):
        if side not in SIDES:
            raise ValueError(side)
        self.side = side
        self.path = urdf_path(side)
        r = ET.parse(self.path).getroot()
        self.links = [l.get("name") for l in r.findall("link")]
        self.masses = {l.get("name"): float(l.find("inertial/mass").get("value")) for l in r.findall("link") if l.find("inertial/mass") is not None}
        self.joints: list[Joint] = []
        for j in r.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(x) for x in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
            a = j.find("axis")
            lim = j.find("limit")
            self.joints.append(Joint(j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"),
                                     homog(rpy_matrix(*rpy), xyz),
                                     np.array([float(x) for x in a.get("xyz").split()]) if a is not None else np.zeros(3),
                                     float(lim.get("lower")) if lim is not None and lim.get("lower") else 0.0,
                                     float(lim.get("upper")) if lim is not None and lim.get("upper") else 0.0,
                                     float(lim.get("effort")) if lim is not None and lim.get("effort") else 0.0,
                                     float(lim.get("velocity")) if lim is not None and lim.get("velocity") else 0.0))
        children = {j.child for j in self.joints}
        roots = [l for l in self.links if l not in children]
        if roots != [f"{side}_hand_flange"]:
            raise ValueError(f"unexpected root links {roots}")
        self.root = roots[0]
        # topological order
        order, frontier = [], [self.root]
        by_parent: dict[str, list[Joint]] = {}
        for j in self.joints:
            by_parent.setdefault(j.parent, []).append(j)
        while frontier:
            p = frontier.pop(0)
            for j in by_parent.get(p, []):
                order.append(j)
                frontier.append(j.child)
        self.chain = order
        self.dof = [j for j in self.chain if j.kind == "revolute"]
        self.dof_names = [j.name for j in self.dof]
        if len(self.dof) != 22 or any(j.kind not in ("revolute", "fixed") for j in self.joints):
            raise ValueError(f"expected 22 revolute + fixed joints, got {len(self.dof)}")
        self.lower = torch.tensor([j.lower for j in self.dof], dtype=torch.float64)
        self.upper = torch.tensor([j.upper for j in self.dof], dtype=torch.float64)
        self.collisions: list[Collision] = []
        for l in r.findall("link"):
            for c in l.findall("collision"):
                m = c.find("geometry/mesh")
                if m is None:
                    raise ValueError(f"non-mesh collision on {l.get('name')}")
                o = c.find("origin")
                xyz = np.array([float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()])
                rpy = [float(x) for x in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()]
                fn = m.get("filename").replace(f"package://{side}_sharpa_wave/", "")
                self.collisions.append(Collision(l.get("name"), self.path.parent / fn, homog(rpy_matrix(*rpy), xyz)))

    @property
    def mass(self) -> float:
        return sum(self.masses.values())

    def open_pose(self) -> torch.Tensor:
        """Natural open hand: every joint at 0 (inside every limit; the USD rest targets are within 1 deg of 0)."""
        q = torch.zeros(22, dtype=torch.float64)
        return torch.clamp(q, self.lower, self.upper)

    def fk(self, q: torch.Tensor, base: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Link poses (..., 4, 4) for joint angles q (..., 22) in the order of self.dof_names; base = root pose (..., 4, 4)."""
        q = torch.as_tensor(q, dtype=torch.float64)
        batch = q.shape[:-1]
        eye = torch.eye(4, dtype=torch.float64).expand(*batch, 4, 4)
        T = {self.root: base if base is not None else eye}
        k = 0
        for j in self.chain:
            Tj = T[j.parent] @ torch.as_tensor(j.origin, dtype=torch.float64)
            if j.kind == "revolute":
                Tj = Tj @ _axis_angle_h(torch.as_tensor(j.axis, dtype=torch.float64), q[..., k])
                k += 1
            T[j.child] = Tj
        return T

    @lru_cache(maxsize=4)
    def surface_samples(self, per_mesh: int = 96, seed: int = 0) -> tuple[list[str], np.ndarray]:
        """Points on the collision meshes in their link frames: (link per point, (N, 3))."""
        import trimesh
        rng = np.random.default_rng(seed)
        links, pts = [], []
        for c in self.collisions:
            m = trimesh.load(c.mesh, force="mesh")
            s, _ = trimesh.sample.sample_surface(m, per_mesh, seed=int(rng.integers(1 << 31)))
            p = (c.origin[:3, :3] @ np.vstack([s, m.vertices[rng.choice(len(m.vertices), min(32, len(m.vertices)), replace=False)]]).T).T + c.origin[:3, 3]
            links += [c.link] * len(p)
            pts.append(p)
        return links, np.vstack(pts)

    def points_world(self, q: torch.Tensor, base: torch.Tensor | None = None, per_mesh: int = 96):
        """Collision surface samples in the base frame (..., N, 3) and the link name of each point."""
        links, local = self.surface_samples(per_mesh)
        T = self.fk(q, base)
        local_t = torch.as_tensor(local, dtype=torch.float64)
        out = torch.empty(*q.shape[:-1], len(links), 3, dtype=torch.float64)
        idx: dict[str, list[int]] = {}
        for i, l in enumerate(links):
            idx.setdefault(l, []).append(i)
        for l, ii in idx.items():
            ii_t = torch.tensor(ii)
            Tl = T[l]
            out[..., ii_t, :] = (Tl[..., None, :3, :3] @ local_t[ii_t][..., None]).squeeze(-1) + Tl[..., None, :3, 3]
        return out, links

    def palm_landmarks(self) -> dict[str, np.ndarray]:
        """Wrist and MCP points at q = 0 in the flange frame."""
        T = self.fk(torch.zeros(22, dtype=torch.float64))
        s = self.side
        jo = {j.name: j for j in self.joints}
        pt = lambda jn: (T[jo[jn].parent] @ torch.as_tensor(jo[jn].origin))[:3, 3].numpy()  # joint origin  # noqa: E731
        return dict(wrist=T[f"{s}_hand_wrist"][:3, 3].numpy(), index=pt(f"{s}_index_MCP_FE"), middle=pt(f"{s}_middle_MCP_FE"),
                    pinky=pt(f"{s}_pinky_MCP_FE"))

    def palm_basis(self) -> np.ndarray:
        """Palm frame columns [forward, across, normal] in the flange frame (human A1 construction)."""
        lm = self.palm_landmarks()
        return palm_basis_from_landmarks(lm["wrist"], lm["index"], lm["middle"], lm["pinky"], self.side)


def _axis_angle_h(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    a = axis / torch.linalg.norm(axis)
    x, y, z = a
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    R = torch.stack([torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
                     torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
                     torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)
    out = torch.zeros(*angle.shape, 4, 4, dtype=torch.float64)
    out[..., :3, :3] = R
    out[..., 3, 3] = 1.0
    return out


def palm_basis_from_landmarks(wrist, index, middle, pinky, side: str) -> np.ndarray:
    forward = np.asarray(middle, float) - np.asarray(wrist, float)
    forward /= np.linalg.norm(forward)
    across = (1.0 if side == "left" else -1.0) * (np.asarray(index, float) - np.asarray(pinky, float))
    across -= across @ forward * forward
    across /= np.linalg.norm(across)
    normal = np.cross(forward, across)
    normal /= np.linalg.norm(normal)
    across = np.cross(normal, forward)
    return np.column_stack([forward, across, normal])


def rubber_palm_basis(side: str) -> np.ndarray:
    """The A1 solver's rubber-hand palm basis in <side>_wrist_yaw_link (interaction_mesh_retargeter)."""
    n = RUBBER_PALM_NORMAL[side] / np.linalg.norm(RUBBER_PALM_NORMAL[side])
    f = np.array([1.0, 0.0, 0.0]) - n[0] * n
    f /= np.linalg.norm(f)
    a = np.cross(n, f)
    a /= np.linalg.norm(a)
    return np.column_stack([f, a, n])


def mount_transform(side: str, hand: SharpaHand | None = None) -> tuple[np.ndarray, dict]:
    """<side>_wrist_yaw_link <- <side>_hand_flange (4x4) and how it was chosen."""
    hand = hand or SharpaHand(side)
    B_sh = hand.palm_basis()
    # flange +z -> wrist +x
    base = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # columns: images of flange x, y, z
    assert np.allclose(base @ np.array([0, 0, 1.0]), [1, 0, 0])
    n_target = RUBBER_PALM_NORMAL[side] / np.linalg.norm(RUBBER_PALM_NORMAL[side])
    best = None
    for k in range(3600):  # roll about wrist +x, 0.1 deg grid, then refine
        th = 2 * math.pi * k / 3600
        R = _rot_x(th) @ base
        c = float((R @ B_sh[:, 2]) @ n_target)
        if best is None or c > best[0]:
            best = (c, th)
    lo, hi = best[1] - 2 * math.pi / 3600, best[1] + 2 * math.pi / 3600
    for _ in range(60):  # golden-section refine of the roll angle
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        f = lambda th: float((_rot_x(th) @ base @ B_sh[:, 2]) @ n_target)  # noqa: E731
        lo, hi = (m1, hi) if f(m1) < f(m2) else (lo, m2)
    th = (lo + hi) / 2
    R = _rot_x(th) @ base
    B_wy = R @ B_sh
    B_rub = rubber_palm_basis(side)
    ang = lambda a, b: math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))  # noqa: E731
    info = dict(roll_about_wrist_x_deg=math.degrees(th) % 360, flange_origin_in_wrist_yaw=RUBBER_MOUNT_XYZ[side].tolist(),
                palm_normal_angle_to_rubber_deg=ang(B_wy[:, 2], B_rub[:, 2]),
                palm_forward_angle_to_rubber_deg=ang(B_wy[:, 0], B_rub[:, 0]),
                palm_basis_rotation_to_rubber_deg=math.degrees(np.linalg.norm(_rotvec(B_rub @ B_wy.T))))
    return homog(R, RUBBER_MOUNT_XYZ[side]), info


def _rot_x(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def pd_table(side: str) -> dict[str, dict[str, float]]:
    """Per finger joint: kp, kd (per rad; USD per-degree values * 180/pi), armature, friction, effort, learn-mode action scale."""
    t = usd_path(side).read_text()
    body = t.split('over "joints"', 1)[1].split(f'over "{side}_hand_C_MC"', 1)[0]
    hand = SharpaHand(side)
    lim = {j.name: j for j in hand.dof}
    out = {}
    for b in re.split(r'\n\s*over "', body)[1:]:
        name = b.split('"', 1)[0]
        g = lambda k: float(re.search(k + r'\s*=\s*([-\d.eE+]+)', b).group(1))  # noqa: E731
        kp, kd = g('drive:angular:physics:stiffness') * 180 / math.pi, g('drive:angular:physics:damping') * 180 / math.pi
        out[name] = dict(kp=kp, kd=kd, armature=g('physxJoint:armature'), friction=g('physxJoint:jointFriction'),
                         effort=lim[name].effort, action_scale=0.25 * lim[name].effort / kp)
    if set(out) != set(hand.dof_names):
        raise ValueError("USD joints differ from the URDF")
    return out
