"""Offline, asset-fixed geometry for a CORE4D interaction-graph review.

The selected vertices are geometric landmarks, not contact labels or a palm
normal. No reference, robot asset, collision shape, or training state is edited.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Expected finite wxyz quaternions")
    if np.any(np.linalg.norm(q, axis=-1) < 1e-12):
        raise ValueError("A zero quaternion is not a rotation")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(
        q.shape[:-1] + (3, 3)
    )


def build_robot_landmarks(xml_path: str | Path) -> tuple[list[dict], dict]:
    """Return the original 15 SMPL-X/G1 points followed by four hand vertices.

    MuJoCo stores compiled mesh vertices in its own mesh frame. Applying the
    compiled geom rotation/translation restores body-local coordinates.
    """
    import mujoco
    from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS

    path = Path(xml_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    mapping = JOINTS_MAPPINGS[("smplx", "g1")]
    if len(mapping) != 15:
        raise ValueError("Expected the reviewed 15-point Omni SMPL-X/G1 mapping")
    specs = []
    for name, body in mapping.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
            raise ValueError(f"Missing mapped body: {body}")
        specs.append({
            "name": name,
            "body": body,
            "local_xyz": [0.0, 0.0, 0.0],
            "group": {"L_Wrist": "left_hand", "R_Wrist": "right_hand"}.get(name, name),
            "kind": "omni_body_origin",
        })

    hand_metadata = {}
    for side in ("left", "right"):
        body = f"{side}_rubber_hand_link"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, body)
        if geom_id < 0 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"Expected rubber-hand mesh geom: {body}")
        owner = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        if owner != body:
            raise ValueError(f"Hand mesh owner mismatch: {owner} != {body}")
        mesh_id = int(model.geom_dataid[geom_id])
        v0, nv = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        f0, nf = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[v0 : v0 + nv].astype(np.float64)
        faces = model.mesh_face[f0 : f0 + nf].astype(np.int64)
        rotation = _rotation_wxyz(model.geom_quat[geom_id])
        body_vertices = vertices @ rotation.T + model.geom_pos[geom_id]
        # Consider only vertices which participate in real mesh triangles.
        used = np.unique(faces)
        first = int(used[np.argmax(np.sum(body_vertices[used] ** 2, axis=1))])
        first_point = body_vertices[first]
        first_norm = float(np.linalg.norm(first_point))
        if first_norm < 1e-10:
            raise ValueError(f"Degenerate hand mesh: {body}")
        off_axis = np.linalg.norm(np.cross(first_point, body_vertices[used]), axis=1) / first_norm
        second = int(used[np.argmax(off_axis)])
        area = float(0.5 * np.linalg.norm(np.cross(first_point, body_vertices[second])))
        if area < 1e-10:
            raise ValueError(f"Hand mesh points are collinear with the original origin: {body}")
        for suffix, vertex_id in (("farthest", first), ("off_axis", second)):
            specs.append({
                "name": f"{side}_rubber_hand_surface_{suffix}",
                "body": body,
                "local_xyz": body_vertices[vertex_id].tolist(),
                "group": f"{side}_hand",
                "kind": "fixed_mesh_vertex",
                "mesh_vertex_index": vertex_id,
            })
        hand_metadata[side] = {
            "body": body,
            "geom_name": body,
            "mesh_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id),
            "mesh_vertex_count": nv,
            "mesh_face_count": nf,
            "compiled_geom_pos_body": model.geom_pos[geom_id].tolist(),
            "compiled_geom_quat_wxyz": model.geom_quat[geom_id].tolist(),
            "selected_vertex_indices": [first, second],
            "origin_surface_surface_triangle_area_m2": area,
            "body_local_vertices_sha256": hashlib.sha256(body_vertices.tobytes()).hexdigest(),
        }
    metadata = {
        "version": "omni15_plus_four_rubber_hand_mesh_points_v1",
        "xml_path": str(path),
        "xml_sha256": _sha256(path),
        "original_mapping": dict(mapping),
        "num_landmarks": len(specs),
        "selection": "original origins unchanged; farthest mesh vertex then farthest from origin-first axis",
        "quaternion_order": "wxyz",
        "hands": hand_metadata,
        "contact_semantics": "geometric landmarks only; no force or contact or precise palm-normal claim",
    }
    return specs, metadata


def landmarks_world(
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    body_names: list[str] | np.ndarray,
    specs: list[dict],
) -> np.ndarray:
    """Apply fixed local landmark offsets to body poses, including [T,2,B,*]."""
    pos = np.asarray(positions, dtype=np.float64)
    quat = np.asarray(quats_wxyz, dtype=np.float64)
    names = [str(name) for name in body_names]
    if pos.shape[-2:] != (len(names), 3) or quat.shape != pos.shape[:-1] + (4,):
        raise ValueError("Body positions, wxyz rotations, and body_names have incompatible shapes")
    if len(set(names)) != len(names) or not np.isfinite(pos).all():
        raise ValueError("Expected unique body names and finite positions")
    lookup = {name: index for index, name in enumerate(names)}
    indices = [lookup[item["body"]] for item in specs]
    offsets = np.asarray([item["local_xyz"] for item in specs], dtype=np.float64)
    rotation = _rotation_wxyz(quat[..., indices, :])
    return pos[..., indices, :] + np.einsum("...nij,nj->...ni", rotation, offsets)


def _numbers(text: str | None, default: str) -> np.ndarray:
    result = np.fromstring(text if text is not None else default, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Expected three finite URDF numbers")
    return result


def load_box_object_mesh(urdf_path: str | Path) -> tuple[trimesh.Trimesh, dict]:
    """Construct the exposed boundary of a single-link axis-aligned box union.

    Face rectangles are split at all other box bounds before checking whether
    their outward side is buried. This excludes partially hidden surfaces
    without a boolean-mesh dependency or any modification of collision assets.
    """
    path = Path(urdf_path).resolve()
    root = ET.parse(path).getroot()
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise ValueError("Expected a fixed single-link URDF object")
    components = []
    for index, collision in enumerate(links[0].findall("collision")):
        box = collision.find("geometry/box")
        if box is None:
            raise ValueError("This offline mesh builder requires box collision geometry")
        origin = collision.find("origin")
        xyz = _numbers(None if origin is None else origin.get("xyz"), "0 0 0")
        rpy = _numbers(None if origin is None else origin.get("rpy"), "0 0 0")
        if np.max(np.abs(rpy)) > 1e-12:
            raise ValueError("Rotated boxes are not supported by the axis-aligned exposed-surface builder")
        size = _numbers(box.get("size"), "0 0 0")
        if np.any(size <= 0):
            raise ValueError("Box sizes must be positive")
        components.append({
            "name": collision.get("name", f"box_{index}"),
            "center_xyz": xyz.tolist(),
            "size_xyz": size.tolist(),
            "lower_xyz": (xyz - size / 2).tolist(),
            "upper_xyz": (xyz + size / 2).tolist(),
        })
    if not components:
        raise ValueError("No box collisions found")
    lower = np.asarray([item["lower_xyz"] for item in components])
    upper = np.asarray([item["upper_xyz"] for item in components])
    scale = float(np.max(upper.max(axis=0) - lower.min(axis=0)))
    epsilon = max(scale * 1e-8, 1e-11)
    vertices, faces, component_ids, face_axes = [], [], [], []
    buried_area = 0.0
    for component in range(len(components)):
        for axis in range(3):
            uv = [value for value in range(3) if value != axis]
            cuts = []
            for dimension in uv:
                candidates = np.concatenate((lower[:, dimension], upper[:, dimension]))
                interior = candidates[(candidates > lower[component, dimension]) & (candidates < upper[component, dimension])]
                cuts.append(np.unique(np.r_[lower[component, dimension], interior, upper[component, dimension]]))
            for sign in (-1, 1):
                plane = lower[component, axis] if sign < 0 else upper[component, axis]
                normal = np.zeros(3)
                normal[axis] = sign
                for u0, u1 in zip(cuts[0][:-1], cuts[0][1:]):
                    for v0, v1 in zip(cuts[1][:-1], cuts[1][1:]):
                        center = np.zeros(3)
                        center[axis] = plane
                        center[uv] = [(u0 + u1) / 2, (v0 + v1) / 2]
                        probe = center + epsilon * normal
                        inside = np.all(probe > lower, axis=1) & np.all(probe < upper, axis=1)
                        inside[component] = False
                        # Coincident exposed rectangles are represented once, by
                        # the lower-indexed component, instead of double sampling.
                        contained = np.all(center >= lower - epsilon / 10, axis=1) & np.all(center <= upper + epsilon / 10, axis=1)
                        duplicate = bool(np.any(contained[:component]))
                        if inside.any() or duplicate:
                            buried_area += float((u1 - u0) * (v1 - v0))
                            continue
                        quad = np.tile(center, (4, 1))
                        quad[:, uv] = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
                        first_face = [0, 1, 2]
                        second_face = [0, 2, 3]
                        if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), normal) < 0:
                            first_face.reverse()
                            second_face.reverse()
                        start = len(vertices)
                        vertices.extend(quad.tolist())
                        faces.extend([[start + i for i in first_face], [start + i for i in second_face]])
                        component_ids.extend([component, component])
                        face_axes.extend([f"{'+' if sign > 0 else '-'}{'xyz'[axis]}"] * 2)
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    metadata = {
        "version": "single_link_axis_aligned_box_union_exterior_v1",
        "urdf_path": str(path),
        "urdf_sha256": _sha256(path),
        "link_name": links[0].get("name"),
        "coordinate_frame": "unchanged URDF link frame, metres; not COM or world frame",
        "components": components,
        "face_component_ids": component_ids,
        "face_axis_labels": face_axes,
        "sampling_surface_area_m2": float(mesh.area),
        "removed_buried_or_duplicate_area_m2": buried_area,
        "face_count": len(mesh.faces),
        "vertices_faces_sha256": hashlib.sha256(mesh.vertices.tobytes() + mesh.faces.tobytes()).hexdigest(),
        "urdf_modified": False,
    }
    return mesh, metadata


def _sample_counts(face_ids: np.ndarray, metadata: dict) -> dict:
    components = np.asarray(metadata["face_component_ids"], dtype=int)[face_ids]
    axes = np.asarray(metadata["face_axis_labels"])[face_ids]
    result = {}
    for index, item in enumerate(metadata["components"]):
        chosen = components == index
        result[item["name"]] = {
            "count": int(chosen.sum()),
            "face_axis_counts": {axis: int(np.sum(chosen & (axes == axis))) for axis in ("-x", "+x", "-y", "+y", "-z", "+z")},
        }
    return result


def sample_object_points(mesh: trimesh.Trimesh, metadata: dict, count: int, seed: int = 42) -> tuple[np.ndarray, dict]:
    """Use Omni's even surface sampler verbatim, without filling short returns."""
    if count <= 0:
        raise ValueError("Requested sample count must be positive")
    points, face_ids = trimesh.sample.sample_surface_even(mesh, int(count), seed=int(seed))
    points = np.asarray(points, dtype=np.float64)
    return points, {
        "sampler": "trimesh.sample.sample_surface_even",
        "requested_count": int(count),
        "actual_count": len(points),
        "seed": int(seed),
        "filled_shortfall": False,
        "sampling_mesh_sha256": metadata["vertices_faces_sha256"],
        "points_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "face_indices": face_ids.tolist(),
        "per_component": _sample_counts(face_ids, metadata),
    }


def coverage_report(mesh: trimesh.Trimesh, metadata: dict, points: np.ndarray, seed: int = 2026) -> dict:
    """Compare any budget to the same seeded 4096-point dense surface pool."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Coverage requires a nonempty finite [N,3] point array")
    dense, face_ids = trimesh.sample.sample_surface(mesh, count=4096, seed=int(seed))
    distances = cKDTree(points).query(dense)[0]
    component_ids = np.asarray(metadata["face_component_ids"], dtype=int)[face_ids]
    axis_labels = np.asarray(metadata["face_axis_labels"])[face_ids]

    def summary(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {"dense_count": 0, "mean_m": None, "p95_m": None, "max_m": None}
        return {"dense_count": len(values), "mean_m": float(values.mean()), "p95_m": float(np.quantile(values, 0.95)), "max_m": float(values.max())}

    per_component = {}
    for index, item in enumerate(metadata["components"]):
        chosen = component_ids == index
        per_component[item["name"]] = {
            **summary(distances[chosen]),
            "face_axes": {axis: summary(distances[chosen & (axis_labels == axis)]) for axis in ("-x", "+x", "-y", "+y", "-z", "+z")},
        }
    return {
        "method": "fixed_seed_area_weighted_dense_surface_pool_nearest_sample",
        "dense_seed": int(seed),
        "dense_count": len(dense),
        "dense_points_sha256": hashlib.sha256(dense.tobytes()).hexdigest(),
        "sample_count": len(points),
        "overall": summary(distances),
        "per_component": per_component,
        "interpretation": "finite surface coverage audit, not proof of contact or a worst-case geometric bound",
    }
