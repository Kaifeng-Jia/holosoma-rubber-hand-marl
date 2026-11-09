#!/usr/bin/env python3
"""Offline paired reference figures: exact saved poses, no solver or mj_step.

The visual-mesh/mocap approach is shared with export_recorded_demo_media.py.
Source identity, wrist-only invariants, cameras and mesh hashes are recorded.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yourdfpy

from export_recorded_demo_media import sha, numbers, pose
from visualize_core4d_pair import EXPECTED_ROBOT_JOINT_NAMES

WRIST_COLUMNS = [26, 27, 28, 33, 34, 35]


def load_reference(path, kind):
    with np.load(path, allow_pickle=False) as archive:
        raw = {k: archive[k].copy() for k in archive.files}
    if kind == 'push':
        robot = raw['qpos'][:, None, :36]
        obj = raw['qpos'][:, 36:43]
    else:
        robot, obj = raw['robot_qpos'], raw['object_qpos']
    if robot.shape[2] != 36 or obj.shape != (len(robot), 7):
        raise ValueError('Unexpected reference pose schema')
    for array in (robot, obj):
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite pose')
        if not np.allclose(np.linalg.norm(array[..., 3:7], axis=-1), 1, atol=1e-5):
            raise ValueError('Quaternion is not unit length')
    return robot, obj, raw


def validate_pair(a, b, spec):
    robot_a, obj_a, raw_a = a
    robot_b, obj_b, raw_b = b
    assert robot_a.shape == robot_b.shape == tuple(spec['robot_shape'])
    non_wrist = [i for i in range(36) if i not in WRIST_COLUMNS]
    assert np.array_equal(robot_a[..., non_wrist], robot_b[..., non_wrist])
    assert np.array_equal(obj_a, obj_b)
    assert int(raw_a['fps']) == int(raw_b['fps']) == 30
    unchanged = ['human_joints'] if spec['name'] == 'push' else [
        'human_heights', 'human_to_robot_scales', 'scale_anchor', 'shared_object_scale']
    for key in unchanged:
        assert np.array_equal(raw_a[key], raw_b[key]), key
    changed = np.flatnonzero(np.any(robot_a != robot_b, axis=(0, 1))).tolist()
    assert set(changed).issubset(WRIST_COLUMNS) and changed
    return dict(non_wrist_bit_exact=True, object_bit_exact=True,
                additional_bit_exact_fields=unchanged, changed_qpos_columns=changed,
                robot_shape=list(robot_a.shape), frames=len(robot_a), fps=30,
                quaternion_order='wxyz', world_axes='Original Z-up world, no display transform',
                indices='0-based original30Hz frame; not runtime50Hz')


class VisualScene:
    def __init__(self, robot_path, object_path, count, width, height):
        self.urdfs = [yourdfpy.URDF.load(str(p.resolve()), load_collision_meshes=False,
                                      build_collision_scene_graph=False)
                      for p in [robot_path] * count + [object_path]]
        for u in self.urdfs[:count]:
            assert tuple(u.actuated_joint_names) == EXPECTED_ROBOT_JOINT_NAMES
            assert {'left_rubber_hand_link', 'right_rubber_hand_link'} <= set(u.link_map)
        self.count = count
        xml = ET.Element('mujoco', model='exact_reference_visual_only')
        visual = ET.SubElement(xml, 'visual')
        ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height), fovy='45')
        ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='4')
        ET.SubElement(visual, 'headlight', diffuse='.55 .55 .55', ambient='.3 .3 .3', specular='.05 .05 .05')
        asset = ET.SubElement(xml, 'asset')
        ET.SubElement(asset, 'material', name='neutral_ground', rgba='.95 .95 .95 1', emission='.5')
        ET.SubElement(asset, 'texture', type='skybox', builtin='gradient', rgb1='.98 .98 .98', rgb2='.98 .98 .98', width='64', height='64')
        world = ET.SubElement(xml, 'worldbody')
        ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1', diffuse='.45 .45 .45', specular='0 0 0', castshadow='true')
        ET.SubElement(world, 'geom', type='plane', size='20 20 .01', material='neutral_ground', contype='0', conaffinity='0')
        assets, self.mapping, mesh_ids = {}, [], {}
        self.mesh_records = []
        for agent, u in enumerate(self.urdfs):
            if not u.scene.geometry:
                raise ValueError('Empty visual scene')
            for node in u.scene.graph.nodes_geometry:
                _, geom = u.scene.graph.get(node)
                mesh = u.scene.geometry[geom]
                identity = ('robot' if agent < count else 'object', geom)
                if identity not in mesh_ids:
                    mid = f'mesh_{len(mesh_ids)}'; mesh_ids[identity] = mid
                    assets[mid + '.stl'] = mesh.export(file_type='stl')
                    ET.SubElement(asset, 'mesh', name=mid, file=mid + '.stl')
                body_name = f'visual_{len(self.mapping)}'
                body = ET.SubElement(world, 'body', name=body_name, mocap='true')
                if hasattr(mesh.visual, 'main_color'):
                    color = np.asarray(mesh.visual.main_color) / 255
                else:
                    material = mesh.visual.material
                    if getattr(material, 'image', None) is not None:
                        raise ValueError('Textured asset requires explicit texture support; do not silently recolor')
                    color = np.asarray(material.diffuse) / 255
                ET.SubElement(body, 'geom', type='mesh', mesh=mesh_ids[identity],
                              rgba=numbers(color),
                              contype='0', conaffinity='0', mass='0')
                self.mapping.append((agent, node, body_name))
        for path in (robot_path, object_path):
            tree = ET.parse(path)
            for node in tree.findall('.//visual/geometry/mesh'):
                p = Path(node.attrib['filename'])
                p = p if p.is_absolute() else path.parent / p
                self.mesh_records.append(dict(urdf=str(path), file=str(p.resolve()), sha256=sha(p),
                                              scale=node.attrib.get('scale', '1 1 1')))
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'), assets)
        self.data = mujoco.MjData(self.model)
        self.ids = [self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)] for _, _, n in self.mapping]
        self.renderer = mujoco.Renderer(self.model, height, width)

    def set_pose(self, robot, obj):
        self.bases = [pose(q[:3], q[3:7][[1, 2, 3, 0]]) for q in robot]
        self.bases.append(pose(obj[:3], obj[3:7][[1, 2, 3, 0]]))
        for u, q in zip(self.urdfs[:self.count], robot):
            u.update_cfg(q[7:36])
        for (agent, node, _), mocap in zip(self.mapping, self.ids):
            tf = self.bases[agent] @ self.urdfs[agent].scene.graph.get(node)[0]
            self.data.mocap_pos[mocap] = tf[:3, 3]
            self.data.mocap_quat[mocap] = Rotation.from_matrix(tf[:3, :3]).as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)  # Exact saved geometry; NO mj_step.

    def hand_center(self, agent, side):
        return (self.bases[agent] @ self.urdfs[agent].get_transform(side + '_rubber_hand_link'))[:3, 3]

    def render(self, settings, output):
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = settings['lookat']; camera.distance = settings['distance']
        camera.azimuth = settings['azimuth']; camera.elevation = settings['elevation']
        self.renderer.update_scene(self.data, camera=camera)
        Image.fromarray(self.renderer.render()).save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); spec = json.loads(args.spec.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    manifests = []
    for item in spec['scenes']:
        paths = [Path(item[k]) for k in ('before', 'after')]
        for path, expected in zip(paths, item['source_sha256']):
            assert sha(path) == expected, path
        a, b = [load_reference(path, item['name']) for path in paths]
        invariants = validate_pair(a, b, item)
        dest = args.output / item['name']; dest.mkdir()
        robot_path, object_path = Path(item['robot_urdf']), Path(item['object_urdf'])
        scene = VisualScene(robot_path, object_path, a[0].shape[1], spec['width'], spec['height'])
        images = []
        try:
            for selection in item['candidates']:
                frame, agent, side = selection['frame'], selection['agent'], selection['hand']
                centers = []
                for arr in (a, b):
                    scene.set_pose(arr[0][frame], arr[1][frame]); centers.append(scene.hand_center(agent, side))
                hand_camera = dict(selection['hand_camera'], lookat=np.mean(centers, axis=0).tolist())
                context_camera = dict(selection.get('context_camera', item['context_camera']))
                for stage, arr in zip(('before', 'after'), (a, b)):
                    scene.set_pose(arr[0][frame], arr[1][frame])
                    for view, cam in [('context', context_camera), ('hand', hand_camera)]:
                        filename = f'frame_{frame:03d}_{stage}_{view}.png'
                        scene.render(cam, dest / filename)
                        images.append(dict(file=item['name']+'/'+filename, sha256=sha(dest/filename),
                                           frame=frame, time_s=frame/30, variant=stage, view=view,
                                           agent_index_0based=agent, source_person='person'+str(agent+1), hand=side,
                                           camera=cam, projection='perspective', vertical_fov_degrees=45))
        finally:
            scene.renderer.close()
        for path, stage in zip(paths, ('before', 'after')):
            shutil.copyfile(path, dest/(stage+'_original_reference.npz'))
        manifests.append(dict(name=item['name'], sources=[dict(path=str(path), sha256=sha(path)) for path in paths],
                              invariants=invariants, robot_urdf=dict(path=str(robot_path),sha256=sha(robot_path)),
                              object_urdf=dict(path=str(object_path),sha256=sha(object_path)),
                              visual_meshes=scene.mesh_records, resolution=[spec['width'],spec['height']],
                              lighting=dict(background_rgb=[.98]*3,ground_rgb=[.95]*3,ground_emission=.5,headlight_diffuse=[.55]*3,
                                            headlight_ambient=[.3]*3,light_pos=[0,0,5],light_diffuse=[.45]*3),
                              images=images))
        print('Rendered', item['name'], len(images), 'native PNGs', flush=True)
    record=dict(scenes=manifests, script=str(Path(__file__).resolve()),script_sha256=sha(Path(__file__)),
                command=[sys.executable,*sys.argv], tool='yourdfpy visual FK + MuJoCo mocap rasterizer',
                no_physics_steps=True, no_pose_changes=True, labels_axes_ui=False,
                scope='Reference palm orientation illustration; no contact/load/policy performance claim')
    (args.output/'render_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
