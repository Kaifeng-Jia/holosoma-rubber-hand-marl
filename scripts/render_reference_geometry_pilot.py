#!/usr/bin/env python3
"""True-mesh source/baseline evidence, not a new-candidate comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image
from render_wrist_reference_pairs import VisualScene
import mujoco
from reference_geometry_pilot import ROBOT, write, sha

FFMPEG = Path('/home/kevin/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); records = []
    for case in ['push', 'pull']:
        folder = args.output / case
        manifest = json.loads((folder / 'input_manifest.json').read_text())
        src = np.load(folder / 'preprocessed_source.npz')
        old = np.load(folder / 'legacy_pair_reference.npz')
        # One camera per case for both input and legacy views. These have
        # different object geometries and are NOT an old/new method comparison.
        points = np.concatenate([old['robot_qpos'][..., :3].reshape(-1, 3), old['object_qpos'][:, :3]])
        center = (points.min(0) + points.max(0)) / 2
        center[2] = .65
        camera = dict(lookat=center.tolist(), distance=max(3.2, np.linalg.norm(np.ptp(points, axis=0))*1.7),
                      azimuth=140 if case=='push' else 55, elevation=-18)
        frames = [30, 117, 230] if case=='push' else [40, 150, 240]
        for name, data, object_key in [('source', src, 'source_object'), ('legacy_pair', old, 'target_object')]:
            robot = data['robot_qpos']; obj = data['object_qpos']; fps = int(data['fps'])
            if robot.ndim == 2: robot = robot[:, None]
            dest = folder / 'media' / name; dest.mkdir(parents=True, exist_ok=False)
            scene = VisualScene(ROBOT, Path(manifest[object_key]['path']), robot.shape[1], 1280, 720)
            video = dest / 'full_reference.mp4'
            command = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
                       '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', str(fps), '-i', '-', '-an',
                       '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-threads', '2',
                       '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = camera['lookat']; cam.distance = camera['distance']
            cam.azimuth = camera['azimuth']; cam.elevation = camera['elevation']
            images = []
            try:
                for t in range(len(robot)):
                    scene.set_pose(robot[t], obj[t]); scene.renderer.update_scene(scene.data, camera=cam)
                    pixels = scene.renderer.render(); process.stdin.write(pixels.tobytes())
                    if t in frames:
                        file = dest / f'frame_{t:03d}_whole.png'; Image.fromarray(pixels).save(file)
                        images.append(dict(path=str(file.relative_to(args.output)), frame=t, time_s=t/fps, camera=camera))
                        for side in ['left', 'right']:
                            close = dict(camera, lookat=scene.hand_center(0, side).tolist(), distance=.65)
                            file = dest / f'frame_{t:03d}_{side}_hand.png'; scene.render(close, file)
                            images.append(dict(path=str(file.relative_to(args.output)), frame=t, time_s=t/fps,
                                               robot_index=0, hand=side, camera=close))
                process.stdin.close()
                if process.wait() != 0: raise RuntimeError('Video encoding failed')
            finally:
                scene.renderer.close()
                if process.poll() is None: process.kill(); process.wait()
            records.append(dict(case=case, variant=name, images=images, video=str(video.relative_to(args.output)),
                                video_sha256=sha(video), frames=len(robot), fps=fps,
                                robot_urdf=str(ROBOT), object=manifest[object_key], camera=camera))
            print(case, name, 'rendered', len(robot), 'frames', flush=True)
    write(args.output/'render_manifest.json', dict(records=records, physics_steps=0,
        scope='Input and legacy diagnostics only; no new candidate selected, no old/new improvement comparison.',
        pose_changes=False, image_labels=False, renderer='Actual URDF visual meshes; MuJoCo mocap FK/rasterizer',
        resolution=[1280,720], ffmpeg=str(FFMPEG), script_sha256=sha(Path(__file__))))


if __name__ == '__main__': main()
