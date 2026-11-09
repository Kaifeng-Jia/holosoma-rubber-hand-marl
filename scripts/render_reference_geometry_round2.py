#!/usr/bin/env python3
"""Same-frame/camera actual-mesh old/new reference media; no time integration."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import mujoco
from PIL import Image
from render_wrist_reference_pairs import VisualScene
from reference_geometry_pilot import ROBOT,write,sha
from run_reference_geometry_round2 import BATCH
from render_reference_geometry_pilot import FFMPEG


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();records=[]
    for case in ['push','pull']:
        folder=args.output/case;m=json.loads((folder/'input_manifest.json').read_text());selection=json.loads((folder/'selection.json').read_text())
        data=[np.load(BATCH/case/'legacy_pair_reference.npz'),np.load(folder/'new_pair_reference.npz')]
        points=np.concatenate([z['robot_qpos'][...,:3].reshape(-1,3) for z in data]+[data[0]['object_qpos'][:,:3]])
        center=(points.min(0)+points.max(0))/2;center[2]=.65
        whole=dict(lookat=center.tolist(),distance=max(3.2,np.linalg.norm(np.ptp(points,axis=0))*1.55),azimuth=140 if case=='push' else 55,elevation=-18)
        frames=[30,117,230] if case=='push' else [40,150,240]
        scenes=[VisualScene(ROBOT,Path(m['target_object']['path']),2,1280,720) for _ in range(2)]
        cameras={}
        for t in frames:
            for sc,z in zip(scenes,data):sc.set_pose(z['robot_qpos'][t],z['object_qpos'][t])
            for agent in [0,1]:
                for hand in ['left','right']:
                    center=np.mean([s.hand_center(agent,hand) for s in scenes],axis=0)
                    direction=center-data[0]['object_qpos'][t,:3];azimuth=float(np.degrees(np.arctan2(direction[1],direction[0])))
                    cameras[(t,agent,hand)]=dict(lookat=center.tolist(),distance=.75,azimuth=azimuth,elevation=-8)
        for variant,sc,z in zip(['old','new'],scenes,data):
            dest=folder/'media'/variant;dest.mkdir(parents=True,exist_ok=False)
            video=dest/'full_reference.mp4';fps=int(z['fps']);images=[]
            command=[str(FFMPEG),'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-',
                '-an','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            proc=subprocess.Popen(command,stdin=subprocess.PIPE)
            camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=whole['lookat'];camera.distance=whole['distance'];camera.azimuth=whole['azimuth'];camera.elevation=whole['elevation']
            try:
                for t in range(len(z['robot_qpos'])):
                    sc.set_pose(z['robot_qpos'][t],z['object_qpos'][t]);sc.renderer.update_scene(sc.data,camera=camera);pixels=sc.renderer.render();proc.stdin.write(pixels.tobytes())
                    if t in frames:
                        file=dest/f'frame_{t:03d}_whole.png';Image.fromarray(pixels).save(file);images.append(dict(file=str(file.relative_to(args.output)),frame=t,time_s=t/fps,camera=whole))
                        for agent in [0,1]:
                            for hand in ['left','right']:
                                file=dest/f'frame_{t:03d}_agent{agent}_{hand}.png';settings=cameras[(t,agent,hand)];sc.render(settings,file)
                                images.append(dict(file=str(file.relative_to(args.output)),frame=t,time_s=t/fps,agent=agent,hand=hand,camera=settings))
                proc.stdin.close();code=proc.wait()
                if code:raise RuntimeError('ffmpeg failed')
            finally:
                sc.renderer.close()
                if proc.poll() is None:proc.kill();proc.wait()
            records.append(dict(case=case,variant=variant,status='legacy_reference' if variant=='old' else selection['status'],
                video=str(video.relative_to(args.output)),video_sha256=sha(video),images=images,frames=len(z['robot_qpos']),fps=fps))
            print(case,variant,'rendered',flush=True)
    write(args.output/'render_manifest.json',dict(records=records,shared_old_new_cameras=True,physics_steps=0,
        images_contain_labels=False,scope='Saved kinematic references, not policy execution. New may be diagnostic_only; see status.',
        visual_assets='Unmodified actual target URDF and robot visual meshes/materials',resolution=[1280,720]))

if __name__=='__main__':main()
