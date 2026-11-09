#!/usr/bin/env python3
"""Render actual pre-wrist nominal chair instances versus the mean shared scene."""
import argparse,json,sys,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from render_wrist_reference_pairs import VisualScene
from export_recorded_demo_media import ROOT,sha,numbers

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dest=a.output.resolve()/'U7_media/shared_scene';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'render_manifest.json').exists():raise FileExistsError(dest)
    folder=ROOT/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908'
    sources=[folder/f'two_stage/person{i}/person{i}_nominal_scaled.npz' for i in [1,2]]
    sources.append(folder/'shared_small_preview/core4d_pair_reference.npz')
    arrays=[]
    for path in sources:
        with np.load(path) as z:arrays.append({k:z[k].copy() for k in z.files})
    pair=arrays[2];frame=117;fps=float(pair['fps']);scales=pair['human_to_robot_scales'];mean=float(pair['shared_object_scale'])
    robot=np.stack([x['qpos'][frame,:36] for x in arrays[:2]])
    assert np.array_equal(robot,pair['robot_qpos'][frame])
    nominal=np.stack([x['qpos'][frame,36:43] for x in arrays[:2]])
    shared=pair['object_qpos'][frame]
    assert np.allclose(nominal[:,:3].mean(0),shared[:3],atol=1e-6)
    srcmesh=Path(str(pair['object_mesh_path']))
    shutil.copy2(srcmesh,dest/'chair021_source.obj')
    mesh=dest/'chair021_source.obj'
    colors=[[.15,.42,.82,1],[.95,.48,.12,1]];objcolors=[[*c[:3],.42] for c in colors]
    robotpath=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
    center=np.concatenate([robot[:,:3],shared[None,:3]]).mean(0);center[2]=.75
    cam=dict(lookat=center.tolist(),distance=2.8,azimuth=135,elevation=-22)
    records=[]
    for variant,qs,ss,cs in [('individual_nominal',nominal,scales,objcolors),('shared_mean',shared[None,:],np.array([mean]),[[.55,.6,.65,1]])]:
        root=ET.Element('robot',name=variant);ET.SubElement(root,'link',name='root')
        for i,(q,s,c) in enumerate(zip(qs,ss,cs)):
            link=ET.SubElement(root,'link',name=f'object_{i}')
            visual=ET.SubElement(link,'visual',name=f'object_{i}_visual')
            geometry=ET.SubElement(visual,'geometry');ET.SubElement(geometry,'mesh',filename=mesh.name,scale=numbers([s]*3))
            material=ET.SubElement(visual,'material',name=f'color_{i}');ET.SubElement(material,'color',rgba=numbers(c))
            joint=ET.SubElement(root,'joint',name=f'object_{i}_joint',type='fixed')
            ET.SubElement(joint,'parent',link='root');ET.SubElement(joint,'child',link=f'object_{i}')
            ET.SubElement(joint,'origin',xyz=numbers(q[:3]),rpy=numbers(Rotation.from_quat(q[3:7][[1,2,3,0]]).as_euler('xyz')))
        urdf=dest/f'{variant}_visual_only.urdf';ET.ElementTree(root).write(urdf,encoding='utf-8',xml_declaration=True)
        scene=VisualScene(robotpath,urdf,2,1920,1080)
        scene.model.vis.headlight.diffuse[:]=.5;scene.model.vis.headlight.ambient[:]=.3;scene.model.vis.headlight.specular[:]=.05
        scene.model.light_pos[0]=center+[0,0,5];scene.model.light_diffuse[0]=.5
        # Same role colors in both panels. Object colors are recorded explicitly.
        for agent,node,bodyname in scene.mapping:
            if agent<2:
                import mujoco
                bid=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,bodyname)
                adr=scene.model.body_geomadr[bid];scene.model.geom_rgba[adr]=colors[agent]
        try:
            scene.set_pose(robot,np.array([0,0,0,1,0,0,0]))
            image=dest/f'{variant}.png';scene.render(cam,image)
            records.append(dict(variant=variant,image=image.name,image_sha256=sha(image),camera=cam,
                projection='perspective',fovy_degrees=float(scene.model.vis.global_.fovy),orthographic=False,
                object_scales=ss.tolist(),object_qpos_wxyz=qs.tolist(),object_rgba=cs,visual_urdf=urdf.name,
                visual_urdf_sha256=sha(urdf),mesh_records=scene.mesh_records))
        finally:scene.renderer.close()
    for i,path in enumerate(sources):shutil.copy2(path,dest/f'source_{i}.npz')
    record=dict(frame=frame,time_s=frame/fps,fps=fps,width=1920,height=1080,labels_axes_ui=False,
        actual_training_rule='mean scaling; comparison deliberately before wrist refinement',
        h_i=pair['human_heights'].tolist(),h_R=1.32,s_i=scales.tolist(),s_obj=mean,
        robot_qpos_identical_between_panels=True,robot_rgba=colors,camera_fixed_between_panels=True,
        lighting=dict(headlight_diffuse=[.5]*3,headlight_ambient=[.3]*3,headlight_specular=[.05]*3,
                      top_light_pos=(center+[0,0,5]).tolist(),top_light_diffuse=[.5]*3,
                      top_light_dir=[0,0,-1],top_light_specular=[0]*3,castshadow=True,shadowsize=2048,offsamples=4,
                      background=[.98]*3,ground_rgba=[.95,.95,.95,1],ground_emission=.5),
        scope='Reference geometry only; not actual execution or contact-force evidence',
        no_physics_steps=True,sources=[dict(path=str(p),sha256=sha(p),package_file=f'source_{i}.npz') for i,p in enumerate(sources)],
        mesh=dict(source=str(srcmesh),sha256=sha(srcmesh),package_file=mesh.name),robot_urdf=str(robotpath),robot_sha256=sha(robotpath),
        renderer_sha256=sha(ROOT/'scripts/render_wrist_reference_pairs.py'),script_sha256=sha(Path(__file__)),command=[sys.executable,*sys.argv],panels=records)
    (dest/'render_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (dest/'camera.json').write_text(json.dumps(dict(camera=cam,resolution=[1920,1080],fovy=45,lighting=record['lighting']),indent=2)+'\n')
    print('Shared chair panels complete')
if __name__=='__main__':main()
