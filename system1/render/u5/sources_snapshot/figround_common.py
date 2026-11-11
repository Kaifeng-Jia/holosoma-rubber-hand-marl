"""Frozen-input CPU figure export helpers. No training or physics stepping."""
import os,sys,ast,json,hashlib,shutil,inspect,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
ROOT=Path('/home/kevin/holosoma-core4d-base');sys.path.insert(0,str(ROOT/'scripts'))
assert os.environ.get('MUJOCO_GL')=='osmesa' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
import mujoco,yourdfpy
from OpenGL import GL
from render_wrist_reference_pairs import VisualScene,load_reference,validate_pair
from export_recorded_demo_media import numbers,pose,configured_joint_names
from visualize_core4d_pair import EXPECTED_ROBOT_JOINT_NAMES
OUT=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_approved'
R0=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_patch01'
OLD=ROOT/'logs/PaperSupplement20260929/unified_evaluation_20261003'
NEW=ROOT/'logs/ExperimentsR2_evaluation_20261026_retry01'
R2=ROOT/'paper_handoff_exports/EXPERIMENTS_R2_20261026'
GEOM=ROOT/'logs/ReferenceGeometry/reference_geometry_round2_20261017'
PILOT=ROOT/'logs/ReferenceGeometry/reference_geometry_pilot_20261017'
WRIST=ROOT/'paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005'
CHAIR=ROOT/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908'
ROBOT=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def source(p,expected=None):
    p=Path(p);s=sha(p)
    if expected:assert s==expected,(str(p),s,expected)
    return dict(path=str(p),sha256=s,size_bytes=p.stat().st_size)
cfg=read(R0/'R0/material_lighting.json')
tree=ast.parse((R0/'sources/previous_r0_helpers.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'colors','setup','bounds','camera_for'}],type_ignores=[]),'approved_helpers','exec'))
# Native execution ground/sky/light exactly follow archived export settings.
native=inspect.getsource(VisualScene).replace('class VisualScene:', 'class NativeScene:')
native=native.replace("ET.SubElement(asset, 'material', name='neutral_ground', rgba='.95 .95 .95 1', emission='.5')", "ET.SubElement(asset,'texture',name='ground',type='2d',builtin='checker',rgb1='.90 .92 .94',rgb2='.96 .97 .98',width='256',height='256')\n        ET.SubElement(asset,'material',name='neutral_ground',texture='ground',texrepeat='80 80',reflectance='0',specular='0',shininess='0')")
native=native.replace("rgb1='.98 .98 .98', rgb2='.98 .98 .98'","rgb1='.97 .98 1', rgb2='.97 .98 1'")
native=native.replace("diffuse='.55 .55 .55'","diffuse='.5 .5 .5'").replace("diffuse='.45 .45 .45'","diffuse='.5 .5 .5'")
exec(compile(native,'native_scene_display_only','exec'))
def newscene(robot,obj,count,preset,w=1920,h=1280):
    scene=(NativeScene if preset in ('g1_native_execution','human_native_execution') else VisualScene)(Path(robot),Path(obj),count,w,h)
    scene.model.vis.global_.ipd=0;scene.preset=preset;scene.width=w;scene.height=h
    scene.mapping_records=[]
    return scene
def styling(scene,center):
    if scene.preset.startswith('cplus'):
        # approved setup assumes two agents; adapt role boundary for single source.
        scene.mapping_records=setup(scene,center)
        for r,(agent,node,body) in zip(scene.mapping_records,scene.mapping):
            if agent==scene.count:
                rgba=colors(cfg['object_hex']);scene.model.geom_rgba[r['geom_id']]=rgba;r['role']='object';r['rgba']=rgba
    else:
        if scene.preset in ('g1_native_execution','human_native_execution'):scene.model.light_pos[0]=np.asarray(center)+[0,0,5]
        for agent,node,body in scene.mapping:
            bid=mujoco.mj_name2id(scene.model,mujoco.mjtObj.mjOBJ_BODY,body);gid=int(scene.model.body_geomadr[bid])
            link=node;parents=scene.urdfs[agent].scene.graph.transforms.parents
            while link not in scene.urdfs[agent].link_map and link in parents:link=parents[link]
            scene.mapping_records.append(dict(role=agent if agent<scene.count else 'object',visual_node=node,link=link,body=body,geom_id=gid,rgba=scene.model.geom_rgba[gid].tolist()))
    return scene.mapping_records
def fit(scene,q,o,frames,az=135,el=-22):
    pts=[]
    for t in frames:scene.set_pose(q[t],o[t]);pts.append(bounds(scene))
    return camera_for(np.concatenate(pts),scene.width,scene.height,az,el)
def render(scene,cam,path,depth=False):
    c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE;c.lookat[:]=cam['lookat'];c.distance=cam['distance'];c.azimuth=cam['azimuth'];c.elevation=cam['elevation']
    scene.renderer.update_scene(scene.data,camera=c)
    if scene.preset.startswith('cplus'):
        for g in scene.renderer.scene.geoms[:scene.renderer.scene.ngeom]:
            g.specular=cfg['geom_specular'];g.shininess=cfg['geom_shininess'];g.reflectance=0
            if g.type==mujoco.mjtGeom.mjGEOM_PLANE:g.specular=0;g.shininess=0
    gl=scene.renderer.scene.camera[0];f=np.array(gl.forward);f/=np.linalg.norm(f);up=np.array(gl.up);up/=np.linalg.norm(up);right=np.cross(f,up)
    E=np.eye(4);E[:3,:3]=np.stack([right,-up,f]);E[:3,3]=-E[:3,:3]@gl.pos
    w,h=scene.width,scene.height;fy=h*gl.frustum_near/(gl.frustum_top-gl.frustum_bottom);K=np.array([[fy,0,(w-1)/2],[0,fy,(h-1)/2],[0,0,1]])
    pixels=scene.renderer.render();assert pixels.shape==(h,w,3) and pixels.std()>10
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(pixels).save(path)
    camera=dict(settings=cam,resolution=[w,h],K=K.tolist(),world_to_camera=E.tolist(),near=float(gl.frustum_near),far=float(gl.frustum_far),
      projection='perspective',fovy=45,pixel_convention='top-left pixel center (0,0), u right v down; camera x right y down z forward',display_transform=np.eye(4).tolist(),crop='none',ipd=0)
    if depth:
        scene.renderer.enable_depth_rendering();d=scene.renderer.render().copy();scene.renderer.disable_depth_rendering()
        scene.renderer.enable_segmentation_rendering();seg=scene.renderer.render().copy();scene.renderer.disable_segmentation_rendering()
        return camera,d,seg
    return camera
def saveasset(scene,path,camera,extra):
    rec=dict(asset_id=Path(path).stem,file=str(Path(path).relative_to(OUT)),sha256=sha(path),material_preset=scene.preset,camera=camera,
      role_geom_mapping=scene.mapping_records,mesh_sources=scene.mesh_records,no_physics_steps=True,no_policy=True,labels_ui=False,**extra)
    write(Path(path).with_suffix('.json'),rec);return rec
def sheet(stage,records):
    w=480;h=320;cols=4;rows=(len(records)+cols-1)//cols
    canvas=Image.new('RGB',(cols*w,rows*(h+28)),'white');d=ImageDraw.Draw(canvas)
    for i,r in enumerate(records):
        x=(i%cols)*w;y=(i//cols)*(h+28);im=Image.open(OUT/r['file']);im.thumbnail((w,h))
        canvas.paste(im,(x,y));d.text((x+5,y+h+3),r['asset_id'],fill='black')
    canvas.save(OUT/'diagnostics'/f'{stage}_contact_sheet.png')
def finish(stage,records,**extra):
    write(OUT/f'{stage}_manifest.json',dict(stage=stage,status='complete',records=records,**extra));sheet(stage,records)
    print(stage,'complete:',len(records),'assets',flush=True)
