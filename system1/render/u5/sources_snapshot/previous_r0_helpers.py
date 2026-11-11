"""R0 only: fixed stored states, CPU OSMesa, no policy/physics stepping."""
import os,sys,json,hashlib,shutil,subprocess,importlib.metadata,platform
from pathlib import Path
ROOT=Path('/home/kevin/holosoma-core4d-base')
sys.path.insert(0,str(ROOT/'scripts'))
assert os.environ['MUJOCO_GL']=='osmesa' and os.environ['LIBGL_ALWAYS_SOFTWARE']=='1'
import numpy as np
import mujoco
from OpenGL import GL
from PIL import Image,ImageDraw,ImageFont
from render_wrist_reference_pairs import VisualScene
from export_recorded_demo_media import configured_joint_names
OUT=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_v4'
OLD=ROOT/'paper_handoff_exports/FIGURES_R2_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(Path(p).read_text())
OUT.mkdir(exist_ok=False)
for d in ['R0','sources','diagnostics']:(OUT/d).mkdir()
for p in (OLD/'sources').iterdir():
    if p.is_file():shutil.copy2(p,OUT/'sources'/p.name)
shutil.copy2(OLD/'input_inventory.json',OUT/'input_inventory.json')
shutil.copy2(OLD/'diagnostics/cpu_preflight.json',OUT/'diagnostics/previous_failed_preflight.json')
shutil.copy2(__file__,OUT/'sources/render_r0_cpu.py')
inv=read(OLD/'input_inventory.json')
for r in inv['inputs']:
    if r['exists']:assert sha(r['path'])==r['sha256'],r['path']
cfg=dict(role_hex=['#AAC0DE','#E8C8A0'],object_hex='#B5B5B5',alpha=1,
    background_rgb=[.98]*3,ground_rgba=[.96,.96,.96,1],ground_emission=.3,
    headlight_ambient=[.28]*3,headlight_diffuse=[.5]*3,headlight_specular=[.04]*3,
    overhead_diffuse=[.3]*3,overhead_specular=[.02]*3,overhead_offset=[-1,-1,4],directional_light=True,
    overhead_dir=[.2,.2,-1],shadows=True,shadowsize=2048,offsamples=4,
    geom_specular=.08,geom_shininess=.12,geom_reflectance=0,exposure='MuJoCo default; no postprocess')
write(OUT/'R0/material_lighting.json',cfg)
robotpath=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
r2=ROOT/'paper_handoff_exports/EXPERIMENTS_R2_20261026/U7_media'
pm=read(r2/'rollouts/Push-registered_seed721/manifest.json')
sel=inv['selections']
with np.load(sel['chair']['source']) as z:
    chairrobot=z['robot_qpos'][117].copy()
with np.load(sel['push']['source']) as z:
    pushrobot=np.concatenate([z['root_pos'],z['root_quat_xyzw'][...,[3,0,1,2]],z['dof_pos']],axis=-1)
    pushobj=np.concatenate([z['object_pos_w'],z['object_quat_xyzw'][...,[3,0,1,2]]],axis=-1)
def colors(hexcode):return [int(hexcode[i:i+2],16)/255 for i in (1,3,5)]+[1]
def setup(scene,center):
    m=scene.model;m.vis.global_.ipd=0
    m.vis.headlight.diffuse[:]=cfg['headlight_diffuse'];m.vis.headlight.ambient[:]=cfg['headlight_ambient'];m.vis.headlight.specular[:]=cfg['headlight_specular']
    m.light_pos[0]=np.asarray(center)+cfg['overhead_offset'];m.light_dir[0]=cfg['overhead_dir'];m.light_type[0]=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    m.light_diffuse[0]=cfg['overhead_diffuse'];m.light_specular[0]=cfg['overhead_specular']
    m.mat_rgba[0]=cfg['ground_rgba'];m.mat_emission[0]=cfg['ground_emission'];m.mat_specular[0]=0;m.mat_shininess[0]=0
    mapping=[]
    for agent,node,body in scene.mapping:
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,body);gid=int(m.body_geomadr[bid])
        rgba=colors(cfg['role_hex'][agent] if agent<2 else cfg['object_hex'])
        m.geom_rgba[gid]=rgba
        graph=scene.urdfs[agent].scene.graph.transforms.parents
        link=node
        while link not in scene.urdfs[agent].link_map and link in graph:link=graph[link]
        mapping.append(dict(role=agent if agent<2 else 'object',visual_node=node,link=link,body=body,geom_id=gid,rgba=rgba))
    return mapping
def bounds(scene):
    pts=[]
    for agent,node,body in scene.mapping:
        tf=scene.bases[agent]@scene.urdfs[agent].scene.graph.get(node)[0]
        geom=scene.urdfs[agent].scene.graph.get(node)[1]
        b=scene.urdfs[agent].scene.geometry[geom].bounds
        corners=np.array([[x,y,z] for x in b[:,0] for y in b[:,1] for z in b[:,2]])
        pts.extend(corners@tf[:3,:3].T+tf[:3,3])
    return np.array(pts)
def camera_for(points,width,height,azimuth,elevation):
    # Fit all stored candidate mesh bounds with 7% border; no pose manipulation.
    center=(points.min(0)+points.max(0))/2
    az,el=np.deg2rad([azimuth,elevation]);back=np.array([-np.cos(az)*np.cos(el),-np.sin(az)*np.cos(el),-np.sin(el)])
    right=np.cross(-back,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,-back)
    q=points-center;tan=np.tan(np.deg2rad(22.5));factor=.86
    distance=max(np.max(q@back+np.abs(q@right)/(tan*width/height*factor)),np.max(q@back+np.abs(q@up)/(tan*factor)))
    return dict(lookat=center.tolist(),distance=float(distance),azimuth=azimuth,elevation=elevation)
def render(scene,cam,dest,w,h):
    c=mujoco.MjvCamera();c.type=mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:]=cam['lookat'];c.distance=cam['distance'];c.azimuth=cam['azimuth'];c.elevation=cam['elevation']
    scene.renderer.update_scene(scene.data,camera=c)
    for g in scene.renderer.scene.geoms[:scene.renderer.scene.ngeom]:
        g.specular=cfg['geom_specular'];g.shininess=cfg['geom_shininess'];g.reflectance=0
        if g.type==mujoco.mjtGeom.mjGEOM_PLANE:g.specular=0;g.shininess=0
    gl=scene.renderer.scene.camera[0]
    f=np.array(gl.forward);f/=np.linalg.norm(f);up=np.array(gl.up);up/=np.linalg.norm(up);right=np.cross(f,up)
    ext=np.eye(4);ext[:3,:3]=np.stack([right,-up,f]);ext[:3,3]=-ext[:3,:3]@np.array(gl.pos)
    near=float(gl.frustum_near);far=float(gl.frustum_far);bottom=float(gl.frustum_bottom);top=float(gl.frustum_top)
    fy=h*near/(top-bottom);K=np.array([[fy,0,(w-1)/2],[0,fy,(h-1)/2],[0,0,1]])
    img=scene.renderer.render();assert img.shape==(h,w,3) and img.std()>10
    Image.fromarray(img).save(dest)
    return dict(settings=cam,resolution=[w,h],projection='perspective',vertical_fov_degrees=45,
       K=K.tolist(),world_to_camera=ext.tolist(),near=near,far=far,
       camera_axes='x right, y down, z forward',pixel_convention='top left center (0,0), u right, v down',
       frustum=dict(bottom=bottom,top=top,center=float(gl.frustum_center)),
       display_transform=np.eye(4).tolist(),crop='none; native raster',ipd=0)
chairpath=r2/'shared_scene/shared_mean_visual_only.urdf'
# Actual pose smoke test must precede high-resolution rendering.
sm=VisualScene(robotpath,chairpath,2,480,320)
sm.set_pose(chairrobot,np.array([0,0,0,1,0,0,0]))
info={k:GL.glGetString(v).decode() for k,v in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
assert 'llvmpipe' in info['renderer'].lower(),info
pts=bounds(sm);cam=camera_for(pts,480,320,135,-30);setup(sm,cam['lookat'])
render(sm,cam,OUT/'diagnostics/cpu_pose_smoke.png',480,320);sm.renderer.close()
lib=ROOT/'.local_render_runtime/osmesa/extracted/usr/lib/x86_64-linux-gnu/libOSMesa.so.8'
info.update(backend='osmesa',software_verified=True,pose_smoke_passed=True,library=str(lib.resolve()),library_sha256=sha(lib),
    python=sys.version,versions={p:importlib.metadata.version(p) for p in ['mujoco','PyOpenGL','numpy','yourdfpy','trimesh','Pillow']},
    environment={k:os.environ.get(k) for k in ['MUJOCO_GL','PYOPENGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH']})
write(OUT/'diagnostics/cpu_preflight.json',info)
print('CPU pose smoke passed:',info['renderer'],flush=True)
records=[]
for task in ['chair','push']:
    w,h=1920,1280;objpath=chairpath if task=='chair' else Path(pm['object_urdf_source'])
    scene=VisualScene(robotpath,objpath,2,w,h)
    assert scene.urdfs[0].actuated_joint_names==configured_joint_names()
    if task=='chair':
        q=chairrobot;o=np.array([0,0,0,1,0,0,0]);scene.set_pose(q,o);pts=bounds(scene)
        # R6 head-to-knee framing, preserving original scene geometry.
        pts=pts[pts[:,2]>=.42]
        cam=camera_for(pts,w,h,315,-30);fitframes=[117]
    else:
        fitframes=[155];cloud=[]
        for t in fitframes:scene.set_pose(pushrobot[t],pushobj[t]);cloud.append(bounds(scene))
        cam=camera_for(np.concatenate(cloud),w,h,135,-22);q=pushrobot[155];o=pushobj[155]
    mapping=setup(scene,cam['lookat']);scene.set_pose(q,o)
    dest=OUT/'R0'/f'{task}_material.png';camera=render(scene,cam,dest,w,h)
    record=dict(asset_id='R0_'+task,file=str(dest.relative_to(OUT)),sha256=sha(dest),used_by=['R0'],
        selection=sel[task],camera=camera,camera_fit_frames=fitframes,role_geom_mapping=mapping,
        mesh_sources=scene.mesh_records,object_urdf=str(objpath),object_urdf_sha256=sha(objpath),
        robot_urdf=str(robotpath),robot_urdf_sha256=sha(robotpath),labels_ui=False,
        material_lighting_sha256=sha(OUT/'R0/material_lighting.json'),no_policy=True,no_physics_steps=True)
    scene.renderer.close();records.append(record);print('Rendered',task,flush=True)
# R0 contact sheet only; previews are explicitly downsampled diagnostics, not new raw frames.
sheet=Image.new('RGB',(1960,1020),'white');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
for i,task in enumerate(['chair','push']):
    img=Image.open(OUT/'R0'/f'{task}_material.png');x=20+i*970
    draw.text((x,10),task.title()+' | R0 material sample',fill='#333333',font=font)
    sheet.paste(img.resize((940,627),Image.Resampling.LANCZOS),(x,45))
    draw.text((x,700),'Single-cell preview (320 x 213 px)',fill='#333333',font=font)
    sheet.paste(img.resize((320,213),Image.Resampling.LANCZOS),(x,740))
sheet.save(OUT/'R0/contact_sheet.png')
manifest=dict(instruction_commit='d6fe6812aa934248a867f34f44e0ed66bc311adc',approval_status='pending',R0_status='rendered_waiting_for_author',
    R1_R6_status='not_started',records=records,contact_sheet_sha256=sha(OUT/'R0/contact_sheet.png'),
    material_lighting_sha256=sha(OUT/'R0/material_lighting.json'),script_sha256=sha(__file__),command=[sys.executable,*sys.argv],
    software_backend=info,working_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
write(OUT/'R0/manifest.json',manifest);write(OUT/'manifest.json',manifest);write(OUT/'cameras.json',{r['asset_id']:r['camera'] for r in records})
for r in inv['inputs']:
    if r['exists']:assert sha(r['path'])==r['sha256']
print('R0 DONE; R1-R6 NOT STARTED; approval pending',flush=True)
