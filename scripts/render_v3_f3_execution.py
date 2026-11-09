"""U5 F3: exact saved physical states, approved native style, isolated EGL render."""
import argparse, hashlib, json, os, sys, types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert os.environ.get('MUJOCO_GL')=='egl'
 job=next(j for j in read(ROOT/'jobs.json') if j['task']==a.task)
 for file,expected in read(ROOT/'input_hashes.json').items():assert sha(ROOT/file)==expected,file
 out=a.output;out.mkdir(parents=True,exist_ok=False)
 # Reuse approved exact native rendering/styling/math. Only root relocation and
 # backend assertion differ; original archived module remains unmodified.
 helper=ROOT/'paper_handoff_exports/FIGURES_R2_20261027_approved/sources/figround_common.py'
 code=helper.read_text()
 oldroot="ROOT=Path('/home/kevin/holosoma-core4d-base')"
 oldbackend="assert os.environ.get('MUJOCO_GL')=='osmesa' and os.environ.get('CUDA_VISIBLE_DEVICES')==''"
 assert code.count(oldroot)==1 and code.count(oldbackend)==1
 code=code.replace(oldroot,'ROOT=Path('+repr(str(ROOT))+')').replace(oldbackend,"assert os.environ.get('MUJOCO_GL')=='egl'")
 module=types.ModuleType('v3_approved_render');module.__file__=str(helper);exec(compile(code,str(helper),'exec'),module.__dict__)
 from OpenGL import GL
 from PIL import Image
 with np.load(ROOT/job['trajectory'],allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
 n=job['horizon_frames'];frames=job['frames'];assert np.array_equal(arr['reference_frame'][:n],np.arange(n))
 robot=np.concatenate([arr['root_pos'],arr['root_quat_xyzw'][...,[3,0,1,2]],arr['dof_pos']],axis=-1)
 obj=np.concatenate([arr['object_pos_w'],arr['object_quat_xyzw'][...,[3,0,1,2]]],axis=-1)
 scene=module.newscene(ROOT/job['robot_urdf'],ROOT/job['object_urdf'],2,'g1_native_execution',1920,1080)
 try:
  # One fixed camera fit to the complete horizon, not independent crops per frame.
  camera=module.fit(scene,robot,obj,range(n),az=135,el=-22)
  module.styling(scene,camera['lookat']);rows=[];cam_id='F3_'+a.task+'_seed721_'+job['state_id']+'_fixed'
  for frac,index in zip([0,25,50,75,100],frames):
   scene.set_pose(robot[index],obj[index]);name=f'F3_exec_{a.task}_{frac:03d}';path=out/(name+'.png')
   cam,depth,seg=module.render(scene,camera,path,depth=True)
   gl={k:GL.glGetString(v).decode() for k,v in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
   assert 'NVIDIA' in gl['vendor'] and 'llvmpipe' not in gl['renderer'].lower(),gl
   row=dict(asset_id=name,kind='saved_physical_execution',task=a.task,configuration=job['configuration'],training_seed=721,state_id=job['state_id'],selection_rule=job['selection_rule'],state_index=index,reference_frame=int(arr['reference_frame'][index]),source_frame=None,source_frame_note='Physical policy execution, not source human frame',fps=50,time_s=index/50,horizon_frames=n,checkpoint_sha256=job['checkpoint_sha256'],robot_urdf_sha256=job['robot_urdf_sha256'],object_urdf_sha256=job['object_urdf_sha256'],input_paths=[job['trajectory'],job['run_manifest'],job['episodes_file']],input_sha256=[job['trajectory_sha256'],job['run_manifest_sha256'],job['episodes_sha256']],state_file_sha256=job['state_file_sha256'],camera_id=cam_id,camera_intrinsics=cam['K'],camera_extrinsics=cam['world_to_camera'],camera=cam,projection='perspective',image_size=[1920,1080],crop_transform=np.eye(3).tolist(),light_material_spec=dict(preset=scene.preset,native_URDF_colors=True,headlight_diffuse=scene.model.vis.headlight.diffuse.tolist(),headlight_ambient=scene.model.vis.headlight.ambient.tolist(),light_pos=scene.model.light_pos.tolist(),light_diffuse=scene.model.light_diffuse.tolist(),geom_rgba=scene.model.geom_rgba.tolist(),source_helper_sha256=sha(helper)),role_geom_mapping=scene.mapping_records,mesh_sources=scene.mesh_records,render_command=[sys.executable,*sys.argv],code_sha256=sha(__file__),image_sha256=sha(path),image=path.name,no_physics_steps=True,no_policy=True,labels_ui=False,backend=gl)
   write(path.with_suffix('.json'),row);rows.append(row);print('Rendered',name,flush=True)
  assert all(r['camera']==rows[0]['camera'] for r in rows)
  # Projection is in the final image's camera with no world/display transforms.
  E=np.asarray(cam['world_to_camera']);K=np.asarray(cam['K'])
  def project(points):
   c=points@E[:3,:3].T+E[:3,3];hom=c@K.T;uv=hom[:,:2]/hom[:,2:]
   front=c[:,2]>0;inside=front&(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
   visible=np.zeros(len(points),bool)
   for i in np.flatnonzero(inside):
    u,v=np.floor(uv[i]+.5).astype(int);u=min(u,1919);v=min(v,1079);visible[i]=c[i,2]<=depth[v,u]+.01
   return dict(world_xyz_m=points.tolist(),pixel_xy=uv.tolist(),camera_depth_m=c[:,2].tolist(),in_front=front.tolist(),inside_raster=inside.tolist(),unoccluded_against_final_frame_depth=visible.tolist())
  traj=dict(asset_id='F3_traj_'+a.task,task=a.task,state_id=job['state_id'],camera_id=cam_id,image=rows[-1]['image'],image_sha256=rows[-1]['image_sha256'],camera=cam,crop_transform=np.eye(3).tolist(),display_transform=np.eye(4).tolist(),point_order='reference_frame ascending, object actor/root origin (not center of mass)',reference_frame=list(range(n)),episode_step=arr['episode_step'][:n].tolist(),time_s=(np.arange(n)/50).tolist(),fps=50,reference=project(arr['reference_object_pos_w'][:n]),execution=project(arr['object_pos_w'][:n]),visibility_definition='In-front and viewport flags plus depth visibility against FINAL scene, tolerance 0.01m. Object origins may be occluded inside their own mesh. Not contact evidence.',trajectory_sha256=job['trajectory_sha256'],terminal_extra_state_excluded=True)
  write(out/('F3_traj_'+a.task+'.json'),traj)
  strip=Image.new('RGB',(5*480,270),'white')
  for i,row in enumerate(rows):strip.paste(Image.open(out/row['image']).resize((480,270)),(i*480,0))
  strip.save(out/'preview_strip.jpg',quality=94)
  write(out/'manifest.json',dict(status='rendered_pending_visual_review',job=job,records=rows,projection_file='F3_traj_'+a.task+'.json',backend_adaptation='Approved archived helper executed unchanged except relocated ROOT and EGL assertion; physics/training unchanged',executed_helper_sha256=hashlib.sha256(code.encode()).hexdigest(),archived_helper_sha256=sha(helper),versions={x:__import__('importlib.metadata',fromlist=['version']).version(x) for x in ['mujoco','PyOpenGL','numpy','yourdfpy','Pillow']}))
  paths=sorted(x for x in out.iterdir() if x.is_file());(out/'SHA256SUMS').write_text(''.join(sha(x)+'  '+x.name+'\n' for x in paths))
 finally:scene.renderer.close()
if __name__=='__main__':main()
