"""Local CPU rendering of fixed reference/human geometry only; rev1.14."""
import os,sys,types,ast,json,hashlib,shutil
from pathlib import Path
import numpy as np
R=Path('/home/kevin/holosoma-core4d-base');O=R/'paper_handoff_exports/V3_20261102/V3_CLOSEOUT_SUPPLEMENT_20261105'
sys.path[:0]=[str(R/'scripts'),str(R/'src/holosoma_retargeting')]
APP=R/'paper_handoff_exports/FIGURES_R2_20261027_approved'
helper=APP/'sources/figround_common.py';c=types.ModuleType('helpers');exec(compile(helper.read_text(),str(helper),'exec'),c.__dict__);c.OUT=O
from scipy.spatial.transform import Rotation as Rot,Slerp
from reference_geometry_math import map_robot
from holosoma_retargeting.dual_a1_layout import g1_sagittal_joint_map
from PIL import Image
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=lambda a:a.tolist())+'\n')
def ident(p):return dict(path=str(p),sha256=sha(p))
def capture(scene,cam,path,meta):
 camera,depth,seg=c.render(scene,cam,path,True)
 record=dict(asset_id=path.stem,image=path.name,image_sha256=sha(path),image_size=[1920,1080],camera=camera,camera_intrinsics=camera['K'],camera_extrinsics=camera['world_to_camera'],crop_transform=np.eye(3).tolist(),light_material_spec=scene.preset,render_command=[sys.executable,*sys.argv],code_sha256=sha(__file__),helper=ident(helper),no_policy=True,no_physics_steps=True,no_labels=True,mesh_sources=scene.mesh_records,**meta)
 write(path.with_suffix('.json'),record);return camera,depth,seg,record
def project(p,cam,depth,seg,scene,role=None):
 v=np.r_[p,1]@np.array(cam['world_to_camera']).T;uv=np.array(cam['K'])@v[:3];uv=uv[:2]/uv[2];inside=bool(v[2]>0 and 0<=uv[0]<1920 and 0<=uv[1]<1080)
 z=None;visibility='out-of-frame'
 if inside:
  x,y=np.minimum(np.floor(uv+.5).astype(int),[1919,1079]);z=float(depth[y,x]);visibility='visible' if v[2]<=z+.005 else 'occluded'
 return dict(world=np.asarray(p).tolist(),display=np.asarray(p).tolist(),camera=v[:3].tolist(),pixel=uv.tolist(),in_front=bool(v[2]>0),in_image=inside,projected_depth=float(v[2]),buffer_depth=z,visibility=visibility,depth_tolerance_m=.005)
def fig2():
 for task,frame in [('push',126),('pull',153)]:
  d=c.GEOM/task;src=np.load(d/'preprocessed_source.npz');pair=np.load(d/'new_pair_reference.npz');sel=read(d/'selection.json');inp=read(d/'input_manifest.json');roles=sel['role_parameters'];jm,sign=g1_sagittal_joint_map(src['joint_names'])
  qp=[];variants=['prescribed','registered'] if task=='push' else ['plus','minus'];maps=[]
  for variant in variants:
   transforms=[];robots=[]
   for i,role in enumerate(roles):
    Q=np.array(role['Q']);t=np.array(role['c']);parity=role['parity']
    if task=='push' and variant=='prescribed':t=np.array([-.4 if i==0 else .4,0.,0.])
    if task=='pull' and variant=='plus' and i==1:
     fit=read(R/'logs/V3_20261102/U2/fit_result.json')['fits']['1'];Q=np.array(fit['Q']);t=np.array(fit['c']);parity=1
    robots.append(map_robot(src['robot_qpos'],src['object_qpos'],Q,t,parity,jm,sign));transforms.append(dict(Q=Q,c=t,parity=parity))
   qp.append(np.stack(robots,axis=1));maps.append(transforms)
  assert np.max(abs(qp[1][:,:,:3]-pair['robot_qpos'][:,:,:3]))<1e-7
  assert np.max(abs(qp[1][:,:,7:]-pair['robot_qpos'][:,:,7:]))<1e-7
  scene=c.newscene(c.ROBOT,inp['target_object']['path'],2,'cplus_reference',1920,1080)
  old=read(APP/'geometry'/f'R4_{task}_{frame:04d}_pair.json');camold=old['camera']['settings']
  cameras=[('front',camold)]
  if task=='pull':
   points=[]
   for q in qp:scene.set_pose(q[frame],pair['object_qpos'][frame]);points.append(c.bounds(scene))
   cameras=[('front',c.camera_for(np.concatenate(points),1920,1080,315,-15))]
  if task=='push':
   # Fixed top camera near the right hand and target face; all variants use it.
   scene.set_pose(qp[1][frame],pair['object_qpos'][frame]);center=scene.hand_center(1,'right')
   cameras.append(('top',dict(lookat=(center+np.array([0.,0.,.15])).tolist(),distance=1.8,azimuth=55,elevation=-75)))
  can=read(d/'fitted_candidates.json');anchors=np.array(old['all_source_anchors']);weights=np.array(old['all_weights']);objects=pair['object_qpos']
  for view,cam in cameras:
   c.styling(scene,cam['lookat'])
   for vi,variant in enumerate(variants):
    scene.set_pose(qp[vi][frame],objects[frame]);path=O/'fig2'/f'F2{ "a" if task=="push" else "b"}_{task}_{view}_{variant}.png'
    camera,depth,seg,record=capture(scene,cam,path,dict(task=task,variant=variant,kind='reference_geometry',reference_frame=frame,fps=50,time_s=frame/50,source_files=[ident(d/n) for n in ['preprocessed_source.npz','new_pair_reference.npz','selection.json','correspondence_spec.json','fitted_candidates.json']],robot_urdf=ident(c.ROBOT),object_urdf=ident(inp['target_object']['path']),pose_note='Rigid O(3) diagnostic only, no retarget/ground correction; plus role is proper rotation, minus is physical mirrored joints'))
    To=scene.bases[-1];points=[];actuals=[]
    for rid,role in enumerate(roles):
     target=np.array(next(x for x in can if x['id']==role['id'])['target_anchors']);tr=maps[vi][rid]
     for sourcehand,choice in old['source_selected_anchor'].items():
      targethand=sourcehand if tr['parity']==1 else sourcehand.replace('left','TMP').replace('right','left').replace('TMP','right')
      face_map=next(x for x in can if x['id']==role['id'])['face_map'][choice['source_face_id']]
      H=scene.bases[rid]@scene.urdfs[rid].get_transform(targethand);meta=dict(role=rid,source_hand=sourcehand,hand=targethand,anchor_slice=choice['anchor_slice'],source_face_id=choice['source_face_id'],target_face_id=face_map['target_face_id'],corner_permutation=face_map['corner_permutation'])
      points.append(dict(type='hand_actual_FK',**meta,**project(H[:3,3],camera,depth,seg,scene,rid)))
      for ai in range(*choice['anchor_slice']):
       world=To[:3,:3]@target[ai]+To[:3,3]
       points.append(dict(type='selected_anchor' if ai==choice['anchor_index'] else 'face_corner',anchor_index=ai,weight=weights[ai],object_local=target[ai].tolist(),**meta,**project(world,camera,depth,seg,scene)))
      actuals.append(dict(role=rid,source_hand=sourcehand,target_hand=targethand,actual_world=H[:3,3].tolist(),actual_link_transform_world=H.tolist()))
    geo=dict(old);geo.update(camera=camera,points=points,actual_ideal=actuals,frozen_roles=maps[vi],image=path.name,image_sha256=sha(path),frame=frame,variant=variant,reflection=None,display_transform=np.eye(4).tolist(),crop_transform=np.eye(3).tolist(),depth_file=None,surface_samples=None,scope='Selected target face corners/anchors and actual FK; obsolete old depth/projection links removed')
    geo['all_anchor_points']=[dict(type='full_fit_anchor',role=rid,anchor_index=ai,weight=float(weights[ai]),object_local=anchor.tolist(),**project(To[:3,:3]@anchor+To[:3,3],camera,depth,seg,scene)) for rid,role in enumerate(roles) for ai,anchor in enumerate(np.array(next(x for x in can if x['id']==role['id'])['target_anchors']))]
    write(path.with_name(path.stem+'_overlay.json'),geo);print('RENDERED',path.name,flush=True)
  scene.renderer.close()
 # Chair must use exactly C01 before/after pair, old image size is not compliant.
 evidence=read(R/'paper_handoff_exports/EXPERIMENTS_R2_20261024_CPU_v2/checks/wrist_recomputed.json')['records'][0]
 before=Path(evidence['before']['path']);after=Path(evidence['after']['path']);qb=np.load(before);qa=np.load(after)
 assert sha(before)==evidence['before']['sha256'] and sha(after)==evidence['after']['sha256']
 frame=122;obj=Path(read(after.parent/'manifest.json')['object_urdf']);scene=c.newscene(c.ROBOT,obj,2,'g1_native_wrist',1920,1080)
 oldr=read(APP/'raw/R5_chair_122_before.json');cam=oldr['camera']['settings'];c.styling(scene,cam['lookat'])
 audit={}
 for v,z,source in [('before',qb,before),('after',qa,after)]:
  oldm=read(APP/'raw'/f'R5_chair_122_{v}.json');oldpath=Path(oldm['source']['path']);oz=np.load(oldpath)
  equal=np.array_equal(oz['robot_qpos'],z['robot_qpos']) and np.array_equal(oz['object_qpos'],z['object_qpos']);audit[v]=dict(old=ident(oldpath),C01=ident(source),arrays_bit_exact=equal,old_image_size=oldm['camera']['resolution'],rerender_reason='1920x1080 required; C01 actual source frozen')
  scene.set_pose(z['robot_qpos'][frame],z['object_qpos'][frame]);capture(scene,cam,O/'fig2'/f'F2d_chair_{v}.png',dict(kind='reference',task='chair',variant=v,reference_frame=frame,fps=30,time_s=frame/30,source=ident(source),robot_urdf=ident(c.ROBOT),object_urdf=ident(obj)));print('RENDERED chair',v,flush=True)
 scene.renderer.close();write(O/'fig2/chair_C01_reuse_audit.json',audit)
def human_scene_class(count):
 tree=ast.parse((APP/'sources/figround_r1.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HumanScene')
 code=ast.unparse(cls).replace('1280','1080').replace('i < 2',f'i < {count}').replace("'.40 .40 .40 1'","'.76 .80 .86 1'")
 ns=dict(c.__dict__);exec(compile(code,'R1_geometry_only_1080','exec'),ns);return ns['HumanScene']
def fig3():
 import trimesh,torch,smplx,joblib
 from holosoma_retargeting.data_utils.core4d_adapter import _load_official_person_npz,CORE4D_TO_OMNI_ROTATION
 torch.set_num_threads(4);manifest=[];topology=Path('/home/kevin/CS6758Project/data/smplx_models/smplx/SMPLX_NEUTRAL.npz');faces=np.load(topology)['f']
 for task,seq,objname,folder in [('chair','20231020/074','chair021','chair'),('bucket','20231020/071','bucket003','bucket'),('table','20231030/001','desk001','desk')]:
  base=Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real');d=base/'human_object_motions'/seq;people=[_load_official_person_npz(d/f'person{i}_poses.npz') for i in [1,2]];poses=np.load(d/'smooth_objposes.npy');rot=CORE4D_TO_OMNI_ROTATION
  objpath=base/'object_models'/folder/(objname+'_m.obj');om=trimesh.load(objpath,force='mesh',process=False);om.vertices=(om.vertices@poses[0,:3,:3].T+poses[0,:3,3])@rot.T
  meshes=[trimesh.Trimesh(vertices=p['vertices'][0]@rot.T,faces=faces,process=False) for p in people]+[om];scene=human_scene_class(2)(meshes);cam=c.camera_for(np.concatenate([m.vertices for m in meshes]),1920,1080,135,-22);scene.model.light_pos[0]=np.array(cam['lookat'])+[0,0,5]
  _,_,_,record=capture(scene,cam,O/'fig3'/f'F3_human_{task}.png',dict(kind='source_human',task=task,source_sequence=seq,source_frame=0,reference_frame=0,state_index=0,time_s=0,fps=15,source_to_display_rotation=rot,source_files=[ident(d/f'person{i}_poses.npz') for i in [1,2]]+[ident(d/'smooth_objposes.npy'),ident(objpath),ident(topology)],selection_rule='First execution reference time0; original source frame0, not later favorable contact frame',human_topology='Original 10475 vertices; SMPLX original face topology only, not body fitting'));scene.renderer.close();manifest.append(record);print('HUMAN',task,flush=True)
 rawpath=Path('/home/kevin/holosoma/data/train_diffusion_manip_seq_joints24.p');data=joblib.load(rawpath);test=None
 for task,seq,seconds in [('push','sub6_largetable_033',0.),('pull','sub3_largetable_010',.02),('kick','sub16_largetable_028',.02)]:
  matches=[v for v in data.values() if v['seq_name']==seq];sourcepath=rawpath
  if not matches:
   sourcepath=Path('/home/kevin/holosoma/data/test_diffusion_manip_seq_joints24.p');test=joblib.load(sourcepath) if test is None else test;matches=[v for v in test.values() if v['seq_name']==seq]
  assert len(matches)==1;v=matches[0];gender=str(np.asarray(v['gender']).item());mp=Path('/home/kevin/CS6758Project/data/smplx_models/smplx')/('SMPLX_'+gender.upper()+'.npz');model=smplx.SMPLX(str(mp),gender=gender,num_betas=16,use_pca=False,flat_hand_mean=True,batch_size=1).cpu().eval()
  if task=='kick':
   ev=Path('/home/kevin/holosoma-rubber-hand-largetable/logs/WholeBodyTracking/20260807_034545-rubberhand_kick_sub16_028_8000_seed42-eval/kick_7999_seed42_physics_eval_v1.npz');z=np.load(ev);seconds=float(z['motion_time_step'][2418])/50
  sf=seconds*30;lo=int(np.floor(sf));hi=int(np.ceil(sf));alpha=sf-lo
  def lerp(arr):return (1-alpha)*arr[lo]+alpha*arr[hi]
  def rotation(arr):
   if lo==hi:return np.asarray(arr[lo])
   return Slerp([lo,hi],Rot.from_rotvec(arr[[lo,hi]]))([sf]).as_rotvec()[0]
  pose=v['pose_body'].reshape(-1,21,3);bp=np.stack([rotation(pose[:,i]) for i in range(21)]).reshape(1,-1)
  with torch.no_grad():result=model(betas=torch.as_tensor(v['betas'],dtype=torch.float32).reshape(1,-1),transl=torch.as_tensor(lerp(v['trans']),dtype=torch.float32).reshape(1,3),global_orient=torch.as_tensor(rotation(v['root_orient']),dtype=torch.float32).reshape(1,3),body_pose=torch.as_tensor(bp,dtype=torch.float32))
  objpath=Path('/home/kevin/holosoma/data/captured_objects/largetable_cleaned_simplified.obj');om=trimesh.load(objpath,force='mesh',process=False);objR=v['obj_rot'][lo] if lo==hi else Slerp([lo,hi],Rot.from_matrix(v['obj_rot'][[lo,hi]]))([sf]).as_matrix()[0];om.vertices=lerp(v['obj_scale'])*(om.vertices@objR.T)+lerp(v['obj_trans']).ravel()
  meshes=[trimesh.Trimesh(vertices=result.vertices[0].numpy(),faces=model.faces,process=False),om]
  # Same raw world display as R1; choose camera using original source-to-canonical yaw.
  pt=Path('/home/kevin/holosoma/OMOMO_new')/(seq+'.pt');a=torch.load(pt,map_location='cpu',weights_only=False).detach().numpy();P=np.array([[0,0,1],[1,0,0],[0,1,0]]);A=Rot.from_quat(a[0,383:591].reshape(52,4)[0]).as_matrix()@P@Rot.from_rotvec(v['root_orient'][0]).as_matrix().T
  cam=c.camera_for(np.concatenate([m.vertices for m in meshes]),1920,1080,(135-np.degrees(np.arctan2(A[1,0],A[0,0])))%360,-22);scene=human_scene_class(1)(meshes);scene.model.light_pos[0]=np.array(cam['lookat'])+[0,0,5]
  params=O/'sources'/f'{task}_human_frame_parameters.npz';np.savez_compressed(params,body_pose=bp,global_orient=rotation(v['root_orient']),transl=lerp(v['trans']),betas=v['betas'],source_frame=sf)
  _,_,_,record=capture(scene,cam,O/'fig3'/f'F3_human_{task}.png',dict(kind='source_human',task=task,source_sequence=seq,source_frame=sf,source_frame_left=lo,source_frame_right=hi,source_interpolation_weight=alpha,reference_frame=0,state_index=0,time_s=seconds,fps=30,selection_rule='Exact original source time corresponding to first saved execution-reference row; no contact-based selection',source_files=[ident(sourcepath),ident(mp),ident(objpath),ident(pt),ident(params)],hand_pose_source='official_default, flat_hand_mean=True zero finger angles; not captured finger articulation',human_topology='SMPL-X forward with original sequence betas/gender/body parameters; fractional time uses SO3 SLERP + linear translations/scales',source_to_canonical_rotation=A))
  scene.renderer.close();manifest.append(record);print('HUMAN',task,sf,flush=True)
 write(O/'fig3/manifest.json',manifest)
if __name__=='__main__':
 for task in sys.argv[1:] or ['fig2','fig3']:{'fig2':fig2,'fig3':fig3}[task]()
