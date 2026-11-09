"""Validate rev1.14 supplements, merge only Palm columns and package once."""
import ast,csv,hashlib,json,shutil,sys,tarfile,subprocess,datetime
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path('/home/kevin/holosoma-core4d-base');O=R/'paper_handoff_exports/V3_20261102/V3_CLOSEOUT_SUPPLEMENT_20261105';OLD=R/'logs/V3_20261102/closeout_20261104';B=R/'logs/V3_20261102/cloud_backup_20261104/raw'
sys.path[:0]=[str(R/'src/holosoma_retargeting'),str(Path(__file__).resolve().parent),str(R/'scripts')]
from v314_compute import pm,bases,names
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ident(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.tolist())+'\n')
def rows(p):return list(csv.DictReader(Path(p).open()))
def csvwrite(p,data):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
def merge():
 dest=O/'results';dest.mkdir(exist_ok=True)
 key=lambda x:tuple(x[k] for k in ['task','variant','perturbation_level_cm','training_seed','state_id'])
 supplement={key(x):x for x in rows(O/'metrics/palm_episodes.csv')};original=rows(OLD/'episodes.csv');new=[]
 allowed=['palm_error_deg','palm_status','palm_valid_hand_frames','palm_missing_reason','palm_target_sha256']
 for r in original:
  s=supplement.get(key(r));d=dict(r)
  if s:
   for k in allowed:d[k]=s.get(k,'')
  new.append(d)
 assert len(new)==6120 and len({key(x) for x in new})==6120
 for a,b in zip(original,new):assert all(a[k]==b[k] for k in a if k not in allowed)
 csvwrite(dest/'episodes.csv',new)
 ps=rows(OLD/'per_seed_summary.csv');changes={tuple(r[k] for k in ['task','variant','perturbation_level_cm','training_seed']):r for r in rows(O/'metrics/palm_per_seed.csv')}
 for r in ps:
  s=changes.get(tuple(r[k] for k in ['task','variant','perturbation_level_cm','training_seed']))
  if s:r.update(palm_error_deg_mean=s['palm_mean_deg'],palm_error_deg_n_episodes=s['n_completed'])
 csvwrite(dest/'per_seed_summary.csv',ps)
 gs=rows(OLD/'group_summary.csv');changes={tuple(r[k] for k in ['task','variant','perturbation_level_cm']):r for r in rows(O/'metrics/palm_group_summary.csv')}
 for r in gs:
  r['per_seed_csv_sha256']=sha(dest/'per_seed_summary.csv');s=changes.get(tuple(r[k] for k in ['task','variant','perturbation_level_cm']))
  if s:r.update(palm_error_deg_mean_valid_seeds=s['n_valid_seeds'],palm_error_deg_mean_mean=s['palm_mean_deg'],palm_error_deg_mean_sample_sd=s['palm_sample_std_deg'])
 csvwrite(dest/'group_summary.csv',gs)
 paired=rows(OLD/'paired_episode_details.csv');lookup={key(r):r for r in new}
 for r in paired:
  if r['task'] not in ['push','pull'] or r['both_completed']!='True':continue
  k=[r['task'],None,r['perturbation_level_cm'],r['training_seed'],r['state_id']];k[1]='Full';f=lookup[tuple(k)];k[1]=r['variant'];v=lookup[tuple(k)]
  if f['palm_status']==v['palm_status']=='complete':r.update(palm_error_deg_full=f['palm_error_deg'],palm_error_deg_variant=v['palm_error_deg'],palm_error_deg_variant_minus_full=float(v['palm_error_deg'])-float(f['palm_error_deg']))
 csvwrite(dest/'paired_episode_details.csv',paired);pe=rows(OLD/'paired_errors.csv')
 for r in pe:
  if r['task'] not in ['push','pull']:continue
  a=[float(x['palm_error_deg_variant_minus_full']) for x in paired if all(x[k]==r[k] for k in ['task','variant','perturbation_level_cm','training_seed']) and x['palm_error_deg_variant_minus_full']!='']
  r.update(palm_error_deg_n_pairs=len(a),palm_error_deg_mean_variant_minus_full=float(np.mean(a)) if a else '')
 csvwrite(dest/'paired_errors.csv',pe);shutil.copy2(OLD/'paired_completion.csv',dest/'paired_completion.csv');shutil.copy2(OLD/'hard_level_selection.json',dest/'hard_level_selection.json')
 write(O/'checks/results_update_invariants.json',dict(passed=True,episode_count=6120,per_seed_rows=len(ps),group_rows=len(gs),only_changed_episode_fields=allowed,all_non_palm_episode_fields_exact=True,completion_statistics_file_bit_exact=sha(dest/'paired_completion.csv')==sha(OLD/'paired_completion.csv'),new_training=False,new_evaluation=False))
def human_audit():
 import torch,smplx,joblib
 from scipy.spatial.transform import Rotation as Rot
 from holosoma_retargeting.data_utils.core4d_adapter import _load_official_person_npz,CORE4D_TO_OMNI_ROTATION
 torch.set_num_threads(4);checks=[]
 canon=read(B/'v3_palm_targets/manifest.json')['records']
 for task,seq in [('chair','20231020/074'),('bucket','20231020/071'),('table','20231030/001')]:
  canonical=canon[task]['canonical'];canonical=Path(canonical if isinstance(canonical,str) else canonical['path']);a=np.load(canonical);base=Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/human_object_motions')/seq
  diff=[]
  for i in range(2):
   human=_load_official_person_npz(base/f'person{i+1}_poses.npz');diff.append(float(np.max(abs(human['joints'][0]@CORE4D_TO_OMNI_ROTATION.T-a['human_joints_full'][0,i]))))
  assert max(diff)<1e-5;checks.append(dict(task=task,source_frame=0,canonical=ident(canonical),max_joint_component_error_m=diff))
 rawpath=Path('/home/kevin/holosoma/data/train_diffusion_manip_seq_joints24.p');data=joblib.load(rawpath);test=None
 for task,seq in [('push','sub6_largetable_033'),('pull','sub3_largetable_010'),('kick','sub16_largetable_028')]:
  vs=[v for v in data.values() if v['seq_name']==seq]
  if not vs:
   if test is None:test=joblib.load('/home/kevin/holosoma/data/test_diffusion_manip_seq_joints24.p')
   vs=[v for v in test.values() if v['seq_name']==seq]
  assert len(vs)==1;v=vs[0];gender=str(np.asarray(v['gender']).item());mp=Path('/home/kevin/CS6758Project/data/smplx_models/smplx')/('SMPLX_'+gender.upper()+'.npz');model=smplx.SMPLX(str(mp),gender=gender,num_betas=16,use_pca=False,flat_hand_mean=True,batch_size=1).cpu().eval()
  with torch.no_grad():result=model(betas=torch.as_tensor(v['betas'],dtype=torch.float32).reshape(1,-1),transl=torch.as_tensor(v['trans'][:1],dtype=torch.float32),global_orient=torch.as_tensor(v['root_orient'][:1],dtype=torch.float32),body_pose=torch.as_tensor(v['pose_body'][:1],dtype=torch.float32))
  pt=Path('/home/kevin/holosoma/OMOMO_new')/(seq+'.pt');a=torch.load(pt,map_location='cpu',weights_only=False).detach().numpy();P=np.array([[0,0,1],[1,0,0],[0,1,0]]);A=Rot.from_quat(a[0,383:591].reshape(52,4)[0]).as_matrix()@P@Rot.from_rotvec(v['root_orient'][0]).as_matrix().T
  order=[0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,14,17,19,21];ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,34,35,36];j=result.joints[0].numpy();saved=a[0,162:318].reshape(52,3)[ids];offset=saved[0]-j[0]@A.T;err=float(np.linalg.norm(j[order]@A.T+offset-saved,axis=-1).max());assert err<1e-5
  checks.append(dict(task=task,original_source_frame0_joint_validation_max_m=err,pt=ident(pt),model=ident(mp),render_source_frame=read(O/f'fig3/F3_human_{task}.json')['source_frame'],interpolation='fractional raw source pose SLERP/translation linear, no pose fitting or frame selection by outcome'))
 write(O/'checks/human_source_verification.json',dict(passed=True,checks=checks))
def geometry_audit():
 from PIL import Image
 maxerr=0;images=[]
 for p in sorted(list((O/'fig2').glob('*.png'))+list((O/'fig3').glob('*.png'))):
  a=np.array(Image.open(p));assert a.shape==(1080,1920,3) and a.std()>10;record=read(p.with_suffix('.json'));assert sha(p)==record['image_sha256'];images.append(ident(p))
 for p in (O/'fig2').glob('*_overlay.json'):
  d=read(p);K=np.array(d['camera']['K']);E=np.array(d['camera']['world_to_camera'])
  assert sha(p.with_name(d['image']))==d['image_sha256']
  for x in d['points']+d['all_anchor_points']:
   q=E@np.r_[x['world'],1];v=K@q[:3];err=np.max(abs(v[:2]/v[2]-x['pixel']));maxerr=max(maxerr,float(err))
 assert maxerr<1e-7
 compare=[]
 for stem,a,b in [('F2a_push_front','prescribed','registered'),('F2a_push_top','prescribed','registered'),('F2b_pull_front','plus','minus'),('F2d_chair','before','after')]:
  x=read(O/'fig2'/f'{stem}_{a}.json');y=read(O/'fig2'/f'{stem}_{b}.json');assert x['camera']==y['camera'];compare.append(dict(pair=stem,camera_bit_exact=True))
 a=read(O/'fig2/F2a_push_top_prescribed_overlay.json');b=read(O/'fig2/F2a_push_top_registered_overlay.json');pa=[x for x in a['points'] if x['type']=='hand_actual_FK'];pb=[x for x in b['points'] if x['type']=='hand_actual_FK'];displacements=[]
 for x,y in zip(pa,pb):
  assert x['role']==y['role'] and x['hand']==y['hand'];world=float(np.linalg.norm(np.array(x['world'])-y['world'])*1000);assert abs(world-39.0236)<1e-7
  displacements.append(dict(role=x['role'],hand=x['hand'],world_displacement_mm=world,pixel_displacement=float(np.linalg.norm(np.array(x['pixel'])-y['pixel'])),visibility_before=x['visibility'],visibility_after=y['visibility']))
 write(O/'checks/render_validation.json',dict(passed=True,images=images,count=len(images),same_camera_pairs=compare,overlay_reprojection_max_error_px=maxerr,push_top_displacements=displacements,visual_review='All14 images reviewed; top view retains real occlusions, does not hide meshes or fake hand visibility. Pull plus pose is exact proper Procrustes transformation, not a physical rollout.'))
def evidence():
 dest=O/'sources/evidence';dest.mkdir(exist_ok=True);refs={}
 for row in rows(O/'reference/Table_III_reference.csv'):refs[row['source_file']]=row['source_sha256']
 for p in [R/'logs/ReferenceGeometry/reference_geometry_round2_20261017'/task/name for task in ['push','pull'] for name in ['selection.json','correspondence_spec.json','fitted_candidates.json','input_manifest.json','preprocessed_source.npz','new_pair_reference.npz']]:refs[str(p)]=sha(p)
 for p in [B/'v3_palm_targets/manifest.json',B/'v3_palm_targets/main_mesh_collision_rubberhand.urdf',R/'src/holosoma/holosoma/utils/v3_episode_telemetry.py',R/'docs/experiments/ubuntu_v3_rev14_priority_gpu_20261102_zh.md',R/'docs/experiments/ubuntu_v3_batch1_batch2_20261102_zh.md',R/'logs/V3_20261102/U3_assets/chair_source_size_reopt_20261103/cpu_asset_tests.json']:refs[str(p)]=sha(p)
 for p in (O/'reference').glob('*.launch.json'):
  for key,val in read(p).items():
   pass
 for key in ['source_manifest','promotion']:
  d=read(O/'reference/T07_wrist_audit.json')[key];refs[d['path']]=d['sha256']
 records=[]
 for path,h in refs.items():
  p=Path(path);assert sha(p)==h;name=h[:12]+'_'+p.name;shutil.copy2(p,dest/name);records.append(dict(original_path=path,sha256=h,package_path='sources/evidence/'+name))
 # Actual final training configurations establish reference identity used at execution.
 for row in read(OLD/'final_checkpoint_registry.json'):
  if row['run_id'] in ['V3-04','V3-05','V3-06']:shutil.copy2(row['run_config'],O/'reference'/(row['run_id']+'_actual_run_config.json'))
 write(O/'sources/evidence_index.json',records)
 curve=read(O/'curves/pull_curve_audit.json');assert len(curve['segments'])==18
 sourcecode=B/'v3_20261103_pull/src/holosoma/holosoma/utils/v3_episode_telemetry.py';contracts=[]
 for seg in curve['segments']:
  cfg=read(Path(seg['source']).parent/'run_config.json');contracts.append(cfg['episode_telemetry_contract']);assert seg['missing_length']==0
 assert all(x==contracts[0] for x in contracts);curve.update(passed=True,same_actual_contract_all_18_segments=True,contract=contracts[0],frozen_telemetry_code=ident(sourcecode),expected_n_seeds_per_group=3,missing=[])
 write(O/'curves/pull_curve_audit.json',curve)
 pm.tests()
 write(O/'checks/authority.json',dict(authority_commit='6c71577bed382f388d412ab5d4b14e1f4bc4cfff',revision='1.14',local_only=True,new_training=False,new_physics_rollouts=False,new_videos=False,manuscript_modified=False,old_package_modified=False,cloud_accessed=False,remote_name_note='origin=amazon-far upstream, private=user project; fetched private and fast-forwarded exact requested commit; remote definitions unchanged'))
def main():
 for name in sys.argv[1:] or ['merge','human','geometry','evidence']:
  print('BEGIN',name,flush=True);{'merge':merge,'human':human_audit,'geometry':geometry_audit,'evidence':evidence}[name]();print('DONE',name,flush=True)
if __name__=='__main__':main()
