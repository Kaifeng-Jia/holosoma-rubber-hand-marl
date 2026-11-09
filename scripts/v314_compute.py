"""Revision 1.14 local-only evidence and offline metrics. No simulation/training."""
import ast,csv,hashlib,json,sys,types,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot,Slerp
R=Path('/home/kevin/holosoma-core4d-base')
O=R/'paper_handoff_exports/V3_20261102/V3_CLOSEOUT_SUPPLEMENT_20261105'
B=R/'logs/V3_20261102/cloud_backup_20261104/raw'
OLD=R/'logs/V3_20261102/closeout_20261104'
for folder in ['metrics','reference','curves','sources','fig2','fig3','checks']:(O/folder).mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(R/'scripts'),str(R/'src/holosoma_retargeting')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ident(p):return dict(path=str(p),sha256=sha(p))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False,default=lambda a:a.tolist())+'\n')
def csvwrite(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in rows for k in x)));w.writeheader();w.writerows(rows)
pm=types.ModuleType('palm_original')
code=(B/'v3_palm_postprocess.py').read_text().replace("B=Path('/root/autodl-tmp');DATA=B/'v3_palm_targets';OUT=B/'v3_palm_metrics_20261104';OUT.mkdir(exist_ok=True)",f"B=Path({str(B)!r});DATA=B/'v3_palm_targets';OUT=Path({str(O/'checks')!r})")
exec(compile(code,'archived_palm_fk','exec'),pm.__dict__)
base=np.load(B/'v3_palm_targets/chair_targets.npz');bases=base['bases'];names=base['joint_names']
def mat(wxyz):return Rot.from_quat(np.asarray(wxyz)[...,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(np.asarray(wxyz).shape[:-1]+(3,3))
def sample(rot,fps,t):
 out=np.empty((len(t),)+rot.shape[1:])
 for h in range(rot.shape[1]):out[:,h]=Slerp(np.arange(len(rot))/fps,Rot.from_matrix(rot[:,h]))(t).as_matrix()
 return out
def palm_targets():
 import torch
 from holosoma_retargeting.src.utils import load_intermimic_data,load_intermimic_wrist_quaternions,build_pt_wrist_palm_orientation_targets
 from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
 from holosoma_retargeting.dual_pull_reference import synthesize_dual_pull_reference
 reports={}
 for task,seq in [('push','sub6_largetable_033'),('pull','sub3_largetable_010')]:
  pt=Path('/home/kevin/holosoma/OMOMO_new')/(seq+'.pt');human,obj=load_intermimic_data(pt)
  target,cal=build_pt_wrist_palm_orientation_targets(human,load_intermimic_wrist_quaternions(pt),SMPLH_DEMO_JOINTS)
  geom=R/'logs/ReferenceGeometry/reference_geometry_round2_20261017'/task
  src=np.load(geom/'preprocessed_source.npz');n=len(src['robot_qpos']);t=np.arange(n)/50
  human50=sample(target,30,t)
  checks={};evidence=[ident(pt),ident(geom/'preprocessed_source.npz'),ident(R/'src/holosoma_retargeting/holosoma_retargeting/src/utils.py')]
  if task=='push':
   sel=read(geom/'selection.json');pair=np.load(geom/'new_pair_reference.npz');desired=np.empty((n,2,2,3,3))
   for role,d in enumerate(sel['role_parameters']):
    Q=np.asarray(d['Q']);assert d['parity']==1
    transform=mat(pair['object_qpos'][:,3:])@Q@mat(src['object_qpos'][:,3:]).swapaxes(-1,-2)
    desired[:,role]=transform[:,None]@human50
   checks['roles']='both copies preserve left/right hands; object-local proper Q applied in world frame'
   # The original PT object rotation and retarget object rotation are unchanged.
   human_obj=sample(mat(obj[:,:4])[:,None],30,t)[:,0]
   checks['human_vs_source_object_rotation_max_deg']=float(pm.angle(human_obj,mat(src['object_qpos'][:,3:])).max())
   assert checks['human_vs_source_object_rotation_max_deg']<.001
   evidence += [ident(geom/'selection.json'),ident(geom/'new_pair_reference.npz')]
  else:
   path=Path('/home/kevin/holosoma-rubber-hand-largetable/logs/WholeBodyTracking/20260807_212122-rubberhand_pull_sub3_010_8000_seed42-eval')
   ep=read(path/'pull_7999_seed42_physics_eval_v1.json')['attempt_details'][8];assert ep['attempt']==8
   rec=np.load(path/'pull_7999_seed42_physics_eval_v1.npz');sl=slice(ep['start'],ep['end']+1)
   raw=np.load(path/'pull_7999_seed42_attempt08_success_viser_qpos.npz')['qpos']
   assert np.array_equal(raw[:,7:36],rec['pre_dof_pos'][sl]);assert np.array_equal(raw[:,:3],rec['pre_root_pos'][sl])
   phases=rec['motion_time_step'][sl];assert np.array_equal(phases,np.arange(n)+1)
   human50=sample(target,30,phases/50)
   source_ref=R/'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/sub3_largetable_010_pull_a1_mj_fps50_w_obj.npz'
   ref=np.load(source_ref);assert np.max(abs(rec['ref_joint_pos'][sl]-ref['joint_pos'][phases,7:]))<1e-5
   # Reset alignment from recorded reference to original reference, then exact source-to-shared transport.
   reset_delta=Rot.from_quat(rec['ref_object_quat_xyzw'][sl]).as_matrix()@mat(ref['object_quat_w'][phases]).swapaxes(-1,-2)
   shared=mat(src['object_qpos'][:,3:]);rawobj=mat(src['raw_object_qpos'][:,3:]);transport=shared@rawobj.swapaxes(-1,-2)
   a=transport[:,None]@reset_delta[:,None]@human50
   S=shared@np.diag([1.,1.,-1.])@shared.swapaxes(-1,-2)
   desired=np.stack([a,S[:,None]@a[:,[1,0]]@np.diag([1.,-1.,1.])],axis=1)
   runtime=R/'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/rubber_hand_largetable_v1/pull/plan5_attempt08_mirrored_pair_runtime.npz';rt=np.load(runtime)
   regenerated=synthesize_dual_pull_reference(raw,fps=50,lateral_leg_offset_m=.4390236)
   assert np.max(abs(rt['agent_joint_pos'][:,0]-regenerated.robot_a_qpos[:,7:]))<1e-6
   assert np.max(abs(rt['agent_joint_pos'][:,1]-regenerated.robot_b_qpos[:,7:]))<1e-6
   root_index=list(rt['body_names']).index('pelvis')
   actual=pm.actual_rotations(rt['agent_joint_pos'],rt['agent_body_quat_w'][:,:,root_index,[1,2,3,0]],rt['joint_names'],bases)
   mirror_error=float(np.max(abs(actual[:,1]-S[:,None]@actual[:,0,[1,0]]@np.diag([1.,-1.,1.]))))
   # Fixed mesh-derived L/R palm bases are not exact mirrors. Bound the
   # discrepancy by that measured basis difference, plus URDF decimal rounding.
   basis_bound=float(np.linalg.norm(bases[1]-np.diag([1.,-1.,1.])@bases[0]@np.diag([1.,-1.,1.])))
   assert mirror_error<=basis_bound+1e-5,(mirror_error,basis_bound)
   checks['fixed_palm_basis_mirror_frobenius_bound']=basis_bound
   checks.update(roles='role0 source left/right; role1 swapped hands, Rmirror=S_world Rsource diag(1,-1,1)',mirror_palm_fk_max_matrix_error=mirror_error,source_qpos_equals_pre_step=True,source_reference_phase='recorded pre-step motion_time_step equals 1..316; saved pair rows0..315 map to these phases',first_human_source_frame=.6,source_time_offset_seconds=.02,recording_code=ident(Path('/home/kevin/holosoma-rubber-hand-largetable/src/holosoma/holosoma/agents/callbacks/recording.py')))
   evidence += [ident(path/'pull_7999_seed42_physics_eval_v1.json'),ident(path/'pull_7999_seed42_physics_eval_v1.npz'),ident(source_ref),ident(runtime)]
  assert np.allclose(np.linalg.det(desired),1,atol=1e-8)
  p=O/'metrics'/(task+'_fixed_human_targets.npz');np.savez_compressed(p,target_palm_world=desired,bases=bases,joint_names=names,fps=50,time_seconds=t)
  reports[task]=dict(status='verified',target=ident(p),calibration_p90_deg=cal,checks=checks,sources=evidence,interpolation='human30Hz SLERP at exact source time to50Hz; no fitted robot targets')
 write(O/'metrics/target_audit.json',reports)
def small_table():
 d=R/'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable';m=read(d/'source_manifest.json');p=Path(m['pair_reference']);assert sha(p)==m['pair_sha256']
 q=np.load(p)['robot_qpos'];checks=[]
 for i,f in enumerate(m['source_person_nominal']):
  z=np.load(f);assert np.array_equal(q[:,i],z['qpos'][:,:36]);checks.append(ident(f))
 targets=np.load(B/'v3_palm_targets/table_targets.npz');t=np.arange(len(q))/30
 target=np.stack([sample(targets['target_palm_world'][:,i],float(targets['fps']),t) for i in range(2)],axis=1)
 actual=pm.actual_rotations(q[:,:,7:],q[:,:,[4,5,6,3]],names,bases);err=pm.angle(target,actual)
 np.savez_compressed(O/'reference/T07_reference_palm_errors.npz',error_deg=err,desired=target,actual=actual)
 runtime=d/'core4d_pair_runtime_fps50.npz';rt=np.load(runtime);five=R/'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/core4d_pair_runtime_fps50.npz';assert sha(five)==sha(runtime)
 write(O/'reference/T07_wrist_audit.json',dict(status='verified_no_wrist_only_A1',reference=ident(p),source_manifest=ident(d/'source_manifest.json'),promotion=ident(d/'training_asset_manifest.json'),nominal_sources=checks,nominal_qpos_bit_exact=True,runtime=ident(runtime),T07_5kg_runtime=ident(five),reference_palm_mean_deg=float(err.mean()),reference_palm_max_deg=float(err.max()),per_role_hand_mean_deg=err.mean(0),per_role_hand_max_deg=err.max(0),fps=30,frames=len(q),target=ident(B/'v3_palm_targets/table_targets.npz'),method='Same fixed human palm basis/URDF FK and SO3 definition as prior execution metrics; all reference frames; no optimization performed'))
def curves():
 p=OLD/'training_scalars.csv';groups={};selected=[]
 for row in csv.DictReader(p.open()):
  rid=row['run_id']
  if rid in ['V3-19','V3-20','V3-21','V3-25','V3-26','V3-27'] or rid.startswith('V3-PullFull'):
   key=(rid,row['training_seed'],row['segment']);g=groups.setdefault(key,dict(run_id=rid,seed=row['training_seed'],segment=row['segment'],phase=row['phase'],rows=0,missing_length=0,tags=set(),min_iteration=10**9,max_iteration=-1,source=row['source_file'],source_sha256=row['source_sha256']))
   g['rows']+=1;g['missing_length']+=not bool(row['mean_episode_length_steps']);g['tags'].add(row['length_source_tag']);g['min_iteration']=min(g['min_iteration'],int(row['iteration']));g['max_iteration']=max(g['max_iteration'],int(row['iteration']));selected.append(row)
 for g in groups.values():g['tags']=sorted(g['tags']);assert sha(g['source'])==g['source_sha256']
 csvwrite(O/'curves/pull_training_scalars.csv',selected)
 write(O/'curves/pull_curve_audit.json',dict(source=ident(p),segments=list(groups.values()),expected='Full-rerun,from-scratch,zero-teammate-obs: seed721/722/723; critic-only50 then joint8000',definition='Actual last100 completed episodes, length in control steps, includes failure and completion; source Episode/mean_length_last100_steps; no inversion of reset counts',no_metric_substitution=True))
def metrics():
 targets={task:np.load(O/'metrics'/(task+'_fixed_human_targets.npz')) for task in ['push','pull']};rows=list(csv.DictReader((OLD/'episodes.csv').open()));new=[]
 for row in rows:
  if row['task'] not in targets:continue
  task=row['task'];z=targets[task];rec=dict(row)
  if row['variant']=='copied-single-prior':rec.update(palm_status='missing',palm_error_deg='',palm_missing_reason='no_completed_episodes')
  elif row['is_nominal']=='True':rec.update(palm_status='excluded_nominal',palm_error_deg='')
  elif row['completed_reference']!='True':rec.update(palm_status='not_completed',palm_error_deg='')
  else:
   p=Path(row['trajectory_path']);assert sha(p)==row['trajectory_sha256'];a=np.load(p);frames=a['reference_frame'].astype(int);desired=z['target_palm_world'][frames]
   actual=pm.actual_rotations(a['dof_pos'],a['root_quat_xyzw'],z['joint_names'],z['bases']);err=pm.angle(desired,actual)
   dest=O/'metrics'/'per_frame'/row['run_id']/('pm'+row['perturbation_level_cm']);dest.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(dest/(row['state_id']+'.npz'),error_deg=err,reference_frame=frames)
   rec.update(palm_status='complete',palm_error_deg=float(err.mean()),palm_valid_hand_frames=int(err.size),palm_target_sha256=sha(O/'metrics'/(task+'_fixed_human_targets.npz')),palm_missing_reason='')
  new.append(rec)
 csvwrite(O/'metrics/palm_episodes.csv',new);by={}
 for row in new:
  if row['is_nominal']=='True':continue
  key=tuple(row[k] for k in ['task','variant','perturbation_level_cm','training_seed']);by.setdefault(key,[]).append(row)
 seeds=[]
 for key,rs in by.items():
  complete=[r for r in rs if r['palm_status']=='complete'];assert len(rs)==50
  seeds.append(dict(zip(['task','variant','perturbation_level_cm','training_seed'],key),n_episodes=50,n_completed=len(complete),palm_mean_deg=float(np.mean([float(r['palm_error_deg']) for r in complete])) if complete else None,status='complete' if complete else 'missing',missing_reason='' if complete else 'no_completed_episodes'))
 csvwrite(O/'metrics/palm_per_seed.csv',seeds);groups={}
 for row in seeds:groups.setdefault(tuple(row[k] for k in ['task','variant','perturbation_level_cm']),[]).append(row)
 result=[]
 for key,rs in groups.items():
  assert len(rs)==3;vals=[r['palm_mean_deg'] for r in rs if r['palm_mean_deg'] is not None]
  result.append(dict(zip(['task','variant','perturbation_level_cm'],key),n_seeds=3,n_valid_seeds=len(vals),palm_mean_deg=float(np.mean(vals)) if len(vals)==3 else None,palm_sample_std_deg=float(np.std(vals,ddof=1)) if len(vals)==3 else None,status='complete' if len(vals)==3 else 'missing'))
 csvwrite(O/'metrics/palm_group_summary.csv',result)
def table3():
 rows=[]
 def add(task,comparison,metric,value,unit,src,selector,**extra):rows.append(dict(task=task,comparison=comparison,metric=metric,value=value,unit=unit,source_file=str(src),source_sha256=sha(src),source_selector=selector,**extra))
 for task in ['push','pull']:
  d=R/'logs/ReferenceGeometry/reference_geometry_round2_20261017'/task;sel=read(d/'selection.json');spec=read(d/'correspondence_spec.json');can=read(d/'fitted_candidates.json');a=np.array(spec['weighted_source']['source_anchors']);w=np.array(spec['weighted_source']['weights'])
  for i,role in enumerate(sel['role_parameters']):
   dest=np.array(next(c for c in can if c['id']==role['id'])['target_anchors']);Q=np.array(role['Q']);c=np.array(role['c'])
   if task=='push':
    for name,tr in [('prescribed_0.80m',np.array([-.4 if i==0 else .4,0,0])),('role_isometry',c)]:
     val=1000*np.sqrt(np.sum(w*np.sum((a@Q.T+tr-dest)**2,axis=1)));add(task,name,'weighted_anchor_RMS',float(val),'mm',d/'correspondence_spec.json',f'weighted_source + selected role{i}',role=i,transform_source_sha256=sha(d/'selection.json'),target_source_sha256=sha(d/'fitted_candidates.json'))
  if task=='pull':
   p=R/'logs/V3_20261102/U2/fit_result.json';s=read(p)
   for parity,fit in s['fits'].items():add(task,'eta='+parity,'weighted_anchor_RMS',1000*fit['weighted_rmse_m'],'mm',p,'fits.'+parity)
 src=R/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/shared_small_preview/manifest.json';s=read(src);scales=np.array(s['source_human_to_robot_scales'])
 for tag,val in [('one-sided',scales[0]),('mean',scales.mean())]:add('chair',tag,'max_size_mismatch',float(100*np.max(abs(val/scales-1))),'percent',src,'source_human_to_robot_scales')
 src=R/'logs/V3_20261102/U3_assets/chair_source_size_reopt_20261103/a1_wrist_only_rev112_preview/residual_decision.json';s=read(src);shutil.copy2(src,O/'reference/chair_source_size_decision.json')
 for hand,v in s['per_hand'].items():
  for field in ['mean_deg','median_deg','p95_deg','max_deg','over6_fraction','over6_count','over6_at_limit_count','over6_converged_count','all_solver_success']:add('chair','source-size-reopt',field,v[field],'deg' if field.endswith('_deg') else 'fraction' if field.endswith('fraction') else 'count_or_boolean',src,'per_hand.'+hand+'.'+field,hand=hand)
 add('chair','source-size-reopt','over6_at_limit_and_converged_fraction',s['over6_at_limit_and_converged_fraction'],'fraction',src,'over6_at_limit_and_converged_fraction')
 for task in ['chair','bucket']:
  src=R/f'paper_handoff_exports/EXPERIMENTS_R2_20261024_CPU_v2/checks/{task}_wrist_so3_arrays.npz';a=np.load(src)
  for v in ['before','after']:
   x=np.stack([a[f'role{i}_{v}_error_deg'] for i in range(2)],axis=1)
   for metric,value in [('palm_mean',x.mean()),('palm_max',x.max())]:add(task,v,metric,float(value),'deg',src,f'role{{0,1}}_{v}_error_deg: all frames/hands')
 csvwrite(O/'reference/Table_III_reference.csv',rows)
 for run in ['V3-04','V3-05','V3-06']:
  for p in (B/'v3_20261103_u3b_ops').glob(run+'*.launch.json'):shutil.copy2(p,O/'reference'/p.name)
if __name__=='__main__':
 for task in sys.argv[1:] or ['targets','table','curves','table3','metrics']:
  print('BEGIN',task,flush=True);{'targets':palm_targets,'table':small_table,'curves':curves,'table3':table3,'metrics':metrics}[task]();print('DONE',task,flush=True)
