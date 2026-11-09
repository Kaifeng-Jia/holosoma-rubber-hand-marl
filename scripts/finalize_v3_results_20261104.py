"""V3 evidence-only closeout: frozen final evaluations and paired statistics."""
import argparse,csv,hashlib,json,math,re,sys,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
R=Path('/home/kevin/holosoma-core4d-base');B=R/'logs/V3_20261102/cloud_backup_20261104/raw'
OUT=R/'logs/V3_20261102/closeout_20261104';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'scripts'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def local(p):
 p=str(p)
 return B/p.removeprefix('/root/autodl-tmp/') if p.startswith('/root/autodl-tmp/') else Path(p)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvsave(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def variant(mid):
 if mid.startswith('V3-PullFull'):return 'Full-rerun',int(mid[-3:])
 n=int(mid.split('_')[0][3:]);groups=[(1,3,'one-sided-scale'),(4,6,'source-size-reopt'),(7,9,'no-wrist'),(16,18,'no-wrist'),(19,21,'zero-teammate-obs'),(25,27,'from-scratch')]
 for lo,hi,label in groups:
  if lo<=n<=hi:return label,721+n-lo
 raise ValueError(mid)
def collect():
 data=[];pending=[]
 for ops,expected in [('v3_continuous_final_eval_20261103_ops',21),('v3_extended_eval_20261104_ops',63),('v3_hard_eval_20261104_ops',18)]:
  count=0
  for p in sorted((B/ops).glob('*.exit.json')):
   d=read(p)
   if not d.get('validated') or d.get('returncode')!=0:continue
   sfile=local(d['output'])/'summary.json'
   if not sfile.exists():pending.append(str(sfile));continue
   assert sha(sfile)==d['summary_sha256'];s=read(sfile)
   if 'job' in d:
    mid=d['job']['id'];v,seed=variant(mid);level=2;model=d
   else:
    model=d['model'];mid=model['id'];level=model['level']
    if model['kind']=='full_scan':v='Full';seed=model['seed']
    elif model['kind']=='ablation_hard':v,seed=variant(mid)
    else:v='copied-single-prior';seed=model['seed']
   assert s['checkpoint_sha256']==model['checkpoint_sha256'] and s['states_sha256']==model['states_sha256']
   data.append(dict(case=s['case'],seed=seed,variant=v,level=level,id=mid,summary=s,summary_file=sfile,exit=p,model=model,summary_sha=sha(sfile)))
   count+=1
  if count!=expected:pending.append(f'{ops}: {count}/{expected}')
 # Twelve historical +/-2 Full cases are embedded with original verified exit records.
 for row in read(B/'v3_pm2_baselines.json'):
  p=Path(row['summary_source']);assert p.exists() and sha(p)==row['summary_sha256'];s=read(p)
  data.append(dict(case=row['case'],seed=row['seed'],variant='Full',level=2,id=s['model_id'],summary=s,summary_file=p,exit=B/'v3_pm2_baselines.json',model={},summary_sha=sha(p)))
 new=R/'logs/ExperimentsR2_evaluation_20261026_retry01'
 for m in read(new/'models.json'):
  folder=new/m['group'];p=folder/'results'/m['model_id']/'summary.json';exit=folder/(m['model_id']+'.exit.json');ex=read(exit)
  assert ex['validated'] and ex['process_returncode']==0;s=read(p);assert s['checkpoint_sha256']==m['checkpoint_sha256']
  data.append(dict(case=m['case'],seed=m['training_seed'],variant='Full',level=2,id=m['model_id'],summary=s,summary_file=p,exit=exit,model={},summary_sha=sha(p)))
 keys=[(x['case'],x['variant'],x['level'],x['seed']) for x in data];assert len(set(keys))==len(keys)
 return data,pending
def main():
 p=argparse.ArgumentParser();p.add_argument('--hand-metrics',action='store_true');a=p.parse_args()
 datasets,pending=collect();palm={}
 extra_palm=OUT/'historical_full_palm'
 if a.hand_metrics and (B/'v3_palm_targets/manifest.json').exists():
  import types
  src=B/'v3_palm_postprocess.py';code=src.read_text()
  code=code.replace("B=Path('/root/autodl-tmp')",'B=Path('+repr(str(B))+')')
  code=code.replace("OUT=B/'v3_palm_metrics_20261104'",'OUT=Path('+repr(str(extra_palm))+')')
  mod=types.ModuleType('v3_local_historical_palm');exec(compile(code,str(src),'exec'),mod.__dict__)
  mod.tests()
  for ds in datasets:
   if ds['variant']=='Full' and ds['level']==2 and ds['case'] in ['chair','bucket','table']:mod.process(ds['summary_file'])
 for p in list((B/'v3_palm_metrics_20261104').glob('*/summary.json'))+list(extra_palm.glob('*/summary.json')):
  x=read(p);palm[x['evaluation_summary_sha256']]=x
 refs={};fk=None;refinfo={}
 if a.hand_metrics:
  from prepare_experiments_r2_cpu import BatchFK,reference,HANDS,BODY
  import torch
  torch.set_num_threads(2);fk=BatchFK();write(OUT/'hand_fk_test.json',fk.validate())
 cache=OUT/'hand_cache';cache.mkdir(exist_ok=True)
 def handmetrics(ds):
  s=ds['summary'];key=ds['summary_sha'];cp=cache/(key+'.json')
  if cp.exists():return read(cp)
  if not a.hand_metrics:return {}
  v=ds['model'].get('reference_variant') or s.get('reference_variant')
  if v:
   vp=local(v);desc=read(vp);root=local(ds['model']['root'])/'src/holosoma';rp=root/desc['runtime_reference_file']
   assert sha(rp)==desc['runtime_reference_sha256'];refkey=sha(rp)
   if refkey not in refs:
    with np.load(rp,allow_pickle=False) as z:refs[refkey]={k:z[k].copy() for k in z.files}
    refinfo[refkey]=dict(path=str(rp),sha256=refkey)
  else:
   spacing=s.get('placement_provenance') or {};spacing=spacing.get('lateral_spacing_m',None)
   if ds['case']=='push' and spacing is None:
    baseline=R/'paper_handoff_exports/EXPERIMENTS_R2_20261026/checks/push_reference_for_fk.npz'
    refkey=sha(baseline)
    if refkey not in refs:
     with np.load(baseline,allow_pickle=False) as z:refs[refkey]={k:z[k].copy() for k in z.files}
     refinfo[refkey]=dict(path=str(baseline),sha256=refkey)
   else:
    refkey=ds['case']+':'+str(spacing)
    if refkey not in refs:
     ref,rp=reference(ds['case'],BODY+HANDS,spacing=spacing or .8)
     refs[refkey]={k:v.numpy() for k,v in ref.sample(torch.arange(ref.num_frames)).items()};refs[refkey]['body_names']=BODY+HANDS
     refinfo[refkey]=dict(path=str(rp),sha256=sha(rp))
  ref=refs[refkey];names=list(map(str,ref['body_names']));hi=[names.index(x) for x in HANDS];rows={}
  for ep in s['episodes']:
   if ep['kind']!='perturbed' or not ep['completed_reference']:continue
   path=ds['summary_file'].parent/Path(ep['trajectory']).name
   if not path.exists():pending.append(str(path));continue
   with np.load(path,allow_pickle=False) as z:
    f=z['reference_frame'].astype(int);offset=z['reference_object_pos_w']-ref['object_pos_w'][f]
    assert np.max(abs(offset-offset[0]))<3e-6,(ds['id'],'reference alignment')
    expected=ref['agent_body_pos_w'][f][:,:,hi,:]+offset[:,None,None,:]
    actual=fk.poses(z['root_pos'],z['root_quat_xyzw'],z['dof_pos'],HANDS)[...,:3,3]
    error=np.linalg.norm(actual-expected,axis=-1)*100
    rows[ep['state_id']]=dict(hand_pos_error_cm=float(error.mean()),n_metric_frames=len(f),trajectory_sha256=sha(path))
  result=dict(reference=refinfo[refkey],episodes=rows,formula='Mean Euclidean rubber-hand-link-origin position error in cm, 2 agents x2 hands x all saved frames including initial and terminal, indexed by reference_frame, no planar alignment. Failed prefixes excluded.')
  if len(rows)==sum(e['kind']=='perturbed' and e['completed_reference'] for e in s['episodes']):write(cp,result)
  return result
 episodes=[];per=[];rawsources=[]
 for ds in datasets:
  s=ds['summary'];eps=s['episodes'];assert len(eps)==51 and len({e['state_id'] for e in eps})==51
  assert sum(e['kind']=='perturbed' for e in eps)==50
  pm=palm.get(ds['summary_sha']);pmrows={x['state_id']:x for x in pm['episodes']} if pm else {};hm=handmetrics(ds)
  rawsources.append(dict(task=ds['case'],variant=ds['variant'],seed=ds['seed'],level=ds['level'],summary_path=str(ds['summary_file']),summary_sha256=ds['summary_sha'],exit_record_path=str(ds['exit'])))
  for e in eps:
   completed=e['completed_reference'];nom=e['kind']=='nominal';path=ds['summary_file'].parent/Path(e['trajectory']).name;ph=pmrows.get(e['state_id']);hh=hm.get('episodes',{}).get(e['state_id'])
   palm_status='not_applicable' if ds['case']=='kick' else 'excluded_nominal' if nom else 'not_completed' if not completed else 'complete' if ph else 'missing'
   row=dict(experiment_group='Full' if ds['variant']=='Full' else 'V3',task=ds['case'],variant=ds['variant'],run_id=ds['id'],training_seed=ds['seed'],model_id=s['model_id'],checkpoint_iteration=s['checkpoint_iteration'],checkpoint_sha256=s['checkpoint_sha256'],perturbation_level_cm=ds['level'],perturbation_level_deg=ds['level'],state_id=e['state_id'],state_file_sha256=s['states_sha256'],is_nominal=nom,completed_reference=completed,termination_reason=json.dumps(e['terminal_terms'],sort_keys=True),executed_steps=e['steps'],reference_horizon_steps=e['horizon_steps'],object_pos_rmse_cm=e['object_rmse_3d_m']*100,object_xy_rmse_cm=e['object_rmse_xy_m']*100,height_rmse_cm=e['height_rmse_m']*100,hand_pos_error_cm=hh['hand_pos_error_cm'] if hh else None,hand_status='complete' if hh else 'excluded_nominal' if nom else 'not_completed' if not completed else 'missing',palm_error_deg=ph['mean_deg'] if ph else None,palm_status=palm_status,n_metric_frames=e['steps']+1,palm_valid_hand_frames=ph['count'] if ph else 0,trajectory_path=str(path),trajectory_sha256=sha(path) if path.exists() else None,config_sha256=None,reference_sha256=(s.get('training_contract') or {}).get('runtime_reference_sha256'),object_urdf_sha256=(s.get('training_contract') or {}).get('object_urdf_sha256'),evaluation_code_sha256=ds['model'].get('evaluator_sha256'),exit_record_path=str(ds['exit']),summary_sha256=ds['summary_sha'],progress=e['progress'],raw_reward_sum_common=e['reward_sum_common'])
   episodes.append(row)
  rs=[r for r in episodes if r['run_id']==ds['id'] and r['perturbation_level_cm']==ds['level'] and not r['is_nominal'] and r['variant']==ds['variant']]
  assert len(rs)==50
  done=[r for r in rs if r['completed_reference']]
  pr=dict(task=ds['case'],variant=ds['variant'],training_seed=ds['seed'],perturbation_level_cm=ds['level'],run_id=ds['id'],n_episodes=50,n_completed=len(done),completion_pct=len(done)*2,summary_sha256=ds['summary_sha'],independent_prior_count=1 if ds['variant']=='copied-single-prior' else None)
  for k in ['object_pos_rmse_cm','hand_pos_error_cm','palm_error_deg']:
   values=[r[k] for r in done if r[k] is not None];pr[k+'_mean']=float(np.mean(values)) if values else None;pr[k+'_n_episodes']=len(values)
  per.append(pr)
  print(ds['id'],ds['level'],len(done),flush=True)
 csvsave('episodes.csv',episodes);csvsave('per_seed_summary.csv',per);write(OUT/'source_index.json',rawsources)
 grouped=defaultdict(list)
 for row in per:grouped[(row['task'],row['variant'],row['perturbation_level_cm'])].append(row)
 groups=[]
 for (task,v,level),rs in sorted(grouped.items()):
  g=dict(task=task,variant=v,perturbation_level_cm=level,n_seeds=len(rs),complete_three_seeds=sorted(r['training_seed'] for r in rs)==[721,722,723],n_completed=sum(r['n_completed'] for r in rs),n_episodes=sum(r['n_episodes'] for r in rs),per_seed_csv_sha256=sha(OUT/'per_seed_summary.csv'))
  for k in ['completion_pct','object_pos_rmse_cm_mean','hand_pos_error_cm_mean','palm_error_deg_mean']:
   vals=[r[k] for r in rs if r[k] is not None];g[k+'_valid_seeds']=len(vals);g[k+'_mean']=float(np.mean(vals)) if vals else None;g[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
  if v=='copied-single-prior':g['replicate_note']='3 evaluation keys, one identical prior per task; not three independently trained policies'
  groups.append(g)
 csvsave('group_summary.csv',groups)
 index={(r['task'],r['variant'],r['perturbation_level_cm'],r['training_seed'],r['state_id']):r for r in episodes if not r['is_nominal']}
 pairrows=[];stats=[];errstats=[]
 for task,v,level in sorted(grouped):
  if v=='Full':continue
  prs=[]
  for seed in [721,722,723]:
   for i in range(50):
    sid=f'perturbed_{i:03d}';full=index.get((task,'Full',level,seed,sid));other=index.get((task,v,level,seed,sid))
    if not full or not other:continue
    assert full['state_file_sha256']==other['state_file_sha256'],(task,v,level,seed)
    q=dict(task=task,variant=v,perturbation_level_cm=level,training_seed=seed,state_id=sid,state_file_sha256=full['state_file_sha256'],full_completed=full['completed_reference'],variant_completed=other['completed_reference'],full_run_id=full['run_id'],variant_run_id=other['run_id'],both_completed=full['completed_reference'] and other['completed_reference'])
    for metric in ['object_pos_rmse_cm','hand_pos_error_cm','palm_error_deg']:
     valid=q['both_completed'] and full[metric] is not None and other[metric] is not None
     q[metric+'_full']=full[metric] if valid else None;q[metric+'_variant']=other[metric] if valid else None;q[metric+'_variant_minus_full']=other[metric]-full[metric] if valid else None
    prs.append(q)
  pairrows+=prs
  a=sum(x['full_completed'] and x['variant_completed'] for x in prs);b=sum(x['full_completed'] and not x['variant_completed'] for x in prs);c=sum(not x['full_completed'] and x['variant_completed'] for x in prs);d=len(prs)-a-b-c
  stats.append(dict(task=task,variant=v,perturbation_level_cm=level,n_pairs=len(prs),complete_150_pairs=len(prs)==150,both_complete=a,only_full_complete=b,only_variant_complete=c,both_fail=d,p_exact=float(binomtest(min(b,c),b+c,.5,alternative='two-sided').pvalue) if b+c else 1.))
  for seed in [721,722,723]:
   es=[x for x in prs if x['training_seed']==seed];er=dict(task=task,variant=v,perturbation_level_cm=level,training_seed=seed,n_common_completed=sum(x['both_completed'] for x in es))
   for k in ['object_pos_rmse_cm','hand_pos_error_cm','palm_error_deg']:
    vals=[x[k+'_variant_minus_full'] for x in es if x[k+'_variant_minus_full'] is not None];er[k+'_n_pairs']=len(vals);er[k+'_mean_variant_minus_full']=float(np.mean(vals)) if vals else None
   errstats.append(er)
 csvsave('paired_episode_details.csv',pairrows);csvsave('paired_completion.csv',stats);csvsave('paired_errors.csv',errstats)
 if (B/'v3_hard_eval_20261104_ops/hard_level_selection.json').exists():shutil.copy2(B/'v3_hard_eval_20261104_ops/hard_level_selection.json',OUT/'hard_level_selection.json')
 write(OUT/'aggregation_status.json',dict(evaluation_units=len(datasets),expected_units=120,episodes=len(episodes),expected_episodes=6120,missing_input=pending,palm_status_counts=dict(__import__('collections').Counter(x['palm_status'] for x in episodes)),hand_status_counts=dict(__import__('collections').Counter(x['hand_status'] for x in episodes)),gpu_evaluation_tables_complete=len(datasets)==120 and not pending,all_paper_requirements_complete=False,note='Statistics are evidence, not manuscript claims. Pending Palm mapping/other U5 items must not be declared complete. Copy-prior rows are repeated evaluation keys of one prior per task.'))
 print('SUMMARY',len(datasets),len(episodes),'pending',len(pending),flush=True)
if __name__=='__main__':main()
