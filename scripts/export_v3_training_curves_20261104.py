"""Export original scalar values, preserving phase/recovery provenance."""
import csv,hashlib,json,re
from pathlib import Path
R=Path('/home/kevin/holosoma-core4d-base');B=R/'logs/V3_20261102/cloud_backup_20261104/raw';O=R/'logs/V3_20261102/closeout_20261104';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for p in B.glob('*/logs/**/metrics.jsonl'):
 if 'V3-' in p.parent.name and 'smoke' not in p.parent.name.lower():sources[p]=('V3',p.parent.name)
for p in (R/'logs/V3_20261102/rev14_backup/raw').rglob('metrics.jsonl'):sources[p]=('V3_pre_resume',p.parent.name)
old=R/'logs/PaperSupplement20260929/unified_evaluation_20261003';new=R/'logs/ExperimentsR2_evaluation_20261026_retry01'
models=[m for m in read(old/'models.json') if m['phase']=='main' and m['case']!='push']+read(new/'models.json')
mapping={m['model_id']:m['local_original'] for m in read(new/'local_checkpoint_map.json')}
missing=[]
for m in models:
 cp=Path(mapping.get(m['model_id'],m['checkpoint']));p=cp.parent/'metrics.jsonl'
 if p.exists():sources[p]=('main_historical',m['model_id'])
 else:missing.append(dict(model_id=m['model_id'],reason='metrics.jsonl not present at verified checkpoint directory',checkpoint=str(cp)))
resumes={}
for p in sources:
 c=read(p.parent/'run_config.json');parent=c.get('resume_checkpoint')
 if parent:
  key=Path(parent).parent.name;resumes[key]=min(resumes.get(key,10**9),int(c['start_iteration']))
resumes.update({'V3-07_chair_no_wrist_seed721':2000,'V3-16_bucket_no_wrist_seed721':2000})
fields=['group','run_id','segment','training_seed','iteration','phase','mean_episode_length_steps','mean_episode_return_raw','mean_step_reward','horizon','control_hz','episode_window_count','accepted_before_resume_boundary','source_file','source_sha256','config_sha256','length_source_tag','return_source_tag','step_reward_source_tag','metrics_status']
inventory=[];total=0
with (O/'training_scalars.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for p,(group,label) in sorted(sources.items(),key=lambda x:str(x[0])):
  c=read(p.parent/'run_config.json');ph=sha(p);ch=sha(p.parent/'run_config.json');horizon=c.get('reference_frames',(c.get('experiment_contract') or {}).get('reference_frames'));n=0;bad=0;missinglen=0
  for line in p.read_text().splitlines():
   try:x=json.loads(line)
   except ValueError:bad+=1;continue
   if 'iteration' not in x:continue
   length=x.get('Episode/mean_length_last100_steps');ret=x.get('Episode/mean_return_raw_last100');missinglen+=length is None
   row=dict(group=group,run_id=label.split('_')[0] if label.startswith('V3-') else label,segment=p.parent.name,training_seed=c.get('seed'),iteration=x['iteration'],phase='critic_only' if c.get('critic_only') else 'joint',mean_episode_length_steps=length,mean_episode_return_raw=ret,mean_step_reward=x.get('reward_mean'),horizon=horizon,control_hz=c.get('control_hz',50),episode_window_count=x.get('Episode/completed_window_count'),accepted_before_resume_boundary=x['iteration']<=resumes.get(p.parent.name,10**9),source_file=str(p),source_sha256=ph,config_sha256=ch,length_source_tag='Episode/mean_length_last100_steps' if length is not None else '',return_source_tag='Episode/mean_return_raw_last100' if ret is not None else '',step_reward_source_tag='reward_mean',metrics_status='actual_episode_telemetry' if length is not None else 'missing_episode_length_not_substituted')
   w.writerow(row);n+=1;total+=1
  inventory.append(dict(file=str(p),sha256=ph,rows=n,malformed_lines=bad,missing_episode_length_rows=missinglen,resume_cutoff=resumes.get(p.parent.name),phase='critic_only' if c.get('critic_only') else 'joint',start_iteration=c.get('start_iteration'),requested_iterations=c.get('iterations_requested')))
(O/'training_curves_inventory.json').write_text(json.dumps(dict(rows=total,sources=inventory,missing=missing,notes=['Raw metrics exported directly, not reconstructed TensorBoard values.','Episode length uses actual last100 finished episodes; never inferred from completion/reset counts.','Critic warmup and recovery segments retained separately. Old points after the resume checkpoint are flagged excluded, not silently spliced.','Historical final-segment-only inputs do not establish full earlier-stage history. Source availability is explicit.']),indent=2)+'\n')
print('CURVES',total,'rows',len(inventory),'sources',len(missing),'missing historical sources')
