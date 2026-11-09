"""Authorized reboot recovery: preserve six-run identity, resume only seed726."""
import fcntl,hashlib,json,os,signal,subprocess,time,sys,shutil
from pathlib import Path
ROOT=Path('/root/autodl-tmp/push_seeds_20261107');OLDOPS=Path('/root/autodl-tmp/push_seeds_20261107_ops');OPS=Path('/root/autodl-tmp/push_seeds_20261108_recovery_ops');PY=Path('/root/autodl-tmp/r2_deps/hssim/bin/python')
sys.path.insert(0,str(ROOT/'scripts'))
import push_20261107_queue_v2 as q
def read(p):return json.loads(Path(p).read_text())
def save(p,x):q.save(p,x)
def audit():
 assert not (OPS/'recovery_plan.json').exists(),'Refuse overwrite recovery'
 OPS.mkdir(exist_ok=True)
 old=read(OLDOPS/'queue_status.json');completed=[]
 for v in ['prescribed','registered']:
  for seed in [724,725]:
   folder=ROOT/'logs'/f'{v}_seed{seed}_stage2_from8050_plus{7000 if v=="prescribed" else 5000}';model=folder/'model_13050.pt';h=q.checkpoint_check(model,13050)
   rec=dict(variant=v,seed=seed,iteration=13050,checkpoint=str(model),sha256=h,config_sha256=q.sha(folder/'run_config.json'),retrain=False)
   if v=='prescribed':
    stop=read(OLDOPS/(folder.name+'.planned_stop.json'));assert stop['sha256']==h
    rec.update(planned_stop=stop,actual_exit_time=None,actual_exit_code=None,note='Stop request recorded; process did not produce a reaped exit record before host shutdown. Actual exit time/code unknown; checkpoint and full metrics through13050 verified.')
   else:
    exit=read(OLDOPS/(folder.name+'.exit.json'));assert exit['validated'] and exit['sha256']==h;rec.update(actual_exit_time=exit['ended_unix'],actual_exit_code=exit['exit_code'])
   assert read(folder/'config_diff.json')['passed']
   rows=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()];selected=[x for x in rows if x['iteration']<=13050];assert [x['iteration'] for x in selected]==list(range(8051,13051));assert q.finite(selected)
   completed.append(rec)
 plans=[]
 for v,gpu,start in [('registered',0,1050),('prescribed',1,50)]:
  olddir=ROOT/'logs'/f'{v}_seed726_stage1_from50_plus8000'
  source=olddir/'model_01050.pt' if start==1050 else ROOT/'logs'/f'{v}_seed726_stage0_from0_plus50/model_00050.pt'
  h=q.checkpoint_check(source,start);records=[]
  for l in (olddir/'metrics.jsonl').read_text().splitlines():
   try:records.append(json.loads(l))
   except ValueError:pass
  plans.append(dict(variant=v,seed=726,gpu=gpu,start=start,target_stage_end=8050,remaining=8050-start,parent=str(source),parent_sha256=h,last_logged_iteration=records[-1]['iteration'],discarded_uncommitted_iterations=records[-1]['iteration']-start,old_segment=str(olddir),reason='User-reported host shutdown/restart; no invalid-run exclusion or new seed'))
 save(OPS/'recovery_plan.json',dict(created_unix=time.time(),old_queue_status_sha256=q.sha(OLDOPS/'queue_status.json'),completed=completed,resume=plans,source_code_changed=False,optimizer_and_normalizers_restored=True,environment_reset_after_resume=True,rng_and_simulator_state_not_saved_by_original_checkpoint=True))
 print(json.dumps(dict(completed=[(r['variant'],r['seed']) for r in completed],resume=plans),indent=2),flush=True)
 return plans
def launch_recovery(plan):
 v=plan['variant'];seed=726;start=plan['start'];count=plan['remaining'];stage=1;out=ROOT/'logs'/f'{v}_seed726_stage1_recovery_20261108_from{start}_plus{count}';assert not out.exists()
 original=ROOT/'baseline_configs'/f'{v}_stage1.json';expected=read(original);expected.update(start_iteration=start,iterations_requested=count)
 derived=OPS/f'{v}_stage1_recovery_expected_config.json';save(derived,expected)
 differences=[dict(field=k,before=read(original)[k],after=expected[k],reason='reboot recovery remaining budget to original boundary8050') for k in ['start_iteration','iterations_requested'] if read(original)[k]!=expected[k]]
 save(OPS/f'{v}_recovery_config_delta.json',dict(original=str(original),original_sha256=q.sha(original),derived=str(derived),derived_sha256=q.sha(derived),differences=differences,original_stage_end=8050,new_stage_end=start+count,original_parent=plan['parent'],original_parent_sha256=plan['parent_sha256']))
 cmd=[str(PY),'-B',str(ROOT/'scripts/push_20261107_entry.py'),'--skill','push','--seed','726','--num-envs','2048','--steps-per-env','24','--iterations',str(count),'--save-interval','1000','--actor-learning-rate','0.00001','--critic-learning-rate','0.001','--output-dir',str(out),'--resume',plan['parent']]
 if v=='registered':cmd+=['--push-placement-proof',str(ROOT/'r2_evidence/preflight_push_registered.json')]
 env=os.environ.copy()
 for k in ['JPY_PARENT_PID','ISAAC_JUPYTER_KERNEL','ISAAC_JUPYTER_PYTHON_PACKAGE','RANK']:env.pop(k,None)
 env.update(CUDA_VISIBLE_DEVICES=str(plan['gpu']),LOCAL_RANK=str(8262+(v=='registered')),WORLD_SIZE='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONPATH=str(ROOT/'src/holosoma')+':'+str(ROOT/'src/holosoma_retargeting'),PUSH721_CONFIG=str(derived),PUSH_IS_SMOKE='0')
 with (OPS/(out.name+'.stdout.log')).open('x') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(label=out.name,variant=v,seed=seed,gpu=plan['gpu'],stage=stage,smoke=False,start=start,count=count,output=str(out),argv=cmd,pid=p.pid,started_unix=time.time(),baseline=str(derived),baseline_sha256=q.sha(derived),parent_checkpoint=plan['parent'],parent_sha256=plan['parent_sha256'],recovery=True)
 save(OPS/(out.name+'.launch.json'),rec);return dict(proc=p,record=rec)
def main():
 OPS.mkdir(exist_ok=True)
 lock=(OLDOPS/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OPS/'queue_status.json').exists(),'Recovery already launched'
 plans=audit();q.OPS=OPS;jobs=[launch_recovery(p) for p in plans];finished=[];failed=[]
 while jobs:
  for j in jobs[:]:
   r=j['record'];q.planned_stop(j)
   if j.get('planned_stop') and j['proc'].poll() is None and time.time()-j['planned_stop']['stop_requested_unix']>30:
    os.killpg(j['proc'].pid,signal.SIGKILL);j['planned_stop']['forced_exit_after_sigterm_timeout_unix']=time.time();save(OPS/(r['label']+'.planned_stop.json'),j['planned_stop'])
   if j['proc'].poll() is None:continue
   result=q.finish(j);finished.append(result);jobs.remove(j)
   if not result['validated']:failed.append(result);continue
   if r['stage']==1:jobs.append(q.launch(r['variant'],726,r['gpu'],2,parent=result['final_checkpoint']))
  active=[]
  for j in jobs:
   r=dict(j['record']);m=q.latest(r['output']);r.update(latest_iteration=m['iteration'] if m else None,seconds_per_iteration=m.get('elapsed_seconds') if m else None);active.append(r)
  save(OPS/'queue_status.json',dict(phase='training' if jobs else 'training_finished_evaluation_pending' if not failed else 'failed_inspection_required',updated_unix=time.time(),supervisor_pid=os.getpid(),active=active,finished=finished,failed=failed,previous_four_completed=read(OPS/'recovery_plan.json')['completed']))
  if jobs:time.sleep(3)
if __name__=='__main__':main()
