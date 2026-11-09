"""Bounded six-run Push supplement. Detached staged training, immutable old results."""
import fcntl,hashlib,json,math,os,signal,subprocess,time,sys,types
from pathlib import Path
ROOT=Path('/root/autodl-tmp/push_seeds_20261107');OPS=Path('/root/autodl-tmp/push_seeds_20261107_ops');PY=Path('/root/autodl-tmp/r2_deps/hssim/bin/python')
STAGES={'prescribed':[(0,50),(50,8000),(8050,7000)],'registered':[(0,50),(50,8000),(8050,5000)]}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,float) or math.isfinite(x)
def latest(out):
 p=Path(out)/'metrics.jsonl'
 if not p.exists():return None
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-64000));lines=f.read().splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except (ValueError,UnicodeDecodeError):pass
 return None
def launch(variant,seed,gpu,stage,smoke=False,parent=None):
 start,count=STAGES[variant][stage]
 if smoke:count=2
 label=f'{variant}_seed{seed}_'+('smoke' if smoke else f'stage{stage}_from{start}_plus{count}')
 out=ROOT/'logs'/label;assert not out.exists(),out
 base=ROOT/'baseline_configs'/f'{variant}_stage{stage}.json'
 cmd=[str(PY),'-B',str(ROOT/'scripts/push_20261107_entry.py'),'--skill','push','--seed',str(seed),'--num-envs','2048','--steps-per-env','24','--iterations',str(count),'--save-interval','1000','--actor-learning-rate','0.00001','--critic-learning-rate','0.001','--output-dir',str(out)]
 if stage==0:cmd+=['--critic-only']
 if variant=='registered':cmd+=['--push-placement-proof',str(ROOT/'r2_evidence/preflight_push_registered.json')]
 if parent:cmd+=['--resume',str(parent)]
 env=os.environ.copy()
 for k in ['JPY_PARENT_PID','ISAAC_JUPYTER_KERNEL','ISAAC_JUPYTER_PYTHON_PACKAGE','RANK']:env.pop(k,None)
 # WORLD_SIZE=1: actual CUDA is selected only by CUDA_VISIBLE_DEVICES;
 # LOCAL_RANK isolates each process's forced USD conversion directory.
 rank=seed*10+stage*2+(variant=='registered')
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),LOCAL_RANK=str(rank),WORLD_SIZE='1',OMNI_KIT_ACCEPT_EULA='YES',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONPATH=str(ROOT/'src/holosoma')+':'+str(ROOT/'src/holosoma_retargeting'),PUSH721_CONFIG=str(base),PUSH_IS_SMOKE='1' if smoke else '0')
 with (OPS/(label+'.stdout.log')).open('x') as f:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(label=label,variant=variant,seed=seed,gpu=gpu,stage=stage,smoke=smoke,start=start,count=count,output=str(out),argv=cmd,pid=proc.pid,started_unix=time.time(),baseline=str(base),baseline_sha256=sha(base),parent_checkpoint=str(parent) if parent else None,parent_sha256=sha(parent) if parent else None)
 save(OPS/(label+'.launch.json'),rec);return dict(proc=proc,record=rec,endpoint_checked=False)
def checkpoint_check(path,expected):
 # Separate CPU process avoids creating a CUDA context in the supervisor.
 code='import torch,sys,math,json; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False);\n'
 code+='def valid(v):\n if torch.is_tensor(v): return bool(torch.isfinite(v).all())\n if isinstance(v,dict): return all(valid(t) for t in v.values())\n if isinstance(v,(list,tuple)): return all(valid(t) for t in v)\n return not isinstance(v,float) or math.isfinite(v)\n'
 code+='assert valid(x); assert x["iter"]==int(sys.argv[2]),x.keys(); print("finite_checkpoint_verified")'
 p=subprocess.run([str(PY),'-B','-c',code,str(path),str(expected)],capture_output=True,text=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
 return sha(path)
def planned_stop(job):
 r=job['record'];p=Path(r['output'])/'model_13050.pt'
 if r['variant']!='prescribed' or r['stage']!=2 or not p.exists() or job.get('planned_stop'):return
 # Check loadability and a stable size before signaling. Preserve the 7000 budget.
 s=p.stat();time.sleep(.5)
 if s.st_size!=p.stat().st_size:return
 try:h=checkpoint_check(p,13050)
 except Exception:return
 when=time.time();os.killpg(job['proc'].pid,signal.SIGTERM)
 job['planned_stop']=dict(reason='User-authorized early termination after valid model_13050; original +7000 run_config retained',checkpoint=str(p),sha256=h,stop_requested_unix=when,mechanism='Supervisor executes the user-authorized termination; not a normal +7000 completion and not an invalid training')
 save(OPS/(r['label']+'.planned_stop.json'),job['planned_stop'])
def finish(job,suffix='.exit.json'):
 r=job['record'];out=Path(r['output']);rc=job['proc'].returncode;end=2 if r['smoke'] else 13050 if job.get('planned_stop') else r['start']+r['count']
 result=dict(**r,exit_code=rc,ended_unix=time.time(),validated=False)
 try:
  assert rc==0 or job.get('planned_stop'),f'Process exit {rc}'
  assert read(out/'config_diff.json')['passed']
  records=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
  selected=[x for x in records if x['iteration']<=end]
  assert [x['iteration'] for x in selected]==list(range(r['start']+1,end+1))
  assert finite(selected),'Nonfinite metric'
  model=out/f'model_{end:05d}.pt';h=checkpoint_check(model,end)
  if not job.get('planned_stop'):assert read(out/'status.json')['passed']
  result.update(validated=True,final_iteration=end,final_checkpoint=str(model),sha256=h,planned_stop=job.get('planned_stop'),post_endpoint_iterations_excluded=[x['iteration'] for x in records if x['iteration']>end])
 except Exception as e:result['error']=repr(e)
 save(OPS/(r['label']+suffix),result);return result
def main():
 assert read(OPS/'deployment.json')['ready']
 lock=(OPS/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reuse_smoke='--formal-after-smoke-recheck' in sys.argv
 if reuse_smoke:
  previous=read(OPS/'queue_status.json');assert previous['phase']=='smoke_failed_no_formal_training'
  save(OPS/'original_smoke_validator_status.json',previous)
 else:assert not (OPS/'queue_status.json').exists(),'Queue already exists; inspect before resuming'
 jobs=[];finished=[];failed=[]
 def publish(phase):
  active=[]
  for j in jobs:
   r=dict(j['record']);m=latest(r['output']);r['latest_iteration']=m['iteration'] if m else None;r['seconds_per_iteration']=m.get('elapsed_seconds') if m else None;active.append(r)
  save(OPS/'queue_status.json',dict(phase=phase,supervisor_pid=os.getpid(),updated_unix=time.time(),active=active,finished=finished,failed=failed))
 def poll():
  for j in jobs[:]:
   r=j['record'];m=latest(r['output'])
   if m and not finite(m):os.killpg(j['proc'].pid,signal.SIGTERM)
   planned_stop(j)
   if j['proc'].poll() is None:continue
   res=finish(j);jobs.remove(j);finished.append(res)
   if not res['validated']:failed.append(res);continue
   if not r['smoke'] and r['stage']<2:jobs.append(launch(r['variant'],r['seed'],r['gpu'],r['stage']+1,parent=res['final_checkpoint']))
  publish('training' if any(not j['record']['smoke'] for j in jobs) else 'smoke')
 if reuse_smoke:
  for variant in ['registered','prescribed']:
   r=read(OPS/f'{variant}_seed724_smoke.launch.json');old=read(OPS/(r['label']+'.exit.json'));assert old['exit_code']==0
   result=finish(dict(record=r,proc=types.SimpleNamespace(returncode=0)),suffix='.validation_recheck.json');assert result['validated'],result
   finished.append(result)
 else:
  jobs=[launch('registered',724,0,0,True),launch('prescribed',724,1,0,True)]
  while jobs:
   poll();time.sleep(3)
 if failed:publish('smoke_failed_no_formal_training');return
 # Two independent tasks per GPU; remaining seeds automatically take free slots.
 pending=[('registered',724,0),('prescribed',724,1),('registered',725,1),('prescribed',725,0),('registered',726,0),('prescribed',726,1)]
 while pending or jobs:
  if (OPS/'STOP_NEW_LAUNCHES').exists():publish('new_launches_paused');time.sleep(10);continue
  for item in pending[:]:
   v,s,g=item
   if sum(j['record']['gpu']==g for j in jobs)<2:
    jobs.append(launch(v,s,g,0));pending.remove(item)
  poll();time.sleep(3)
 publish('training_finished_evaluation_pending' if not failed else 'training_has_failures_inspection_required')
if __name__=='__main__':main()
