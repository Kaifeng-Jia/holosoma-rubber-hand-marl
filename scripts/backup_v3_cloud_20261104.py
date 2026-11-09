"""Bounded V3 preservation, no remote mutation/deletion, SHA256 verified."""
import concurrent.futures,datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/home/kevin/holosoma-core4d-base');D=R/'logs/V3_20261102/cloud_backup_20261104'
SSH=['ssh','-C','-i','/home/kevin/.ssh/id_ed25519_core4d_cloud','-o','UserKnownHostsFile=/tmp/v3_rev14_known_hosts','-o','StrictHostKeyChecking=yes','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','20797']
HOST='root@connect.nmb1.seetacloud.com'
REMOTE=r'''
import pathlib,json,hashlib,time
B=pathlib.Path('/root/autodl-tmp');files={};groups={}
def add(p,priority,reason):
 if not p.is_file() or p.is_symlink():return
 rel=str(p.relative_to(B));old=files.get(rel)
 if old is None or priority<old[0]:files[rel]=(priority,reason)
roots=['v3_20261102_rev16_retry01','v3_20261102_rev19','v3_20261103_pull','v3_20261103_u3b','v3_full_eval_20261104']
ops=['v3_20261102_rev18_ops','v3_20261102_rev19_ops','v3_rev112_recovery_ops','v3_20261103_pull_retry01_ops','v3_pull_zero_eval_retry02_ops','v3_20261103_u3b_ops','v3_continuous_final_eval_20261103_ops','v3_extended_eval_20261104_ops','v3_hard_eval_20261104_ops']
for name in roots:
 root=B/name
 for p in (root/'logs').rglob('*'):
  if not p.is_file() or p.is_symlink():continue
  if p.suffix=='.pt':groups.setdefault(p.parent,[]).append(p);continue
  if p.suffix in ['.json','.jsonl','.csv','.tsv','.yaml','.yml','.log','.npz','.gz','.txt','.sha256'] or 'tfevents' in p.name:add(p,2 if p.suffix in ['.npz','.log','.jsonl'] or 'tfevents' in p.name else 0,'training/evaluation record')
 for folder in ['scripts','src']:
  for p in (root/folder).rglob('*'):
   if p.suffix in ['.py','.toml','.json','.yaml','.yml','.urdf'] and '__pycache__' not in p.parts:add(p,1,'frozen code/config/URDF')
 for p in root.glob('*manifest*.json'):add(p,0,'immutable source manifest')
 for p in root.glob('*.json'):add(p,0,'root audit/config evidence')
 for p in (root/'v3_evidence').rglob('*'):add(p,1,'frozen initialization states and placement evidence')
 for p in (root/'src').rglob('*'):
  if 'v3_20261102' in p.parts and p.suffix in ['.npz','.obj','.stl','.json','.urdf']:add(p,1,'new variant assets')
for parent,models in groups.items():
 numbered=[p for p in models if p.stem.startswith('model_') and p.stem[6:].isdigit()]
 if numbered:add(max(numbered,key=lambda p:int(p.stem[6:])),0,'latest checkpoint in each run/segment')
 for p in models:
  if p.name in ['model_00000.pt','model_00050.pt'] or p not in numbered:add(p,0,'initialization/critic-warmup checkpoint')
for name in ops:
 for p in (B/name).rglob('*'):
  if p.suffix in ['.json','.jsonl','.log','.csv','.tsv','.npz','.pt','.py','.txt','.sha256']:add(p,0 if p.suffix in ['.json','.pt','.py'] else 2,'operation receipt/state/result')
for name in ['v3_palm_metrics_20261104','v3_palm_targets','v3_prior_sources','v3_u5_f3_20261104']:
 for p in (B/name).rglob('*'):
  if p.suffix in ['.json','.jsonl','.log','.csv','.tsv','.npz','.png','.jpg','.mp4','.py','.txt','.sha256','.urdf','.pt'] or p.name=='SHA256SUMS':add(p,1 if p.suffix in ['.mp4','.png','.jpg','.urdf','.pt'] else 2,'derived metrics/media and provenance')
for p in B.glob('v3*.py'):add(p,1,'queue/collector code')
for p in B.glob('v3*.json'):add(p,0,'root receipt or frozen comparison evidence')
rows=[]
for rel,(priority,reason) in sorted(files.items(),key=lambda item:(item[1][0],item[0])):
 p=B/rel;st=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 after=p.stat();rows.append(dict(path=rel,size=st.st_size,mtime_ns=st.st_mtime_ns,sha256=h.hexdigest(),stable_at_scan=(after.st_size==st.st_size and after.st_mtime_ns==st.st_mtime_ns),priority=priority,reason=reason))
def rd(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
std=rd(B/'v3_continuous_final_eval_20261103_ops/status.json');hard=rd(B/'v3_hard_eval_20261104_ops/status.json');ext=rd(B/'v3_extended_eval_20261104_ops/status.json')
print(json.dumps(dict(time=time.time(),files=rows,standard_status=std,hard_status=hard,extended_status=ext,scope='V3 latest+initial checkpoints, all selected run records/evaluation trajectories, frozen code, new variant assets and media; excludes environments/caches and intermediate checkpoints',roots=roots,ops=ops)))
'''
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 D.mkdir(parents=True,exist_ok=True);(D/'raw').mkdir(exist_ok=True)
 with (D/'backup.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  deadline=time.monotonic()+3*3600;cycle=max([int(p.stem.split('_')[-1]) for p in D.glob('inventory_[0-9][0-9][0-9].json')]+[0])
  while time.monotonic()<deadline:
   cycle+=1;print('SCAN',cycle,flush=True)
   p=subprocess.run(SSH+[HOST,'python3 -'],input=REMOTE,text=True,capture_output=True,timeout=600)
   if p.returncode:write(D/'status.json',dict(state='remote_scan_failed',stderr=p.stderr));return
   inv=json.loads(p.stdout);write(D/f'inventory_{cycle:03d}.json',inv);write(D/'latest_inventory.json',inv)
   rows=inv['files'];write(D/'status.json',dict(state='transferring',cycle=cycle,files=len(rows),bytes=sum(x['size'] for x in rows),stage='priority0_models_configs'))
   print('INVENTORY',len(rows),sum(x['size'] for x in rows),flush=True)
   for priority in [0,1,2]:
    selected=[x for x in rows if x['priority']==priority];names=D/f'files_priority{priority}.nul';names.write_bytes(b''.join(x['path'].encode()+b'\0' for x in selected))
    cmd=['rsync','-az','--timeout=120','--partial-dir=.rsync-partial','--from0','--files-from='+str(names),'-e',' '.join(SSH),HOST+':/root/autodl-tmp/',str(D/'raw')+'/']
    with (D/f'transfer_{cycle:03d}_priority{priority}.log').open('w') as out:rc=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,timeout=7200).returncode
    write(D/'status.json',dict(state='transferring' if rc==0 else 'transfer_failed',cycle=cycle,priority_completed=priority,returncode=rc,files=len(rows),bytes=sum(x['size'] for x in rows)))
    print('TRANSFER',priority,rc,flush=True)
    if rc:return
   def check(x):
    local=D/'raw'/x['path'];return dict(path=x['path'],expected_sha256=x['sha256'],actual_sha256=digest(local) if local.exists() else None,stable_at_remote_scan=x['stable_at_scan'])
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:checks=list(pool.map(check,rows))
   bad=[x for x in checks if x['actual_sha256']!=x['expected_sha256']]
   write(D/f'verification_{cycle:03d}.json',dict(files=len(checks),matched=len(checks)-len(bad),mismatches=bad,checks=checks))
   (D/'SHA256SUMS').write_text(''.join(x['expected_sha256']+'  raw/'+x['path']+'\n' for x in checks))
   std=inv['standard_status'];ready=len(std.get('done',[]))==21 and not std.get('active') and not std.get('waiting') and not std.get('blocked')
   state='snapshot_verified' if not bad else 'snapshot_with_changing_files'
   # The final scan occurs only after training and both evaluation queues settle.
   hard=inv['hard_status'];ext=inv['extended_status']
   complete=ready and len(hard.get('done',[]))==18 and len(ext.get('done',[]))==63 and not bad
   write(D/'status.json',dict(state='complete_verified' if complete else state,cycle=cycle,files=len(checks),matched=len(checks)-len(bad),mismatches=[x['path'] for x in bad],standard_completed=len(std.get('done',[])),waiting=std.get('waiting',[]),final_complete=complete,next_scan_seconds=None if complete else 300,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
   print('VERIFIED',len(checks)-len(bad),'/',len(checks),'final',complete,flush=True)
   if complete:return
   time.sleep(300)
 write(D/'watch_deadline.json',dict(reason='3h bounded watcher deadline; inspect status before declaring full backup'))
if __name__=='__main__':main()
