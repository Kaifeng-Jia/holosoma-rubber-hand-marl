#!/usr/bin/env python3
"""Bounded local collection after the authorized R2 cloud evaluation finishes.

Does not train, delete remote files, stop the instance, or push Git commits.
Requires this Ubuntu machine to remain on and the authorized SSH session alive.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/r2_20261024/logs/ExperimentsR2_evaluation_20261026_retry01'
REMOTE_OPS='/root/autodl-tmp/r2_eval_20261026_ops'
DEST=ROOT/'logs/ExperimentsR2_evaluation_20261026_retry01'
OPS=ROOT/'logs/ExperimentsR2/deployment_20261024/evaluation_20261026'
SSH=['ssh','-S','/tmp/core4d_37638_control','-o','UserKnownHostsFile=/tmp/core4d_cloud_37638_known_hosts',
     '-o','StrictHostKeyChecking=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','37638','root@connect.nmb2.seetacloud.com']
RPY='/root/autodl-tmp/r2_deps/hssim/bin/python'

def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def remote_python(code):
    return subprocess.check_output(SSH+[RPY+' -B -c '+shlex.quote(code)],text=True,timeout=120)
def sync(remote,dest):
    dest.mkdir(parents=True,exist_ok=True)
    transport=shlex.join(SSH[:-1])
    subprocess.run(['rsync','-az','--partial','--stats','-e',transport,
        SSH[-1]+':'+remote+'/',str(dest)+'/'],check=True,timeout=1800)

def run():
    import fcntl
    OPS.mkdir(parents=True,exist_ok=True)
    with (OPS/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.time()+4*3600
        while True:
            if time.time()>deadline:raise TimeoutError('Four-hour collection window expired; inspect manually')
            try:
                s=json.loads(remote_python(f'from pathlib import Path; print(Path({REMOTE!r}+"/batch_status.json").read_text())'))
            except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
                save(OPS/'collector_status.json',dict(phase='waiting_for_connection',error=str(exc),updated_unix=time.time()))
                time.sleep(30);continue
            save(OPS/'collector_status.json',dict(phase='waiting_for_evaluation',remote_status=s,updated_unix=time.time()))
            if s['phase'] in ('complete','needs_review','blocked'):break
            time.sleep(30)
        # Hash stable completed artifacts before collection, including failure evidence.
        code='''import json,hashlib
from pathlib import Path
rows=[]
for group,root in ROOTS.items():
 for p in sorted(Path(root).rglob('*')):
  if p.is_symlink(): raise RuntimeError('Unexpected symlink: '+str(p))
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   rows.append(dict(group=group,path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(rows))
'''
        rows=json.loads(remote_python('ROOTS='+repr(dict(evaluation=REMOTE,ops=REMOTE_OPS))+'\n'+code))
        save(OPS/'remote_inventory.json',rows)
        sync(REMOTE,DEST);sync(REMOTE_OPS,OPS/'cloud_ops')
        roots=dict(evaluation=DEST,ops=OPS/'cloud_ops')
        for r in rows:
            p=roots[r['group']]/r['path']
            if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise RuntimeError('Transfer mismatch: '+str(p))
        save(OPS/'transfer_verified.json',dict(verified=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),time=time.time()))
        if s['phase']!='complete':raise RuntimeError('Evaluation not complete; evidence collected for review')
        models=read(DEST/'models.json');replays=[];checkpoints=[]
        for m in models:
            if m['case']=='kick' and m['training_seed']==721:
                cp=Path('/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/model_08000.pt')
            else:
                cp=ROOT/'logs/ExperimentsR2_retry1'/Path(m['checkpoint']).parent.name/Path(m['checkpoint']).name
            if sha(cp)!=m['checkpoint_sha256']:raise RuntimeError('Local model hash mismatch')
            portable=DEST/m['group']/'checkpoints'/f'{m["model_id"]}.pt'
            portable.parent.mkdir(exist_ok=True);shutil.copy2(cp,portable)
            checkpoints.append(dict(model_id=m['model_id'],local_original=str(cp),portable=str(portable.relative_to(DEST)),sha256=sha(portable)))
            group=DEST/m['group'];folder=group/'results'/m['model_id'];summary=read(folder/'summary.json')
            exit_record=read(group/(m['model_id']+'.exit.json'))
            if not (exit_record['validated'] and exit_record['process_returncode']==0 and summary['episode_count']==51):
                raise RuntimeError('Incomplete result '+m['model_id'])
            # Preserve original absolute trajectory fields; provide an explicit local map.
            for e in summary['episodes']:
                replays.append(dict(model_id=m['model_id'],state_id=e['state_id'],original=e['trajectory'],
                    local_relative=str((folder/Path(e['trajectory']).name).relative_to(DEST))))
        save(DEST/'local_checkpoint_map.json',checkpoints);save(DEST/'local_trajectory_map.json',replays)
        if len(replays)!=306:raise RuntimeError('Expected 306 actual trajectories')
        save(OPS/'collector_status.json',dict(phase='rendering_nominal_replays',updated_unix=time.time(),trajectories=306))
        py='/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python'
        ffmpeg='/home/kevin/.holosoma_deps/miniconda3/envs/hssim/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
        for m in models:
            folder=DEST/m['group']/'results'/m['model_id'];manifest=read(folder/'run_manifest.json')
            media=DEST/'media'/m['model_id'];media.parent.mkdir(exist_ok=True)
            # Original visual robot/object meshes, recorded physical states only.
            argv=[py,'-B',str(ROOT/'scripts/export_recorded_demo_media.py'),
                '--rollout',str(folder/'nominal_viser.npz'),
                '--object-urdf',str(ROOT/'src/holosoma'/manifest['object_urdf']),
                '--robot-urdf',str(ROOT/'src/holosoma/holosoma/data/robots'/manifest['robot_urdf']),
                '--output',str(media),'--title',m['model_id'],'--checkpoint-label',str(m['checkpoint_iteration']),
                '--ffmpeg',ffmpeg,'--width','1920','--height','1080']
            with (OPS/(m['model_id']+'.render.log')).open('xb') as log:
                r=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1200,
                    env=dict(os.environ,MUJOCO_GL='egl',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'))
            save(OPS/(m['model_id']+'.render.exit.json'),dict(argv=argv,process_returncode=r.returncode))
            if r.returncode:raise RuntimeError('Render failed: '+m['model_id'])
        save(OPS/'collector_status.json',dict(phase='complete',updated_unix=time.time(),
             trajectories=306,nominal_videos=6,evaluation_root=str(DEST),
             remaining='Review videos; append U3 tracking and remaining paper U-items separately; cloud instance not stopped'))

if __name__=='__main__':
    try:run()
    except BaseException as exc:
        save(OPS/'collector_failure.json',dict(error=repr(exc),updated_unix=time.time()))
        raise
