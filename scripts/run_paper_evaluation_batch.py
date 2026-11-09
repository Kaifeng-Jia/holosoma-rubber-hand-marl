#!/usr/bin/env python3
"""Local serial evaluation queue with actual exit records and resumable model jobs."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from paper_evaluation_utils import sha,write_json,evaluation_result_dir
from summarize_paper_evaluation import summarize

ROOT=Path(__file__).resolve().parents[1]
PY='/home/kevin/.holosoma_deps/miniconda3/envs/hssim/bin/python'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root.resolve()
    with (out/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        models=json.loads((out/'models.json').read_text())
        # New independently trained models first; historical candidates follow.
        models.sort(key=lambda m:(m['training_seed']==721,m['phase']!='main',m['case'],m['training_seed']))
        (out/'results').mkdir(exist_ok=True);failures=[]
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src/holosoma')+':'+str(ROOT/'src/holosoma_retargeting'),
                 PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',WORLD_SIZE='1',LOCAL_RANK='0',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for m in models:
            dest=evaluation_result_dir(out,m);exitfile=out/(m['model_id']+'.exit.json')
            if exitfile.exists():
                record=json.loads(exitfile.read_text())
                if record.get('validated'):continue
                failures.append(record);continue
            if dest.exists():
                failures.append(dict(model_id=m['model_id'],error='Partial output retained; explicit retry needs a new directory'));continue
            assert sha(m['checkpoint'])==m['checkpoint_sha256']
            argv=['taskset','-c','0-7',PY,'-B',str(ROOT/'scripts/evaluate_paper_supplement.py'),
                  '--case',m['case'],'--model-id',m['model_id'],'--checkpoint',m['checkpoint'],
                  '--states',str(out/(m['case']+'_states.json')),'--output-dir',str(dest)]
            now=lambda:dt.datetime.now(dt.timezone.utc).isoformat()
            record=dict(model_id=m['model_id'],argv=argv,started_utc=now(),checkpoint_sha256=m['checkpoint_sha256'],
                        evaluation_directory=str(dest.relative_to(out)),attempt_id=dest.name)
            with (out/(dest.name+'.stdout.log')).open('xb') as log:
                proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write_json(out/'batch_status.json',dict(state='evaluating',current_model=m['model_id'],pid=proc.pid,
                           supervisor_pid=os.getpid(),started_utc=record['started_utc'],failures=failures))
                rc=proc.wait()
            record.update(process_returncode=rc,ended_utc=now(),validated=False)
            try:
                if rc!=0:raise RuntimeError(f'Process exit {rc}')
                s=json.loads((dest/'summary.json').read_text())
                assert s['complete'] and s['episode_count']==51 and s['checkpoint_sha256']==m['checkpoint_sha256']
                assert len({r['state_id'] for r in s['episodes']})==51
                record['validated']=True
            except Exception as exc:record['error']=repr(exc);failures.append(record)
            write_json(exitfile,record);summarize(out)
        n=summarize(out)
        write_json(out/'batch_status.json',dict(state='complete' if not failures and n==21 else 'needs_review',
                   evaluated_models=n,target_models=21,failures=failures,ended_utc=dt.datetime.now(dt.timezone.utc).isoformat()))

if __name__=='__main__':main()
