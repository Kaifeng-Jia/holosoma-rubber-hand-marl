#!/usr/bin/env python3
"""Four independent staged Plan5 runs; deployment only, no research edits."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OPS=Path('/root/autodl-tmp/paper_supplement_20260929_ops')
ROOT=Path('/root/autodl-tmp/paper_supplement_20260929')
PY='/root/autodl-tmp/core4d_deps/hssim/bin/python'
NAME='plan5_batch_20261001'
SLOTS=[('pull',0,0,722),('pull',1,1,723),('push',0,2,722),('push',1,3,723)]

def command(r):
    argv=[PY,'-B',str(ROOT/'scripts/train_plan5_push.py'),'--skill',r['skill'],
          '--seed',str(r['seed']),'--num-envs','2048','--steps-per-env','24',
          '--iterations',str(r['budget']),'--save-interval','1000',
          '--actor-learning-rate','0.00001','--critic-learning-rate','0.001',
          '--output-dir',r['output']]
    return argv+(['--critic-only'] if r['start_iteration']==0 else ['--resume',r['parent_checkpoint']])

def check_config(cfg,expected,r):
    warm=r['start_iteration']==0
    contract=dict(expected,seed=r['seed'],start_iteration=r['start_iteration'],
                  iterations_requested=r['budget'],num_envs=2048,steps_per_env=24,
                  critic_only=warm,teammate_input_only=False,
                  actor_update_mode='critic_only' if warm else 'full',
                  optimizer_state_restored=not warm,actor_learning_rate=1e-5,
                  critic_learning_rate=0.001,fine_tune_checkpoint=None,
                  joint_acceleration_schedule=None)
    for k,v in contract.items():
        if cfg.get(k)!=v:raise RuntimeError(f'Config mismatch {k}: {cfg.get(k)!r} != {v!r}')
    runtime=cfg['runtime_process']
    assert runtime['distributed_data_parallel'] is False
    assert runtime['cuda_visible_devices']==str(r['gpu'])
    assert runtime['local_rank']==str(r['cache_slot'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    if not p.parse_args().execute:
        print(json.dumps({'slots':SLOTS,'pull_final':8050,'push_final':13050}));return
    spec=importlib.util.spec_from_file_location('deployment',OPS/'autodl_supplement_autorun.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    statefile=OPS/(NAME+'.status.json')
    with (OPS/(NAME+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.require(not statefile.exists(),'No duplicate batch or implicit restart')
        h.stop_requested();h.verify_source()
        cases=h.read(OPS/'plan5_expected_20261001.json')['cases']
        for c in cases.values():
            for path,sha in c['assets'].items():h.require(h.digest(ROOT/path)==sha,'Asset changed: '+path)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        h.require(not active.strip(),'Expected idle GPUs before four-job batch')
        jobs=[];finished=[];failed=[]
        def save(state):
            h.save(statefile,dict(state=state,utc=h.now(),supervisor_pid=os.getpid(),
                   runs=[j['record'] for j in jobs],finished=finished,failed=failed,
                   additional_experiments_armed=False))
        def launch(skill,gpu,slot,seed,stage,parent=None):
            h.stop_requested();start,count=cases[skill]['stages'][stage]
            label=f'{skill.capitalize()}_seed{seed}_stage{stage}_from{start}_plus{count}_20261001'
            out=ROOT/'logs/PaperSupplement20260929'/label
            h.require(not out.exists() and not (OPS/(label+'.launch.json')).exists(),'Duplicate stage')
            env=h.environment(gpu);env['LOCAL_RANK']=str(slot)
            r=dict(label=label,skill=skill,gpu=gpu,cache_slot=slot,seed=seed,stage=stage,
                   start_iteration=start,budget=count,output=str(out),started_utc=h.now(),
                   parent_checkpoint=parent,parent_sha256=h.digest(Path(parent)) if parent else None)
            r['argv']=command(r)
            with (OPS/(label+'.stdout.log')).open('xb') as log:
                proc=subprocess.Popen(r['argv'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            r['pid']=proc.pid;jobs.append(dict(proc=proc,record=r,start=time.monotonic(),checked=False))
            h.save(OPS/(label+'.launch.json'),r)
        def stop(job):
            proc=job['proc']
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        save('launching')
        try:
            for skill,gpu,slot,seed in SLOTS:launch(skill,gpu,slot,seed,0)
            while jobs:
                for job in jobs[:]:
                    r=job['record'];out=Path(r['output']);proc=job['proc'];error=None
                    try:
                        if not job['checked'] and (out/'run_config.json').exists():
                            check_config(h.read(out/'run_config.json'),cases[r['skill']]['expected'],r);job['checked']=True
                        elapsed=time.monotonic()-job['start']
                        if elapsed>96*3600:raise RuntimeError('Stage exceeded 96 hours')
                        if not job['checked'] and elapsed>2700:raise RuntimeError('No config within 45 minutes')
                    except Exception as exc:error=repr(exc);stop(job)
                    if proc.poll() is None:continue
                    result=dict(r,process_returncode=proc.wait(),ended_utc=h.now(),validated=False)
                    try:
                        h.require(error is None,error);h.require(result['process_returncode']==0,'Nonzero exit')
                        check_config(h.read(out/'run_config.json'),cases[r['skill']]['expected'],r)
                        end=r['start_iteration']+r['budget']
                        rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
                        h.require([x['iteration'] for x in rows]==list(range(r['start_iteration']+1,end+1)),'Incomplete iterations')
                        h.require(h.finite(rows),'Non-finite metrics')
                        status=h.read(out/'status.json')
                        h.require(status.get('passed') is True and status.get('final_iteration')==end,'Missing final status')
                        model=out/f'model_{end:05d}.pt';h.require(model.stat().st_size>0,'Empty checkpoint')
                        result.update(validated=True,final_checkpoint=str(model),sha256=h.digest(model))
                    except Exception as exc:result['validation_error']=repr(exc)
                    h.save(OPS/(r['label']+'.exit.json'),result);finished.append(result);jobs.remove(job)
                    if not result['validated']:failed.append(result)
                    elif r['stage']+1<len(cases[r['skill']]['stages']):
                        launch(r['skill'],r['gpu'],r['cache_slot'],r['seed'],r['stage']+1,result['final_checkpoint'])
                save('training' if jobs else 'blocked' if failed else 'complete')
                if jobs:time.sleep(10)
        except Exception as exc:
            for job in jobs:stop(job)
            failed.append({'error':repr(exc)});save('blocked');raise

if __name__=='__main__':main()
