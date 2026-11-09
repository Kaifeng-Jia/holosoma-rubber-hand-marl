#!/usr/bin/env python3
"""Launch an explicitly selected fresh pair without changing research code."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

OPS = Path('/root/autodl-tmp/paper_supplement_20260929_ops')
ROOT = Path('/root/autodl-tmp/paper_supplement_20260929')
PY = '/root/autodl-tmp/core4d_deps/hssim/bin/python'
SLOTS = ((0,2,722),(1,3,723))
CASES = {
    'bucket-a-resume': dict(experiment='bucket003', reward='bucket_A', prefix='BA_resume6000',
                     state='bucket_a_resume_20261003', expected='bucket_a_expected_20261002.json',
                     date='20261003', start=6000, slots=((0,0,722),(1,1,723))),
    'table-a': dict(experiment='smalltable5kg_A', reward='bucket_A', prefix='T06',
                     state='table_a_pair_20261003', expected='table_a_expected_20261003.json',
                     date='20261003', slots=((0,2,722),(1,3,723))),
    'bucket-a': dict(experiment='bucket003', reward='bucket_A', prefix='BA',
                     state='bucket_a_pair_20261002', expected='bucket_a_expected_20261002.json',
                     date='20261002', slots=((0,0,722),(1,1,723))),
    'chair': dict(experiment='chair021', reward='baseline', prefix='C01', state='chair_pair_20260930',
                  expected='chair_expected_20260930.json', slots=SLOTS),
    'bucket-rz0': dict(experiment='bucket003', reward='bucket_A_no_rel_no_height', prefix='BRZ0',
                       state='bucket_rz0_pair_20260930', expected='bucket_rz0_expected_20260930.json',
                       slots=((0,0,722),(1,1,723))),
}

def resume_path(seed):
    return ROOT/'logs/PaperSupplement20260929'/f'BA_seed{seed}_fresh12000_concurrent_20261002'/'model_06000.pt'

RESUME_HASHES={722:'e3745d7f92ad8498a5e40e3d2dcbde9636f4d3998dfa884ff7cec38c9a817379',
               723:'ef1f100aad1954c8a889b7ada65aef63b088fd21d3851274d73c680d6874321e'}

def command(seed,output,case='chair'):
    selected=CASES[case]
    argv = [PY,'-B',str(ROOT/'scripts/train_core4d_smalltable.py'),
            '--experiment',selected['experiment'],'--reward-variant',selected['reward'],'--num-envs','2048',
            '--steps-per-env','24','--seed',str(seed),'--iterations',str(12000-selected.get('start',0)),
            '--save-interval','2000','--output-dir',str(output)]
    return argv+(['--resume',str(resume_path(seed))] if selected.get('start') else [])

def check_config(cfg,expected,seed,case='chair'):
    selected=CASES[case]
    contract=dict(expected,initialization='fresh',start_iteration=0,iterations_requested=12000,
                  seed=seed,num_envs=2048,steps_per_env=24,save_interval=2000,
                  experiment=selected['experiment'],reward_variant=selected['reward'],object_z_error_weight=1.0,
                  interaction_contract=None,resume_checkpoint=None)
    if case=='chair': contract['bucket_reward_contract']=None
    if selected.get('start'):
        contract.update(initialization='resumed',start_iteration=6000,iterations_requested=6000,
                        resume_checkpoint=str(resume_path(seed)),resume_checkpoint_sha256=RESUME_HASHES[seed])
    for k,v in contract.items():
        if cfg.get(k)!=v:
            raise RuntimeError(f'{case} config mismatch: {k}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--case',choices=tuple(CASES),default='chair')
    args=parser.parse_args()
    selected=CASES[args.case]
    if not args.execute:
        print(json.dumps({'case':args.case,'slots':selected['slots'],'fresh_iterations':12000,'envs':2048,'save_interval':2000}))
        return
    spec=importlib.util.spec_from_file_location('deployment',OPS/'autodl_supplement_autorun.py')
    h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    statefile=OPS/(selected['state']+'.status.json')
    with (OPS/(selected['state']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h.require(not statefile.exists(),'This pair already attempted; no automatic duplicate/retry')
        h.stop_requested(); h.verify_source()
        expected_record=h.read(OPS/selected['expected'])
        h.require(expected_record['baseline_reward_assignment_ast_equal'],'Baseline reward assignment changed')
        expected=expected_record['expected']
        if args.case in ('bucket-a-resume','table-a'):
            active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                       '--format=csv,noheader,nounits'],text=True).split())
            allowed=set()
            if args.case=='bucket-a-resume':
                for seed in (722,723):
                    h.require(h.digest(resume_path(seed))==RESUME_HASHES[seed],'Recovery checkpoint changed')
            else:
                recovery=h.read(OPS/'bucket_a_resume_20261003.status.json')
                h.require(recovery['state'] in ('starting','training','complete'),'Bucket recovery not healthy')
                allowed={str(r['pid']) for r in recovery.get('runs',[]) if r.get('cache_slot') in (0,1)}
            h.require(active<=allowed,'Unexpected GPU process; respect four-job limit')
        if args.case=='bucket-a':
            plan=h.read(OPS/'plan5_batch_20261001.status.json')
            for seed in (722,723):
                matches=[r for r in plan['finished'] if r['skill']=='pull' and r['seed']==seed and r['stage']==1]
                h.require(len(matches)==1 and matches[0]['validated'] and matches[0]['process_returncode']==0,'Pull slot not released')
            remaining=plan.get('runs',[])
            h.require(all(r['skill']=='push' and r['cache_slot'] in (2,3) for r in remaining),'Unexpected active slot')
            allowed={str(r['pid']) for r in remaining}
            active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                       '--format=csv,noheader,nounits'],text=True).split())
            h.require(active<=allowed,'Unknown GPU processes; do not exceed four jobs')
        if args.case=='bucket-rz0':
            prior=h.read(OPS/'resume_t07_20260930.status.json')
            h.require(prior['state']=='complete' and len(prior['runs'])==2,'Table pair not completed')
            for run in prior['runs']:
                h.require(run['validated'] and run['process_returncode']==0,'Invalid table completion')
                h.require(h.digest(Path(run['final_checkpoint']))==run['sha256'],'Table checkpoint changed')
            chairs=h.read(OPS/'chair_pair_20260930.status.json')
            allowed={str(r['pid']) for r in chairs.get('runs',[])}
            active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                         '--format=csv,noheader,nounits'],text=True).split())
            h.require(active<=allowed,'Unexpected GPU process; do not exceed authorized slots')
        jobs=[]; results=[]
        h.save(statefile,{'state':'launching','utc':h.now(),'supervisor_pid':os.getpid()})
        try:
            for gpu,slot,seed in selected['slots']:
                h.stop_requested()
                label=f'{selected["prefix"]}_seed{seed}_fresh12000_concurrent_{selected.get("date","20260930")}'
                out=ROOT/'logs/PaperSupplement20260929'/label
                h.require(not out.exists() and not (OPS/(label+'.launch.json')).exists(),'Duplicate run identity')
                env=h.environment(gpu); env['LOCAL_RANK']=str(slot)
                argv=command(seed,out,args.case)
                with (OPS/(label+'.stdout.log')).open('xb') as log:
                    proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record={'label':label,'seed':seed,'gpu':gpu,'cache_slot':slot,'pid':proc.pid,
                        'argv':argv,'output':str(out),'started_utc':h.now(),
                        'environment_overrides':{'CUDA_VISIBLE_DEVICES':str(gpu),'LOCAL_RANK':str(slot),'WORLD_SIZE':'1'},
                        'budget':12000-selected.get('start',0),'start_iteration':selected.get('start',0),
                        'num_envs':2048,'save_interval':2000,'fresh':not bool(selected.get('start')),'no_auto_retry':True}
                if selected.get('start'):
                    record.update(parent_checkpoint=str(resume_path(seed)),parent_sha256=RESUME_HASHES[seed],
                                  resume_limit='Models, Adam and normalizers restored; simulator/RNG not saved; not bitwise continuation')
                h.save(OPS/(label+'.launch.json'),record)
                jobs.append({'proc':proc,'record':record,'start':time.monotonic(),'checked':False})
            h.save(statefile,{'state':'starting','utc':h.now(),'runs':[j['record'] for j in jobs]})
            while jobs:
                for job in jobs[:]:
                    proc=job['proc']; record=job['record']; out=Path(record['output']); failure=None
                    try:
                        if not job['checked'] and (out/'run_config.json').exists():
                            check_config(h.read(out/'run_config.json'),expected,record['seed'],args.case)
                            job['checked']=True
                        if time.monotonic()-job['start']>48*3600:
                            raise RuntimeError('48-hour process limit exceeded')
                        if not job['checked'] and time.monotonic()-job['start']>2700:
                            raise RuntimeError('No validated run configuration within 45 minutes')
                    except Exception as exc:
                        failure=repr(exc)
                        if proc.poll() is None:
                            os.killpg(proc.pid,signal.SIGTERM)
                            try: proc.wait(timeout=30)
                            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
                    if proc.poll() is not None:
                        result=dict(record,process_returncode=proc.wait(),ended_utc=h.now(),validated=False)
                        try:
                            h.require(failure is None,failure)
                            h.require(result['process_returncode']==0,'Nonzero process exit')
                            check_config(h.read(out/'run_config.json'),expected,record['seed'],args.case)
                            rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
                            h.require([r['iteration'] for r in rows]==list(range(selected.get('start',0)+1,12001)),'Incomplete iterations')
                            h.require(h.finite(rows),'Non-finite metrics')
                            if args.case in ('bucket-rz0','bucket-a','bucket-a-resume','table-a'):
                                h.require(all(r.get('Bucket/gate_mean')==1 for r in rows),'Unexpected contact gating')
                            status=h.read(out/'status.json')
                            h.require(status.get('passed') is True and status.get('final_iteration')==12000,'No valid final status')
                            model=out/'model_12000.pt'; h.require(model.stat().st_size>0,'Empty final model')
                            result.update(validated=True,sha256=h.digest(model),final_checkpoint=str(model))
                        except Exception as exc: result['validation_error']=repr(exc)
                        h.save(OPS/(record['label']+'.exit.json'),result)
                        results.append(result);jobs.remove(job)
                if jobs:
                    h.save(statefile,{'state':'training' if all(j['checked'] for j in jobs) else 'starting',
                                      'utc':h.now(),'runs':[j['record'] for j in jobs], 'finished':results,
                                      'supervisor_pid':os.getpid(),'additional_experiments_armed':False})
                    time.sleep(10)
            h.save(statefile,{'state':'complete' if all(r['validated'] for r in results) else 'blocked',
                              'utc':h.now(),'runs':results,'additional_experiments_armed':False})
        except Exception as exc:
            # Only our own children are stopped; other experiments remain untouched.
            for job in jobs:
                proc=job['proc']
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                h.save(OPS/(job['record']['label']+'.exit.json'),dict(job['record'],
                    process_returncode=proc.returncode,ended_utc=h.now(),validated=False,validation_error=repr(exc)))
            h.save(statefile,{'state':'blocked','utc':h.now(),'error':repr(exc)})
            raise

if __name__=='__main__':
    main()
