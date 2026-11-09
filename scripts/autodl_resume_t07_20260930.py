#!/usr/bin/env python3
"""Recover the two interrupted T07 runs only; never launch new experiment seeds."""
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
EXPECTED = {722: 'bee3587550e8ab546b21e895a4fc8ec65f6bb1eb174be36df1ddd7b4e81de5e9',
            723: 'd8f024a27892a99770accad2614a15756249d8e37126ca309abc5b37a11a41e8'}

def load_helpers():
    spec = importlib.util.spec_from_file_location('deployment', OPS/'autodl_supplement_autorun.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'seeds': [722,723], 'resume_iteration':10000, 'additional_iterations':2000,
                          'num_envs':2048, 'steps_per_env':24, 'save_interval':2000,
                          'preflight': '2 resumed iterations per GPU, discarded', 'new_experiments':False}))
        return
    h = load_helpers()
    statefile = OPS/'resume_t07_20260930.status.json'
    with (OPS/'autorun.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h.require(not statefile.exists(), 'Recovery already attempted; no automatic retry')
        h.stop_requested()
        h.save(statefile, {'state':'verifying', 'utc':h.now(), 'supervisor_pid':os.getpid()})
        try:
            h.verify_source()
            active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                              '--format=csv,noheader,nounits'],text=True)
            h.require(not active.strip(), 'Compute processes already active')
            bases = {}
            for seed in EXPECTED:
                base = ROOT/'logs/PaperSupplement20260929'/f'T07_seed{seed}_fresh12000_cache_isolated02'
                h.require(h.digest(base/'model_10000.pt') == EXPECTED[seed], 'Resume hash mismatch')
                bases[seed] = base

            def launch(gpu, seed, preflight):
                h.stop_requested()
                n = 2 if preflight else 2000
                label = f'T07_seed{seed}_resume10000_{"preflight2" if preflight else "to12000"}_clone20260930'
                output = ROOT/'logs/PaperSupplement20260929'/label
                h.require(not output.exists() and not (OPS/(label+'.launch.json')).exists(), 'Duplicate output')
                argv = h.command(output,seed,True)
                argv[argv.index('--iterations')+1] = str(n)
                argv += ['--resume',str(bases[seed]/'model_10000.pt')]
                with (OPS/(label+'.stdout.log')).open('xb') as log:
                    proc = subprocess.Popen(argv,cwd=ROOT,env=h.environment(gpu),stdout=log,
                                            stderr=subprocess.STDOUT,start_new_session=True)
                record = {'label':label,'gpu':gpu,'seed':seed,'pid':proc.pid,'argv':argv,
                          'output':str(output),'started_utc':h.now(),'preflight':preflight,
                          'start_iteration':10000,'additional_iterations':n,
                          'parent_sha256':EXPECTED[seed], 'interrupted_parent':str(bases[seed]),
                          'not_a_new_training_seed':True,
                          'resume_limit':'Models, Adam states and normalizers restored; simulator/RNG state not saved. Not bitwise continuation.',
                          'environment_overrides':{'CUDA_VISIBLE_DEVICES':str(gpu),'LOCAL_RANK':str(gpu),'WORLD_SIZE':'1'}}
                h.save(OPS/(label+'.launch.json'),record)
                return proc,record,time.monotonic()

            def finish(job,timeout=False):
                proc,record,_ = job
                if timeout and proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL)
                result = dict(record,process_returncode=proc.wait(),timed_out=timeout,ended_utc=h.now(),validated=False)
                try:
                    h.require(result['process_returncode']==0 and not timeout,'Process failed or timed out')
                    out = Path(record['output'])
                    cfg = h.read(out/'run_config.json')
                    expected = dict(initialization='resumed',start_iteration=10000,
                                    iterations_requested=record['additional_iterations'],seed=record['seed'],
                                    resume_checkpoint_sha256=record['parent_sha256'],num_envs=2048,steps_per_env=24,
                                    save_interval=2000,actor_input_dim=158,critic_input_dim=527,
                                    reward_variant=h.VARIANT,experiment='smalltable5kg_A')
                    for k,v in expected.items(): h.require(cfg.get(k)==v,f'Config mismatch: {k}')
                    parent = h.read(bases[record['seed']]/'run_config.json')
                    for k in ('object_physics','physics_hz','control_hz','reward_terms','termination_terms',
                              'runtime_reference_sha256','object_urdf_sha256','training_robot_urdf_sha256',
                              'bucket_reward_contract','experiment_contract','object_z_error_weight'):
                        h.require(cfg.get(k)==parent.get(k),f'Parent contract changed: {k}')
                    rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
                    end=10000+record['additional_iterations']
                    h.require([r['iteration'] for r in rows]==list(range(10001,end+1)),'Iteration gap')
                    h.require(h.finite(rows),'Non-finite metrics')
                    h.require(all(r.get('Bucket/gate_mean')==1 for r in rows),'Unexpected gate')
                    status=h.read(out/'status.json')
                    h.require(status.get('passed') is True and status.get('final_iteration')==end,'Incomplete result')
                    model=out/f'model_{end:05d}.pt'
                    h.require(model.stat().st_size>0,'Empty checkpoint')
                    result.update(validated=True,final_iteration=end,final_checkpoint=str(model),sha256=h.digest(model))
                except Exception as exc: result['validation_error']=repr(exc)
                h.save(OPS/(record['label']+'.exit.json'),result)
                return result

            h.save(statefile,{'state':'preflight','utc':h.now(),'supervisor_pid':os.getpid()})
            for gpu,seed in ((0,722),(1,723)):
                job=launch(gpu,seed,True)
                try:
                    job[0].wait(timeout=2700)
                    result=finish(job)
                except subprocess.TimeoutExpired: result=finish(job,True)
                h.require(result['validated'],f'Preflight failed: {result}')
            h.verify_source()
            jobs=[]
            try:
                for gpu,seed in ((0,722),(1,723)): jobs.append(launch(gpu,seed,False))
            except Exception:
                for job in jobs: finish(job,True)
                raise
            h.save(statefile,{'state':'training','utc':h.now(),'supervisor_pid':os.getpid(),
                              'runs':[j[1] for j in jobs], 'next_batch_armed':False})
            results=[]
            while jobs:
                for job in jobs[:]:
                    timeout=time.monotonic()-job[2]>24*3600
                    if job[0].poll() is not None or timeout:
                        results.append(finish(job,timeout)); jobs.remove(job)
                if jobs: time.sleep(10)
            h.require(all(r['validated'] for r in results),'Recovery segment failed; see exit records')
            h.save(statefile,{'state':'complete','utc':h.now(),'runs':results,'next_batch_armed':False})
        except Exception as exc:
            h.save(statefile,{'state':'blocked','utc':h.now(),'error':repr(exc)})
            raise

if __name__ == '__main__':
    main()
