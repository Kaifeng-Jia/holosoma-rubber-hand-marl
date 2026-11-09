#!/usr/bin/env python3
"""R2 only: six fixed endpoints, paired 51-state evaluation, two isolated GPUs."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,x):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n'); tmp.replace(p)
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def csv_save(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def aggregate(root):
    import numpy as np
    models=read(root/'models.json'); episodes=[]; nominal=[]; per=[]
    for m in models:
        ep=root/(m['model_id']+'.exit.json')
        if not ep.exists() or not read(ep)['validated']:continue
        rows=read(root/'results'/m['model_id']/'summary.json')['episodes']
        rs=[]
        for raw in rows:
            r=dict(raw,configuration=m['configuration'],training_seed=m['training_seed'],
                   training_reuse_status=m['reuse_status'],checkpoint_sha256=m['checkpoint_sha256'])
            r['terminal_terms']=json.dumps(r['terminal_terms'],sort_keys=True)
            (nominal if r['kind']=='nominal' else episodes).append(r)
            if r['kind']=='perturbed':rs.append(r)
        done=[r for r in rs if r['completed_reference']]
        per.append(dict(model_id=m['model_id'],configuration=m['configuration'],training_seed=m['training_seed'],
            reuse_status=m['reuse_status'],episodes=len(rs),completion_rate=len(done)/len(rs),
            progress_mean=float(np.mean([r['progress'] for r in rs])),
            completed_object_rmse_3d_m=float(np.mean([r['object_rmse_3d_m'] for r in done])) if done else None,
            completed_episodes=len(done),prefix_object_rmse_3d_m=float(np.mean([r['object_rmse_3d_m'] for r in rs]))))
    csv_save(root/'episodes.csv',episodes);csv_save(root/'nominal_results.csv',nominal)
    csv_save(root/'per_seed_summary.csv',per)
    tables=[]
    for conf in dict.fromkeys(m['configuration'] for m in models):
        valid=[r for r in per if r['configuration']==conf]
        row=dict(configuration=conf,training_seeds_evaluated=len(valid),target_training_seeds=3,
                 complete_three_seed_result=len(valid)==3)
        for key in ('completion_rate','progress_mean','completed_object_rmse_3d_m'):
            values=[r[key] for r in valid if r[key] is not None]
            publish=len(values)==3 if key.startswith('completed_') else bool(values)
            row[key+'_valid_seeds']=len(values)
            row[key+'_mean']=float(np.mean(values)) if publish else None
            row[key+'_sample_sd']=float(np.std(values,ddof=1)) if publish and len(values)>1 else None
        tables.append(row)
    csv_save(root/'main_results.csv',tables)
    save(root/'aggregation_status.json',dict(evaluated_models=len(per),target_models=len(models),
          perturbed_episodes=len(episodes),nominal_episodes=len(nominal),
          note='Historical Kick721 reused as requested; original process exit code not inferred. No nominal pooling.'))

def prepare(a):
    require(not a.output.exists(),'Choose a new evaluation output')
    allmodels=[]
    for skill,folder,conf,seeds in [('push','U5_push_registered','Push-registered',(721,722,723)),
                                  ('kick','U6_kick','Kick-baseline',(721,722,723))]:
        root=a.output/folder;root.mkdir(parents=True)
        states=a.project/'r2_evidence'/f'{skill}_states.json'
        expected={'push':'1799ea1eec1a3ff097d46e08e6492694d6455490ccf9f2de26586e079fef895d',
                  'kick':'c1ac57c62345363c85f96f98363d778898a23e1f35a2596fbcd33a68367527c4'}[skill]
        require(sha(states)==expected,'State hash mismatch')
        shutil.copy2(states,root/states.name)
        (root/'results').mkdir();models=[]
        for seed in seeds:
            if skill=='kick' and seed==721:
                cp=a.ops/'historical_kick721/model_08000.pt'; cfg=read(cp.parent/'run_config.json')
                for key,value in dict(skill='kick',seed=721,start_iteration=0,iterations_requested=8000,
                    actor_obs_dim=158,critic_obs_dim=527,num_envs=2048,steps_per_env=24,
                    source_checkpoint_sha256='1555968f678c2b69fcd6f09c64d0a6252683eab902edd773a84acc205d0f5491').items():
                    require(cfg.get(key)==value,'Historical config mismatch: '+key)
                reuse='historical_checkpoint_reused_config_verified'; training_exit=None
            else:
                stage='stage2_from8050_plus5000' if skill=='push' else 'stage0_from0_plus8000'
                ep=a.train_ops/f'{skill}_seed{seed}_{stage}.exit.json'; record=read(ep)
                require(record['validated'] and record['process_returncode']==0,'Training not verified')
                cp=Path(record['final_checkpoint']);require(sha(cp)==record['sha256'],'Checkpoint changed')
                reuse='new_training_verified'; training_exit=str(ep)
            require(cp.is_file(),'Missing checkpoint')
            m=dict(model_id=f'{conf}_seed{seed}',case=skill,configuration=conf,training_seed=seed,
                   reuse_status=reuse,checkpoint=str(cp),checkpoint_sha256=sha(cp),
                   checkpoint_iteration=13050 if skill=='push' else 8000,
                   training_exit_record=training_exit,group=folder)
            models.append(m);allmodels.append(m)
        save(root/'models.json',models)
    save(a.output/'models.json',allmodels)
    source=a.output/'sources';source.mkdir()
    files=[a.project/'scripts'/n for n in ('evaluate_paper_supplement.py','paper_evaluation_utils.py','plan5_registered_placement.py')]
    files += [Path(__file__).resolve(),a.project/'r2_evidence/preflight_push_registered.json']
    for p in files:shutil.copy2(p,source/p.name)
    save(a.output/'provenance.json',dict(scripts={str(p):sha(p) for p in files},
         target_models=6,episodes_per_model=51,actor='mean only',noise=False,
         geometry='Push original loader with registered spacing; Kick legacy mirrored runtime',
         historical_kick721_original_exit_code='unknown; new evaluation exit is explicitly recorded'))

def validate_result(a,m,dest,rc):
    import numpy as np
    require(rc==0,f'Process exit {rc}')
    summary=read(dest/'summary.json'); states=read(a.output/m['group']/f'{m["case"]}_states.json')
    require(summary['complete'] and summary['episode_count']==51,'Incomplete episodes')
    require(summary['checkpoint_sha256']==m['checkpoint_sha256'],'Checkpoint mismatch')
    require(summary['checkpoint_iteration']==m['checkpoint_iteration'],'Wrong fixed endpoint')
    require(summary['states_sha256']==sha(a.output/m['group']/f'{m["case"]}_states.json'),'States differ')
    require(summary['inference']=='actor_mean_only_critic_forbidden' and not summary['observation_noise'],'Inference changed')
    require([r['state_id'] for r in summary['episodes']]==[r['state_id'] for r in states['states']],'State order/inventory differs')
    require(sum(r['kind']=='perturbed' for r in summary['episodes'])==50,'Perturbed count')
    for r,init in zip(summary['episodes'],states['states']):
        require(r['kind']==init['kind'],'State kind mismatch')
        initial=read(dest/(r['state_id']+'.initial.json'))
        require(initial['valid'] and initial['requested']==init,'Reset mismatch')
        with np.load(dest/(r['state_id']+'.trajectory.npz'),allow_pickle=False) as z:
            require(all(np.isfinite(z[k]).all() for k in z.files),'Nonfinite trajectory')
            require(len(z['dof_pos'])==r['steps']+1,'Frame inventory mismatch')
            require(z['dof_pos'].shape[1:]==(2,29),'DOF shape mismatch')
    require((dest/'nominal_viser.npz').is_file(),'No nominal replay')

def execute(a):
    with (a.output/'batch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (a.output/'batch_status.json').exists(),'Refusing duplicate batch')
        # One model of each task first; 3 independent seeds per configuration.
        models=read(a.output/'models.json')
        pending=sorted(models,key=lambda m:(m['training_seed'],m['case']=='kick'))
        active=[]; finished=[]
        def publish(phase):
            rows=[]
            for j in active:
                r=dict(j['record']); ep=Path(r['output'])/'episodes.json'
                if ep.exists():
                    try:r['episodes_recorded']=len(read(ep))
                    except json.JSONDecodeError:pass
                rows.append(r)
            save(a.output/'batch_status.json',dict(phase=phase,updated_unix=time.time(),
                 supervisor_pid=os.getpid(),active=rows,pending=[m['model_id'] for m in pending],
                 finished=finished,target_models=len(models)))
        try:
            while pending or active:
                require(not (a.output/'STOP_EVALUATION').exists(),'Stop requested')
                used={j['record']['gpu'] for j in active}
                for gpu in (0,1):
                    if gpu in used or not pending:continue
                    m=pending.pop(0);root=a.output/m['group'];dest=root/'results'/m['model_id']
                    require(not dest.exists(),'Output exists: '+str(dest))
                    require(sha(m['checkpoint'])==m['checkpoint_sha256'],'Checkpoint changed')
                    argv=[str(a.python),'-B',str(a.project/'scripts/evaluate_paper_supplement.py'),
                          '--case',m['case'],'--checkpoint',m['checkpoint'],'--states',str(root/f'{m["case"]}_states.json'),
                          '--model-id',m['model_id'],'--output-dir',str(dest)]
                    if m['case']=='push':argv+=['--push-placement-proof',str(a.project/'r2_evidence/preflight_push_registered.json')]
                    env=os.environ.copy()
                    for key in ('RANK','JPY_PARENT_PID','ISAAC_JUPYTER_KERNEL','ISAAC_JUPYTER_PYTHON_PACKAGE'):env.pop(key,None)
                    # LOCAL_RANK also selects Isaac's converted_rankN asset cache.
                    # Device inside each process remains cuda:0 after visibility masking.
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu),LOCAL_RANK=str(gpu),WORLD_SIZE='1',OMNI_KIT_ACCEPT_EULA='YES',
                        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',
                        PYTHONPATH=str(a.project/'src/holosoma')+':'+str(a.project/'src/holosoma_retargeting'))
                    with (root/(m['model_id']+'.stdout.log')).open('xb') as log:
                        proc=subprocess.Popen(argv,cwd=a.project,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    r=dict(model_id=m['model_id'],gpu=gpu,pid=proc.pid,argv=argv,started_unix=time.time(),output=str(dest))
                    save(root/(m['model_id']+'.launch.json'),r)
                    active.append(dict(proc=proc,record=r,model=m))
                for job in active[:]:
                    proc=job['proc'];r=job['record'];m=job['model']
                    if proc.poll() is None:continue
                    result=dict(r,process_returncode=proc.returncode,ended_unix=time.time(),validated=False)
                    try:validate_result(a,m,Path(r['output']),proc.returncode);result['validated']=True
                    except Exception as exc:result['error']=repr(exc)
                    save(a.output/m['group']/(m['model_id']+'.exit.json'),result)
                    finished.append(result);active.remove(job);aggregate(a.output/m['group'])
                publish('evaluating')
                if active:time.sleep(5)
            publish('complete' if len(finished)==6 and all(r['validated'] for r in finished) else 'needs_review')
        except BaseException as exc:
            # Only this evaluator's children, never training processes.
            for j in active:
                if j['proc'].poll() is None:j['proc'].terminate()
            save(a.output/'batch_failure.json',dict(error=repr(exc),time=time.time()))
            publish('blocked');raise

def main():
    p=argparse.ArgumentParser()
    for name in ('project','python','ops','train-ops','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.prepare:prepare(a)
    if a.execute:execute(a)
    if not a.prepare and not a.execute:p.error('Choose --prepare or --execute')

if __name__=='__main__':main()
