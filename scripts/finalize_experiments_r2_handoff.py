#!/usr/bin/env python3
"""Finish R2 evidence from frozen references and recorded rollouts; no training."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from prepare_experiments_r2_cpu import (ROOT, EVAL, BODY, HANDS, JOINTS, ROBOT,
    BatchFK, reference, read, write, sha, identity, csv_write, quat_error)

CPU=ROOT/'paper_handoff_exports/EXPERIMENTS_R2_20261024_CPU_v2'
NEW=ROOT/'logs/ExperimentsR2_evaluation_20261026_retry01'

def csv_read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def assemble(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    shutil.copytree(CPU,out/'CPU_first_batch')
    for name in ['ref_role_synthesis.csv','ref_role_synthesis_by_hand.csv','ref_shared_scene.csv','wrist_orientation.csv','preflight_push_registered.json']:
        shutil.copy2(CPU/name,out/name)
    for group in ['U5_push_registered','U6_kick']:
        shutil.copytree(NEW/group,out/group)
        (out/group/'videos').mkdir()
        for m in read(NEW/group/'models.json'):
            shutil.copytree(NEW/'media'/m['model_id'],out/group/'videos'/m['model_id'])
    for name in ['models.json','local_checkpoint_map.json','local_trajectory_map.json','provenance.json','batch_status.json']:
        shutil.copy2(NEW/name,out/('evaluation_'+name))
    (out/'checks').mkdir();(out/'sources').mkdir();(out/'U7_media').mkdir()
    mapping=[]
    for m in read(NEW/'models.json'):
        group=m['group'];folder=NEW/group/'results'/m['model_id']
        for e in read(folder/'episodes.json'):
            path=folder/Path(e['trajectory']).name
            mapping.append(dict(model_id=m['model_id'],state_id=e['state_id'],kind=e['kind'],
                original_trajectory=e['trajectory'],package_path=str(Path(group)/'results'/m['model_id']/path.name),sha256=sha(path)))
    write(out/'checks/trajectory_mapping.json',mapping)
    assert len(mapping)==306
    shutil.copytree(NEW/'sources',out/'sources/evaluation_snapshot')
    dep=ROOT/'logs/ExperimentsR2/deployment_20261024'
    shutil.copy2(dep/'retry1/r2_transfer_manifest_20261026.json',out/'checks/training_transfer_manifest.json')
    shutil.copy2(dep/'evaluation_20261026/transfer_verified.json',out/'checks/evaluation_transfer_verified.json')
    shutil.copy2(dep/'evaluation_20261026/remote_inventory.json',out/'checks/evaluation_transfer_manifest.json')
    print('assembled',out,flush=True)

def tracking(out):
    fk=BatchFK();fkcheck=fk.validate();names=BODY+HANDS
    spacing=read(out/'preflight_push_registered.json')['lateral_spacing_m']
    refs={};checks=[];rows=[]
    for case in ['push','kick']:
        ref,path=reference(case,names,spacing=spacing if case=='push' else .8)
        samples={k:v.numpy() for k,v in ref.sample(torch.arange(ref.num_frames)).items()}
        refs[case]=(samples,path)
        np.savez_compressed(out/'checks'/f'{case}_reference_for_fk.npz',**samples,
                            body_names=names,joint_names=JOINTS,fps=ref.fps)
    for m in read(NEW/'models.json'):
        folder=NEW/m['group']/'results'/m['model_id'];ref,path=refs[m['case']];n=len(ref['object_pos_w'])
        manifest=read(folder/'run_manifest.json'); assert manifest['checkpoint_iteration']==m['checkpoint_iteration']
        if m['case']=='push':assert manifest['placement_provenance']['lateral_spacing_m']==spacing
        check=dict(model_id=m['model_id'],reference=identity(path),body_names_to_track=BODY,hand_names=HANDS,
                   reference_array=f"checks/{m['case']}_reference_for_fk.npz",episodes=[])
        for e in read(folder/'episodes.json'):
            if e['kind']!='perturbed':continue
            file=folder/Path(e['trajectory']).name
            with np.load(file,allow_pickle=False) as z:
                frame=z['reference_frame'].astype(int);K=len(frame)
                assert K==e['steps']+1 and np.array_equal(frame,np.minimum(z['episode_step'],n-1))
                offset=z['reference_object_pos_w']-ref['object_pos_w'][frame];origin=offset[0]
                assert abs(offset-origin).max()<2e-6 and abs(origin).max()<1e-5
                assert quat_error(z['reference_object_quat_xyzw'],ref['object_quat_w'][frame])['max_rad']<2e-6
                poses=fk.poses(z['root_pos'],z['root_quat_xyzw'],z['dof_pos'],names)
                d=np.linalg.norm(poses[...,:3,3]-ref['agent_body_pos_w'][frame]-origin,axis=-1)
                row=dict(model_id=m['model_id'],configuration=m['configuration'],seed=m['training_seed'],state_id=e['state_id'],
                    completed=e['completed_reference'],frames=K,E_body_cm=float(d[:,:,:len(BODY)].mean()*100),E_hand_cm=float(d[:,:,len(BODY):].mean()*100))
                rows.append(row)
                check['episodes'].append(dict(state_id=e['state_id'],frames=K,origin=origin,sha256=sha(file),
                    reward_reference_frame_mismatches=int(np.count_nonzero(z['reward_reference_frame']!=frame))))
        checks.append(check);print('U3 new',m['model_id'],flush=True)
    assert len(rows)==300
    csv_write(out/'checks/body_tracking_new_episodes.csv',rows)
    old=csv_read(CPU/'body_tracking_episodes.csv')
    for r in old:
        r['seed']=int(r['seed']);r['frames']=int(r['frames']);r['completed']=r['completed']=='True'
        for k in ['E_body_cm','E_hand_cm']:r[k]=float(r[k])
    rows=old+rows;assert len(rows)==1350
    csv_write(out/'body_tracking_episodes.csv',rows)
    per=[];summary=[]
    for config in sorted({r['configuration'] for r in rows}):
        for scope in ['completed','prefix']:
            seeds=sorted({r['seed'] for r in rows if r['configuration']==config})
            items=[]
            for seed in seeds:
                rs=[r for r in rows if r['configuration']==config and r['seed']==seed and (scope=='prefix' or r['completed'])]
                v=dict(configuration=config,scope=scope,seed=seed,episodes=len(rs))
                for k in ['E_body_cm','E_hand_cm']:v[k]=float(np.mean([r[k] for r in rs])) if rs else '--'
                items.append(v);per.append(v)
            valid=[r for r in items if r['episodes']]
            s=dict(configuration=config,scope=scope,training_seeds_evaluated=len(seeds),valid_seeds=len(valid),episodes=sum(r['episodes'] for r in valid))
            for k in ['E_body_cm','E_hand_cm']:
                values=[r[k] for r in valid];s[k+'_mean']=float(np.mean(values)) if values else '--'
                s[k+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else '--'
            summary.append(s)
    csv_write(out/'body_tracking_per_seed.csv',per);csv_write(out/'body_tracking_summary.csv',summary)
    write(out/'checks/fk_alignment_check.json',dict(FK=fkcheck,new_models=checks,
        old_alignment='CPU_first_batch/checks/fk_alignment_check.json',new_episodes=300,total_episodes=1350,
        protocol='World link-origin FK, includes initial and terminal, reference_frame alignment, no planar realignment, episode->seed means, ddof=1'))

def stats(out):
    from paper_evaluation_utils import metrics
    evidence=[];terms=[];paired=[]
    for m in read(NEW/'models.json'):
        folder=NEW/m['group']/'results'/m['model_id'];s=read(folder/'summary.json')
        states=read(NEW/m['group']/f'{m["case"]}_states.json')
        assert s['states_sha256']==sha(NEW/m['group']/f'{m["case"]}_states.json')
        assert [e['state_id'] for e in s['episodes']]==[x['state_id'] for x in states['states']]
        exits=read(NEW/m['group']/(m['model_id']+'.exit.json'));assert exits['process_returncode']==0 and exits['validated']
        checks=[]
        for e in s['episodes']:
            with np.load(folder/Path(e['trajectory']).name,allow_pickle=False) as z:
                recompute=metrics(z)
                assert all(abs(recompute[k]-e[k])<1e-9 for k in recompute)
                assert e['completed_reference']==(e['steps']==e['horizon_steps'] and not e['terminal_terms'].get('joint_bad_tracking',False))
            checks.append(dict(state_id=e['state_id'],passed=True))
            for term,active in e['terminal_terms'].items():
                terms.append(dict(model_id=m['model_id'],case=m['case'],kind=e['kind'],state_id=e['state_id'],term=term,active=active))
        evidence.append(dict(model_id=m['model_id'],states_sha256=s['states_sha256'],episodes=len(checks),metric_recompute='passed',process_returncode=0))
    oldmodels=read(EVAL/'models.json')
    for seed in [721,722,723]:
        oldmodel=next(m for m in oldmodels if m['model_id']==f'Push-livePD_seed{seed}')
        old=read(EVAL/oldmodel.get('evaluation_directory','results/'+oldmodel['model_id'])/'summary.json')
        new=read(NEW/'U5_push_registered/results'/f'Push-registered_seed{seed}'/'summary.json')
        assert old['states_sha256']==new['states_sha256']
        for a,b in zip(old['episodes'],new['episodes']):
            assert a['state_id']==b['state_id']
            if a['kind']=='nominal':continue
            paired.append(dict(seed=seed,state_id=a['state_id'],old_completed=a['completed_reference'],new_completed=b['completed_reference'],
                old_progress=a['progress'],new_progress=b['progress'],old_object_rmse_m=a['object_rmse_3d_m'],new_object_rmse_m=b['object_rmse_3d_m']))
    csv_write(out/'checks/paired_push_episodes.csv',paired);csv_write(out/'checks/termination_terms.csv',terms)
    comparison=[]
    for seed in [721,722,723]:
        rs=[r for r in paired if r['seed']==seed]
        comparison.append(dict(seed=seed,paired_episodes=len(rs),old_completion_rate=np.mean([r['old_completed'] for r in rs]),
             new_completion_rate=np.mean([r['new_completed'] for r in rs]),
             gained=sum(not r['old_completed'] and r['new_completed'] for r in rs),
             lost=sum(r['old_completed'] and not r['new_completed'] for r in rs)))
    csv_write(out/'push_paired_comparison.csv',comparison)
    write(out/'checks/statistics_check.json',dict(verified=True,models=evidence,total_episodes=306,perturbed=300,nominal=6,
        nominal_excluded_from_main=True,paired_push_state_hash_verified=True,paired_push_episodes=150,
        original_csv_schema_preserved=True,sample_sd_ddof=1))
    for name in ['main_results.csv','per_seed_summary.csv','nominal_results.csv']:
        shutil.copy2(EVAL/name,out/('old_'+name))
    # Include exact configured runtime body names, not URDF-only name guesses.
    from holosoma.config_values.marl.g1.reward import g1_29dof_plan5_push_reward
    from dataclasses import asdict
    import re
    term=g1_29dof_plan5_push_reward.terms['undesired_contacts'];pattern=term.params['undesired_contacts_body_names']
    p=NEW/'U5_push_registered/results/Push-registered_seed721/actual_reference_channels.npz'
    with np.load(p) as z:names=z['body_names'].tolist()
    csv_write(out/'checks/runtime_selected_contact_bodies.csv',[dict(link=n,selected_for_net_force_history=True,
        penalty_regex_match=bool(re.match(pattern,n)),weight=term.weight,threshold_N=term.params['threshold']) for n in names])
    write(out/'checks/contact_runtime_evidence.json',dict(runtime_names_source=identity(p),body_names=names,
        source=identity(ROOT/'src/holosoma/holosoma/simulator/isaacsim/isaacsim.py'),
        inference='Simulator maps contact sensor names to configured runtime body names; completed initialization establishes these selected names existed. Full sensor list and force time-series were not exported.',
        hand_note='rubber_hand links excluded from selected 32-body list; wrist_yaw excluded by regex. Do not claim per-hand force measurements.',
        undesired_contacts=asdict(term)))
    print('statistics checked, paired comparison:',comparison,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['assemble','tracking','stats'],required=True);a=p.parse_args()
    globals()[a.phase](a.output.resolve())
if __name__=='__main__':main()
