#!/usr/bin/env python3
"""Aggregate real episodes; expose missing seeds and conditional RMSE explicitly."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from paper_evaluation_utils import write_json,evaluation_result_dir

def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def summarize(root):
    models=json.loads((root/'models.json').read_text());episodes=[];nominal=[];seeds=[]
    for model in models:
        path=evaluation_result_dir(root,model)/'summary.json'
        if not path.exists():continue
        exit_path=root/(model['model_id']+'.exit.json')
        if not exit_path.exists():continue
        exit_record=json.loads(exit_path.read_text())
        if not exit_record.get('validated') or exit_record.get('process_returncode')!=0:continue
        s=json.loads(path.read_text())
        if not s.get('complete') or s['episode_count']!=51:continue
        if len({r['state_id'] for r in s['episodes']})!=51:continue
        if sum(r['kind']=='perturbed' for r in s['episodes'])!=50:continue
        rs=[]
        for raw in s['episodes']:
            r=dict(raw,configuration=model['configuration'],training_seed=model['training_seed'],
                   training_reuse_status=model['reuse_status'],checkpoint_sha256=model['checkpoint_sha256'])
            r['terminal_terms']=json.dumps(r['terminal_terms'],sort_keys=True)
            (nominal if r['kind']=='nominal' else episodes).append(r)
            if r['kind']=='perturbed':rs.append(r)
        completed=[r for r in rs if r['completed_reference']]
        seeds.append(dict(model_id=model['model_id'],configuration=model['configuration'],training_seed=model['training_seed'],
                          reuse_status=model['reuse_status'],episodes=len(rs),completion_rate=len(completed)/len(rs),
                          progress_mean=float(np.mean([r['progress'] for r in rs])),
                          completed_object_rmse_3d_m=float(np.mean([r['object_rmse_3d_m'] for r in completed])) if completed else None,
                          completed_episodes=len(completed),prefix_object_rmse_3d_m=float(np.mean([r['object_rmse_3d_m'] for r in rs]))))
    save_csv(root/'episodes.csv',episodes);save_csv(root/'nominal_results.csv',nominal);save_csv(root/'per_seed_summary.csv',seeds)
    tables=[]
    for conf in dict.fromkeys(m['configuration'] for m in models):
        valid=[s for s in seeds if s['configuration']==conf and s['reuse_status'] in ('new_training_verified','historical_effective_code_verified')]
        row=dict(configuration=conf,training_seeds_evaluated=len(valid),target_training_seeds=3,complete_three_seed_result=len(valid)==3)
        for key in ('completion_rate','progress_mean','completed_object_rmse_3d_m'):
            values=[s[key] for s in valid if s[key] is not None]
            row[key+'_valid_seeds']=len(values)
            publish=len(values)==3 if key.startswith('completed_') else bool(values)
            row[key+'_mean']=float(np.mean(values)) if publish else None
            row[key+'_sample_sd']=float(np.std(values,ddof=1)) if publish and len(values)>1 else None
        tables.append(row)
    save_csv(root/'main_results.csv',[r for r in tables if r['configuration'] in ('Push-livePD','Pull-baseline','C-01','B-RZ0','T-07')])
    save_csv(root/'reward_comparison.csv',[r for r in tables if r['configuration'] in ('B-RZ0','B-A','T-07','T-06')])
    write_json(root/'aggregation_status.json',dict(evaluated_models=len(seeds),perturbed_episodes=len(episodes),nominal_episodes=len(nominal),
               models_with_pending_source_audit=[s['model_id'] for s in seeds if s['reuse_status']=='source_audit_pending'],
               note='Pending historical candidates are retained in episode/per-seed files but excluded from aggregate tables. Missing completion RMSE remains null.'))
    return len(seeds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print('Models summarized:',summarize(p.parse_args().root))
