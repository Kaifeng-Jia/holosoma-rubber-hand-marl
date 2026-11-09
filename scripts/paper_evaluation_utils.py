"""Simulator-free protocol helpers for reproducible paired-policy evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,data):
    # Readers may poll while the queue updates; publish complete JSON atomically.
    import os
    path=Path(path)
    temporary=path.with_name(path.name+f'.writing.{os.getpid()}')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,path)

def evaluation_result_dir(root,model):
    """Explicit retry selection; old attempts stay intact and are never pooled."""
    root=Path(root).resolve()
    selected=(root/model.get('evaluation_directory','results/'+model['model_id'])).resolve()
    if not selected.is_relative_to(root/'results'):
        raise ValueError('Evaluation directory must remain under this batch results/')
    return selected

def make_states(case,seed=241003):
    rng=np.random.default_rng(seed)
    rows=[dict(state_id='nominal',kind='nominal',robot_xy_m=[[0.,0.],[0.,0.]],robot_yaw_rad=[0.,0.])]
    for i in range(50):
        rows.append(dict(state_id=f'perturbed_{i:03d}',kind='perturbed',
                         robot_xy_m=rng.uniform(-.02,.02,(2,2)).tolist(),
                         robot_yaw_rad=rng.uniform(-np.deg2rad(2),np.deg2rad(2),2).tolist()))
    return dict(version='relative_initial_state_v1',case=case,generation_seed=seed,
                sampling='independent uniform per robot world XY and yaw about its own root',
                position_bound_m=.02,yaw_bound_deg=2.,object_perturbed=False,
                joint_pose_and_heights_unchanged=True,states=rows)

def yaw_transform(root,xy,yaw):
    """World XY shift and world yaw; rotate root velocities; preserve height."""
    root=np.asarray(root).copy();xy=np.asarray(xy);yaw=np.asarray(yaw)
    root[...,:2]+=xy
    c=np.cos(yaw);s=np.sin(yaw);sz=np.sin(yaw/2);cz=np.cos(yaw/2)
    x,y,z,w=np.moveaxis(root[...,3:7],-1,0)
    root[...,3:7]=np.stack((cz*x-sz*y,cz*y+sz*x,cz*z+sz*w,cz*w-sz*z),axis=-1)
    for start in (7,10):
        vx=root[...,start].copy();vy=root[...,start+1].copy()
        root[...,start]=c*vx-s*vy;root[...,start+1]=s*vx+c*vy
    return root

def metrics(channels):
    delta=channels['object_pos_w']-channels['reference_object_pos_w']
    if not np.isfinite(delta).all():raise ValueError('Non-finite trajectory')
    return dict(object_rmse_3d_m=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),
                object_rmse_xy_m=float(np.sqrt(np.mean(np.sum(delta[:,:2]**2,axis=-1)))),
                height_rmse_m=float(np.sqrt(np.mean(delta[:,2]**2))),
                height_bias_m=float(np.mean(delta[:,2])))
