#!/usr/bin/env python3
"""CPU-only R2 preflight, evidence inventory and old-rollout tracking analysis.

Never starts a simulator, optimizer, renderer or remote process. Old outputs
are read-only. Run in hssim with --output pointing at a new directory.
"""
import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/holosoma'))
from holosoma.managers.command.terms.wbt import MotionLoader
from holosoma.managers.command.terms.marl import PairedA1MotionCommand
from holosoma.envs.marl import PairedA1Reference, PairedMotionReference
from holosoma.config_values.marl.g1.command import g1_29dof_paired_a1_command, motion_config
from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline
from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
from paper_evaluation_utils import evaluation_result_dir

PKG = ROOT / 'src/holosoma'
EVAL = ROOT / 'logs/PaperSupplement20260929/unified_evaluation_20261003'
GEOM = ROOT / 'logs/ReferenceGeometry/reference_geometry_round2_20261017'
MARL = Path('/home/kevin/holosoma-rubber-hand-marl')
ROBOT = PKG / 'holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
HANDS = ['left_rubber_hand_link', 'right_rubber_hand_link']
JOINTS = list(g1_29dof_plan5_push_baseline.robot.dof_names)
BODY = list(motion_config.body_names_to_track)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    return dict(path=str(p), exists=p.is_file(), sha256=sha(p) if p.is_file() else None)


def json_default(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default, allow_nan=False)+'\n')


def csv_write(path, rows, fields=None):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def error(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return dict(max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d*d))))


def quat_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    r = (Rotation.from_quat(a.reshape(-1, 4)).inv() * Rotation.from_quat(b.reshape(-1, 4))).magnitude()
    return dict(max_rad=float(r.max()), rms_rad=float(np.sqrt(np.mean(r*r))))


class BatchFK:
    """Vectorized URDF link-origin FK, checked independently against yourdfpy."""
    def __init__(self, path=ROBOT):
        self.u = yourdfpy.URDF.load(str(path), load_meshes=False, load_collision_meshes=False)
        if self.u.actuated_joint_names != JOINTS:
            raise ValueError('URDF and simulator DOF order differ')
        self.base = self.u.base_link
        self.joints = []
        pending = list(self.u.robot.joints)
        visited = {self.base}
        while pending:
            available = [j for j in pending if j.parent in visited]
            if not available: raise ValueError('Disconnected joint tree')
            for j in available:
                if j.mimic is not None: raise ValueError('Mimic requires explicit implementation')
                if j.type not in ('fixed', 'revolute', 'continuous', 'prismatic'): raise ValueError(j.type)
                self.joints.append(j); visited.add(j.child); pending.remove(j)

    def poses(self, root_pos, root_quat_xyzw, q, names):
        shape = np.shape(q)[:-1]
        q = np.asarray(q).reshape(-1, len(JOINTS))
        n = len(q)
        base = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        base[:, :3, :3] = Rotation.from_quat(np.asarray(root_quat_xyzw).reshape(-1,4)).as_matrix()
        base[:, :3, 3] = np.asarray(root_pos).reshape(-1,3)
        poses = {self.base: base}
        for j in self.joints:
            origin = np.eye(4) if j.origin is None else j.origin
            local = np.broadcast_to(origin, (n,4,4)).copy()
            if j.type != 'fixed':
                axis = np.asarray(j.axis, dtype=float)
                axis /= np.linalg.norm(axis)
                value = q[:, JOINTS.index(j.name)]
                if j.type == 'prismatic':
                    local[:, :3, 3] += (origin[:3,:3] @ axis)[None] * value[:,None]
                else:
                    local[:, :3, :3] = origin[:3,:3] @ Rotation.from_rotvec(value[:,None]*axis).as_matrix()
            poses[j.child] = poses[j.parent] @ local
        return np.stack([poses[n] for n in names], axis=1).reshape(*shape,len(names),4,4)

    def validate(self):
        rng = np.random.default_rng(241024)
        q = rng.uniform(-.5,.5,(9,29)); p = rng.uniform(-1,1,(9,3))
        r = Rotation.random(9, random_state=rng).as_quat()
        names = BODY + HANDS
        actual = self.poses(p,r,q,names)
        expected = np.empty_like(actual)
        for i in range(9):
            self.u.update_cfg(q[i]); base = np.eye(4)
            base[:3,:3] = Rotation.from_quat(r[i]).as_matrix(); base[:3,3] = p[i]
            for k,name in enumerate(names): expected[i,k] = base @ self.u.get_transform(name)
        e = error(actual, expected)
        assert e['max_abs'] < 1e-10, e
        return dict(independent_backend='yourdfpy', root_link=self.base, joint_order=JOINTS,
                    random_states=9, max_transform_error=e, passed=True)


def reference(case, names, spacing=.8):
    if case == 'push':
        path = PKG / motion_config.motion_file
        loader = MotionLoader(str(path), names, JOINTS, device='cpu')
        return PairedA1Reference.from_motion_loader(loader, lateral_spacing_m=spacing), path
    if case in ('pull','kick'):
        from holosoma.config_values.marl.g1.command import PLAN5_PULL_PAIRED_REFERENCE_FILE
        from holosoma.config_values.marl.g1.kick_command import PLAN5_KICK_RUNTIME_REFERENCE_FILE
        rel = PLAN5_PULL_PAIRED_REFERENCE_FILE if case=='pull' else PLAN5_KICK_RUNTIME_REFERENCE_FILE
    else:
        e = get_core4d_pair_experiment(dict(chair='chair021',bucket='bucket003',table='smalltable5kg_A')[case])
        rel = e.runtime_reference_file
    path = PKG / rel
    return PairedMotionReference(str(path), names, JOINTS, device='cpu'), path


class CaptureSimulator:
    def __init__(self, bodies):
        self._body_list = bodies; self.dof_names = JOINTS
        self.scene = SimpleNamespace(env_origins=torch.zeros(1,3))
    def get_actor_indices(self, *a, **kw): return torch.tensor([0])
    def set_agent_root_states(self, ids, values): self.roots = values.clone()
    def set_agent_dof_states(self, ids, pos, vel): self.joints = pos.clone()
    def set_actor_states(self, names, ids, values): self.object = values.clone()


def push_preflight(out):
    m = read(GEOM/'push/input_manifest.json')
    assets = []
    for k in ['source','source_object','target_object','actual_robot']:
        a = identity(m[k]['path']); a['manifest_sha256'] = m[k]['sha256']
        assert a['sha256'] == a['manifest_sha256'], k
        assets.append(a)
    selection = read(GEOM/'push/selection.json'); params = selection['role_parameters']
    spacing = float(np.linalg.norm(np.array(params[1]['c'])-params[0]['c']))
    spec = read(GEOM/'push/correspondence_spec.json')
    widths = spec['source_tabletop_width_m']
    assert abs(spacing - (1.4-widths)) < 1e-10
    assert all(p['parity']==1 and np.allclose(p['Q'], np.eye(3),atol=1e-12) for p in params)
    bodies = ['pelvis']+[n for n in BODY+HANDS if n!='pelvis']
    snapshots = {}; checks = {}
    # Exercise each actual hook configuration with a capture-only simulator.
    # No physics, no Isaac and no replacement reference loader.
    for phase in ['setup_terms','reset_terms','step_terms']:
        original = getattr(g1_29dof_paired_a1_command, phase)['paired_motion_command']
        cfg = replace(original, params={**original.params, 'lateral_spacing_m':spacing})
        sim = CaptureSimulator(bodies)
        env = SimpleNamespace(num_envs=1,device='cpu',simulator=sim,is_evaluating=True)
        command = PairedA1MotionCommand(cfg,env); command.setup(); command.reset(None)
        n = command.reference.num_frames
        reset_roots = sim.roots.numpy().copy()
        assert int(command.time_steps[0])==0
        samples = []
        for t in range(n):
            assert int(command.time_steps[0])==t
            samples.append({k:v.numpy().copy()[0] for k,v in command._sample().items()})
            command.step()
        assert int(command.time_steps[0])==0
        channels = {k:np.stack([s[k] for s in samples]) for k in samples[0]}
        assert np.array_equal(reset_roots[0,:,:3], channels['agent_body_pos_w'][0,:,0])
        snapshots[phase] = channels
        checks[phase] = dict(lateral_spacing_m=spacing,paired_reference_file=command.paired_reference_file,
                             full_phase_cycle_verified=True,reset_to_zero_verified=True)
    z = snapshots['setup_terms']
    for other in snapshots.values():
        for k in z: assert np.array_equal(z[k],other[k])
    np.savez_compressed(out/'checks/push_registered_loader_channels.npz', **z,body_names=bodies,joint_names=JOINTS,fps=50)
    offline = np.load(GEOM/'push/new_pair_reference.npz')
    q = offline['robot_qpos']; obj = offline['object_qpos']
    differences = dict(root_position_m=error(z['agent_body_pos_w'][:,:,0],q[:,:,:3]),
        root_rotation=quat_error(z['agent_body_quat_w'][:,:,0],q[:,:,[4,5,6,3]]),
        joint_position_rad=error(z['agent_joint_pos'],q[:,:,7:]),
        object_position_m=error(z['object_pos_w'],obj[:,:3]),
        object_rotation=quat_error(z['object_quat_w'],obj[:,[4,5,6,3]]))
    assert max(differences[k]['max_abs'] for k in ['root_position_m','joint_position_rad','object_position_m'])<2e-6
    assert max(differences[k]['max_rad'] for k in ['root_rotation','object_rotation'])<2e-6
    old,_ = reference('push', bodies, .8)
    old = {k:v.numpy() for k,v in old.sample(torch.arange(old.num_frames)).items()}
    invariants = {}
    for k in z:
        if k in ('agent_body_pos_w','agent_body_lin_vel_w'): continue
        invariants[k] = bool(np.array_equal(z[k],old[k])); assert invariants[k], k
    checkpoint = MARL/'logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt'
    ck = identity(checkpoint)
    assert ck['sha256']=='11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224'
    stages=[]
    for i in range(3):
        path=EVAL/f'training_evidence/Push-livePD_seed722/stage{i}_run_config.json'
        cfg=read(path)
        stages.append(dict(evidence=identity(path),start=cfg['start_iteration'],count=cfg['iterations_requested'],
            critic_only=cfg['critic_only'],actor_lr=cfg['actor_learning_rate'],critic_lr=cfg['critic_learning_rate'],
            source_checkpoint_sha256=cfg['source_checkpoint_sha256']))
    assert [(s['start'],s['count']) for s in stages]==[(0,50),(50,8000),(8050,5000)]
    assert all(s['source_checkpoint_sha256']==ck['sha256'] for s in stages)
    states=identity(EVAL/'push_states.json')
    assert states['sha256']=='1799ea1eec1a3ff097d46e08e6492694d6455490ccf9f2de26586e079fef895d'
    data=dict(status='cpu_reference_preflight_passed_training_not_started',assets=assets,
        selection=identity(GEOM/'push/selection.json'),original_diagnostic_status=selection['status'],
        source_width_m=widths,target_axis_length_m=1.4,role_parameters=params,lateral_spacing_m=spacing,
        hooks=checks,differences=differences,comparison_tolerance_m_rad=2e-6,
        tolerance_reason='Original loader uses float32; offline geometry uses float64. Rotation error normalizes quaternions.',
        invariant_channels_bitwise_equal=invariants,only_changed_channels=['agent_body_pos_w','agent_body_lin_vel_w'],
        body_velocity_reason='Unchanged original torch.gradient(offset), not new runtime velocities',
        loader_output=identity(out/'checks/push_registered_loader_channels.npz'),
        offline_runtime_crosscheck=identity(GEOM/'push/new_runtime.npz'),offline_runtime_used_for_training=False,
        source_checkpoint=ck,stages=stages,actor_updates=13000,critic_updates=13050,
        mass_kg=20,num_envs=2048,steps_per_env=24,seeds=[721,722,723],paired_states=states,
        actual_training_command=None,command_missing_reason='Awaiting cloud GPU availability; launcher not started',
        diagnostic_note='Original source-relative foot drift failure retained. No IK, reward or tolerance repair.')
    write(out/'preflight_push_registered.json',data)
    print('Push CPU preflight:', differences, flush=True)
    return data


def inventory(out):
    models=read(EVAL/'models.json'); rows=[]; evidence=[]
    for model in models:
        folder=evaluation_result_dir(EVAL,model)
        episodes=read(folder/'episodes.json'); perturbed=[e for e in episodes if e['kind']=='perturbed']
        nominal=[e for e in episodes if e['kind']=='nominal']
        assert len(perturbed)==50 and len(nominal)==1
        assert len({e['state_id'] for e in perturbed})==50
        manifest=read(folder/'run_manifest.json')
        evidence.append(dict(model_id=model['model_id'],manifest=identity(folder/'run_manifest.json'),
            episodes=identity(folder/'episodes.json'),selected_directory=str(folder),
            checkpoint_sha256_recorded=manifest['checkpoint_sha256'],
            terminal_terms=dict(Counter(k for e in perturbed for k,v in e['terminal_terms'].items() if v)),
            perturbed=50,nominal=1))
        for e in episodes:
            path=folder/Path(e['trajectory']).name
            rows.append(dict(model_id=model['model_id'],state_id=e['state_id'],kind=e['kind'],
                original_trajectory=e['trajectory'],local_trajectory=str(path),sha256=sha(path)))
    assert len(models)==21 and sum(e['perturbed'] for e in evidence)==1050
    write(out/'checks/state_inventory.json',dict(models=21,perturbed=1050,nominal=21,evidence=evidence,
        retry_selection='models.json evaluation_directory honored; excluded old attempts not pooled'))
    csv_write(out/'checks/trajectory_inventory.csv',rows)
    # Record Kick actual budget (not Push warmup assumptions).
    kp=MARL/'logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/run_config.json'
    k=read(kp); ck=MARL/k['source_checkpoint']
    assert sha(ck)==k['source_checkpoint_sha256']
    write(out/'checks/kick_training_inventory.json',dict(source=identity(kp),
        actual_config={x:v for x,v in k.items() if not isinstance(v,(list,dict))},
        initialization=identity(ck),existing_final=identity(kp.parent/'model_08000.pt'),
        budget=8000,critic_warmup=0,actor_lr=k['actor_learning_rate'],new_seeds=[722,723],
        classification='legacy mirrored; registration equivalence NOT yet established',
        registration_check_status='pending; do not relabel legacy seed721 as registration training',training_started=False))
    print('Inventory verified: 21 models, 1050 perturbed +21 nominal; Kick budget8000 no warmup.',flush=True)
    return models


def tracking(out, models, fk):
    names=BODY+HANDS; refs={}; rows=[]; alignment=[]; perseed=[]
    for case in sorted({m['case'] for m in models}):
        ref,path=reference(case,names)
        refs[case]=({k:v.numpy() for k,v in ref.sample(torch.arange(ref.num_frames)).items()},path)
    for model in models:
        case=model['case']; ref,path=refs[case]; n=len(ref['object_pos_w'])
        folder=evaluation_result_dir(EVAL,model); episodes=read(folder/'episodes.json')
        cfgpath=Path(model.get('config',Path(model['checkpoint']).parent/'run_config.json'))
        cfg=read(cfgpath)
        # Actual command list: compare the saved config if present; retain source hashes otherwise.
        if 'body_names_to_track' in cfg: assert cfg['body_names_to_track']==BODY
        a=dict(model_id=model['model_id'],reference=identity(path),config=identity(cfgpath),
               body_names_to_track=BODY,hand_links=HANDS,episodes=[],root_link=fk.base,
               frame_rule='reference_frame; includes initial and terminal captured state',
               quaternion_input='xyzw',world_alignment='no planar realignment; verify object origin with recorded reference')
        for e in episodes:
            if e['kind']!='perturbed':continue
            file=folder/Path(e['trajectory']).name
            with np.load(file,allow_pickle=False) as z:
                frame=z['reference_frame'].astype(int); K=len(frame)
                assert K==int(e['steps'])+1 and int(z['episode_step'][0])==0
                assert np.array_equal(frame,np.minimum(z['episode_step'],n-1))
                offset=z['reference_object_pos_w']-ref['object_pos_w'][frame]
                origin=offset[0]
                assert np.max(abs(offset-origin))<2e-6
                assert quat_error(z['reference_object_quat_xyzw'],ref['object_quat_w'][frame])['max_rad']<2e-6
                assert np.max(abs(origin))<1e-5, 'Nonzero environment origin requires explicit per-env handling'
                poses=fk.poses(z['root_pos'],z['root_quat_xyzw'],z['dof_pos'],names)
                distance=np.linalg.norm(poses[...,:3,3]-(ref['agent_body_pos_w'][frame]+origin),axis=-1)
                body=float(distance[:,:,:len(BODY)].mean()*100)
                hand=float(distance[:,:,len(BODY):].mean()*100)
                reward_mismatch=int(np.count_nonzero(z['reward_reference_frame']!=frame))
                a['episodes'].append(dict(state_id=e['state_id'],frames=K,origin=origin,
                    reward_reference_frame_mismatch_count=reward_mismatch,trajectory_sha256=sha(file)))
            rows.append(dict(model_id=model['model_id'],configuration=model['configuration'],seed=model['training_seed'],
                state_id=e['state_id'],completed=e['completed_reference'],frames=K,E_body_cm=body,E_hand_cm=hand))
        alignment.append(a)
        print('U3',model['model_id'],'50 trajectories complete',flush=True)
    for config in sorted({r['configuration'] for r in rows}):
        for scope in ['prefix','completed']:
            for seed in sorted({r['seed'] for r in rows if r['configuration']==config}):
                selected=[r for r in rows if r['configuration']==config and r['seed']==seed and (scope=='prefix' or r['completed'])]
                perseed.append(dict(configuration=config,scope=scope,seed=seed,episodes=len(selected),
                    E_body_cm=float(np.mean([r['E_body_cm'] for r in selected])) if selected else '--',
                    E_hand_cm=float(np.mean([r['E_hand_cm'] for r in selected])) if selected else '--'))
    summaries=[]
    for config in sorted({r['configuration'] for r in rows}):
        for scope in ['completed','prefix']:
            allseed=[r for r in perseed if r['configuration']==config and r['scope']==scope]
            valid=[r for r in allseed if r['episodes']>0]
            record=dict(configuration=config,scope=scope,training_seeds_evaluated=len(allseed),valid_seeds=len(valid),episodes=sum(r['episodes'] for r in valid))
            for field in ['E_body_cm','E_hand_cm']:
                values=[r[field] for r in valid]
                record[field+'_mean']=float(np.mean(values)) if values else '--'
                record[field+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else '--'
            summaries.append(record)
    assert len(rows)==1050
    csv_write(out/'body_tracking_episodes.csv',rows)
    csv_write(out/'body_tracking_per_seed.csv',perseed)
    csv_write(out/'body_tracking_summary.csv',summaries)
    write(out/'checks/fk_alignment_check.json',dict(FK=fk.validate(),models=alignment,
        aggregation='Episode means, equal episodes within seed, equal seed means; ddof=1; completed does not include failures',
        phase_origin='Initialized state and final pre-autoreset state included',units='cm'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'checks').mkdir()
    torch.set_num_threads(2); started=time.time()
    write(out/'provenance.json',dict(started_unix=started,python=sys.executable,python_version=sys.version,
        command=sys.argv,script=identity(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        instruction_commit='ab915c56e3bd6341a3214131fcfd7f539cb6f89e',training_authorized_location='cloud_only_wait_for_user',
        simulator_started=False,training_started=False))
    push_preflight(out); models=inventory(out);fk=BatchFK();tracking(out,models,fk)
    write(out/'cpu_stage_status.json',dict(complete=True,return_code=0,elapsed_s=time.time()-started,
        scope=['U5_CPU_reference_preflight','U6_training_inventory_only','U3_old1050'],training_started=False))


if __name__=='__main__':main()
