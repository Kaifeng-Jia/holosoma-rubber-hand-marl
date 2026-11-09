#!/usr/bin/env python3
"""Actor-only paired evaluation on an explicit immutable initial-state list."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from paper_evaluation_utils import sha,write_json,yaw_transform,metrics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/holosoma'))
sys.path.insert(0,str(ROOT/'src/holosoma_retargeting'))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',choices=('push','pull','kick','chair','bucket','table'),required=True)
p.add_argument('--push-placement-proof',type=Path,default=None)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--states',type=Path,required=True)
p.add_argument('--model-id',required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--limit',type=int,default=0)
p.add_argument('--reference-variant',type=Path,default=None)
p.add_argument('--object-mass-kg',type=float,default=None,help='System1 T04 mass variant; default keeps the original asset')
a=p.parse_args()
if a.reference_variant is not None and a.case not in ('chair','bucket'):raise ValueError('V3 reference variants only supported for chair/bucket')
record=json.loads(a.states.read_text());states=record['states'][:a.limit or None]
if record['case']!=a.case:raise ValueError('Initial-state case mismatch')
from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_smoke,with_pair_experiment
from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
from holosoma.config_values.marl.g1.core4d_smalltable_observation import g1_29dof_core4d_smalltable_evaluation_observation
from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline,g1_29dof_plan5_pull_baseline
from holosoma.config_values.marl.g1.kick_experiment import g1_29dof_plan5_kick_baseline
from plan5_registered_placement import configure_push_placement,export_runtime_reference
from holosoma.utils.eval_utils import init_sim_imports
experiment=None
mass_variant=None
if a.case in ('push','pull','kick'):
    config=dict(push=g1_29dof_plan5_push_baseline,pull=g1_29dof_plan5_pull_baseline,
                kick=g1_29dof_plan5_kick_baseline)[a.case]
    if a.object_mass_kg is not None:
        sys.path.insert(0,str(ROOT))
        from system1.mass_variants import ensure_mass_variant
        _urdf,mass_variant=ensure_mass_variant(config.robot.object.object_urdf_path,a.object_mass_kg)
        config=replace(config,robot=replace(config.robot,object=replace(config.robot.object,object_urdf_path=_urdf)))
else:
    experiment=get_core4d_pair_experiment(dict(chair='chair021',bucket='bucket003',table='smalltable5kg_A')[a.case])
    from holosoma.config_values.marl.g1.v3_reference_variant import with_reference_variant
    experiment=with_reference_variant(experiment,a.reference_variant)
    mass_variant=None
    if a.object_mass_kg is not None:
        sys.path.insert(0,str(ROOT))
        from system1.mass_variants import core4d_experiment_with_mass
        experiment,mass_variant=core4d_experiment_with_mass(experiment,a.object_mass_kg)
    config=with_pair_experiment(g1_29dof_core4d_smalltable_smoke,experiment)
    if a.case in ('bucket','table'):
        config=replace(config,env_class='holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager',
                       simulator=replace(config.simulator,config=replace(config.simulator.config,enable_object_hand_contact=True)))
config=replace(config,observation=g1_29dof_core4d_smalltable_evaluation_observation,
               training=replace(config.training,num_envs=1,seed=241003,headless=True))
config,placement_provenance=configure_push_placement(config,a.push_placement_proof,skill=a.case,root=ROOT)
app=init_sim_imports(config)
import torch
from holosoma.agents.mappo.core4d_smalltable_initialization import initialize_core4d_smalltable_model_bundle
from holosoma.agents.mappo.core4d_smalltable_evaluation import deterministic_core4d_smalltable_actions,validate_core4d_smalltable_checkpoint,save_core4d_smalltable_viser
from holosoma.config_types.env import get_tyro_env_config
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app

def main():
    env=None
    try:
        out=a.output_dir.resolve()
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        checkpoint=a.checkpoint.resolve();state=torch.load(checkpoint,map_location='cuda:0',weights_only=False)
        ablation=state.get('v3_ablation')
        if ablation is not None:
            from holosoma.agents.mappo.v3_plan5_ablation import ablation_contract
            if a.case!='pull' or ablation!=ablation_contract(from_scratch=ablation.get('from_scratch',False),zero_teammate_obs=ablation.get('zero_teammate_obs',False)):
                raise ValueError('Invalid V3 Pull ablation contract')
        zero_teammate_obs=bool(ablation and ablation['zero_teammate_obs'])
        if state.get('placement_provenance') != placement_provenance:
            raise ValueError('Checkpoint and evaluation placement differ')
        if state.get('plan5_skill',a.case) != a.case:
            raise ValueError('Checkpoint skill differs from evaluation case')
        if experiment:
            validate_core4d_smalltable_checkpoint(state,experiment_contract=experiment.checkpoint_contract)
            for path,expected in ((experiment.runtime_reference_file,experiment.runtime_reference_sha256),
                                  (experiment.object_urdf_file,experiment.object_urdf_sha256),
                                  (experiment.training_promotion_file,experiment.training_promotion_sha256)):
                if sha(ROOT/'src/holosoma'/path)!=expected:raise ValueError('Asset hash mismatch: '+path)
        else:
            c=state['plan5_mappo']
            if [c['num_agents'],c['actor_obs_dim'],c['critic_obs_dim'],c['action_dim']]!=[2,158,527,29]:raise ValueError('Plan5 shape contract')
        models=initialize_core4d_smalltable_model_bundle(config.algo.config,device='cuda:0')
        models.actor.load_state_dict(state['actor_model_state_dict'],strict=True)
        models.actor_obs_normalizer.load_state_dict(state['actor_obs_normalizer_state_dict'],strict=True)
        models.actor.eval();models.actor_obs_normalizer.eval()
        # A forbidden Critic call fails immediately rather than silently scoring with it.
        def forbidden(*args,**kwargs):raise RuntimeError('Critic invoked during actor-only evaluation')
        models.critic.evaluate=forbidden
        torch.manual_seed(241003)
        env=get_class(config.env_class)(get_tyro_env_config(config),device='cuda:0')
        env.set_is_evaluating();command=env.command_manager.get_state('paired_motion_command')
        runtime_reference_check=export_runtime_reference(env,out,placement_provenance,a.push_placement_proof)
        ids=torch.tensor([0],device=env.device)
        n=int(command.reference.num_frames)
        if not np.isclose(env.dt,1/command.reference.fps):raise ValueError('Reference/control rate mismatch')
        def array(value):return value[0].detach().cpu().numpy().copy()
        def snapshot():
            step=int(env.episode_length_buf[0].item());frame=min(step,n-1)
            sample=command.reference.sample(torch.tensor([frame],device=env.device))
            origin=env.simulator.scene.env_origins[0]
            result=dict(root_pos=array(env.simulator.agent_root_states[...,:3]),
                        root_quat_xyzw=array(env.simulator.agent_root_states[...,3:7]),
                        root_velocity=array(env.simulator.agent_root_states[...,7:13]),
                        dof_pos=array(env.simulator.agent_dof_pos),dof_vel=array(env.simulator.agent_dof_vel),
                        object_pos_w=array(command.simulator_object_pos_w),object_quat_xyzw=array(command.simulator_object_quat_w),
                        reference_object_pos_w=array(sample['object_pos_w']+origin),
                        reference_object_quat_xyzw=array(sample['object_quat_w']),
                        reference_frame=np.asarray(frame),reward_reference_frame=np.asarray(int(command.time_steps[0].item())),
                        episode_step=np.asarray(step))
            sensor=getattr(env.simulator,'object_hand_contact_sensor',None)
            if sensor is not None:result['hand_object_normal_force_w']=array(sensor.data.force_matrix_w[:,0].reshape(-1,2,2,3))
            return result
        captured=None
        original_reward=env._compute_reward
        def capturing_reward():
            nonlocal captured
            original_reward();captured=snapshot()
        env._compute_reward=capturing_reward
        physics={k:v.detach().cpu().tolist() for k,v in dict(mass=env.simulator._object.root_physx_view.get_masses(),
                   material=env.simulator._object.root_physx_view.get_material_properties()).items()}
        manifest=dict(reference_variant=None if a.reference_variant is None else str(a.reference_variant),reference_variant_sha256=None if a.reference_variant is None else sha(a.reference_variant),case=a.case,model_id=a.model_id,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                      v3_ablation=ablation,checkpoint_iteration=state['iter'],states_sha256=sha(a.states),reference_frames=n,
                      horizon_control_steps=n,dt=env.dt,inference='actor_mean_only_critic_forbidden',observation_noise=False,
                      physics=physics,robot_urdf=config.robot.asset.urdf_file,object_urdf=config.robot.object.object_urdf_path,
                      placement_provenance=placement_provenance,
                      runtime_reference_check=runtime_reference_check,
                      endpoint_rule='Execute each reference frame once; stop at clip end before the next wrapped state',
                      error_alignment='Post-physics at k*dt vs reference frame min(k,N-1); terminal state captured before autoreset',
                      training_contract=state.get('core4d_smalltable_mappo',state.get('plan5_mappo')))
        if mass_variant is not None:manifest['system1_object_mass_variant']=mass_variant
        write_json(out/'run_manifest.json',manifest)
        results=[]
        for init in states:
            env.reset_envs_idx(ids);env._refresh_envs_after_reset(ids)
            if command.time_steps[0].item()!=0:raise ValueError('Reference not reset to frame zero')
            nominal_root=env.simulator.agent_root_states.clone()
            transformed=yaw_transform(nominal_root[0].detach().cpu().numpy(),init['robot_xy_m'],init['robot_yaw_rad'])
            env.simulator.set_agent_root_states(ids,torch.tensor(transformed,device=env.device,dtype=nominal_root.dtype).unsqueeze(0))
            env._refresh_envs_after_reset(ids)
            actual=env.simulator.agent_root_states[0].detach().cpu().numpy()
            if not np.isfinite(actual).all() or not np.allclose(actual,transformed,atol=1e-5):raise ValueError('Perturbation not applied to actual reset state')
            env._compute_observations();env._clip_observations();observations=env.obs_buf_dict
            start=snapshot();frames=[start];terms={};reward=0.;captured=None
            for step in range(1,n+1):
                actions=deterministic_core4d_smalltable_actions(models,observations,zero_teammate_obs=zero_teammate_obs)
                observations,rewards,dones,extras=env.step({'actions':actions})
                if captured is None:raise RuntimeError('Pre-reset state capture missing')
                frames.append(captured);reward+=float(rewards[0].item())
                terms={k:bool(v[0].item()) for k,v in extras['termination_terms'].items()}
                if bool(dones[0].item()):break
            channels={k:np.stack([f[k] for f in frames]) for k in start}
            if not all(np.isfinite(v).all() for v in channels.values()):raise ValueError('Non-finite episode states')
            completed=step==n and not terms.get('joint_bad_tracking',False)
            result=dict(model_id=a.model_id,case=a.case,state_id=init['state_id'],kind=init['kind'],steps=step,horizon_steps=n,
                        completed_reference=completed,progress=step/n,terminal_terms=terms,reward_sum_common=reward,
                        **metrics(channels))
            write_json(out/(init['state_id']+'.initial.json'),dict(requested=init,nominal_roots=nominal_root[0].detach().cpu().tolist(),
                       actual_roots=actual.tolist(),actual_initial_state={k:v.tolist() for k,v in start.items()},
                       valid=True,validity_rule='finite states; requested XY/yaw applied; root heights and joint pose unchanged; no outcome filtering'))
            np.savez_compressed(out/(init['state_id']+'.trajectory.npz'),**channels)
            result['trajectory']=str(out/(init['state_id']+'.trajectory.npz'))
            if init['kind']=='nominal':
                save_core4d_smalltable_viser(out/'nominal_viser.npz',channels,metadata=dict(manifest,fps=int(command.reference.fps),result=result,
                                              experiment_contract=experiment.checkpoint_contract if experiment else {}))
            results.append(result);write_json(out/'episodes.json',results);print(json.dumps(result),flush=True)
        write_json(out/'summary.json',dict(manifest,episodes=results,complete=True,episode_count=len(results)))
        return 0
    except BaseException:
        traceback.print_exc();return 1
    finally:
        if env is not None and hasattr(env.simulator,'close'):env.simulator.close()
        close_simulation_app(app)

if __name__=='__main__':raise SystemExit(main())
