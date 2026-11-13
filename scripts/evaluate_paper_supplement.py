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
p.add_argument('--checkpoint',type=Path,default=None)
p.add_argument('--states',type=Path,required=True)
p.add_argument('--model-id',required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--limit',type=int,default=0)
p.add_argument('--reference-variant',type=Path,default=None)
p.add_argument('--object-mass-kg',type=float,default=None,help='System1 T04 mass variant; default keeps the original asset')
p.add_argument('--controller',choices=('policy','sonic','sonic_residual','sonic_planner'),default='policy',help='System1: policy (paper actor), sonic (scheme A), sonic_residual (scheme C checkpoint) or sonic_planner (scheme B)')
p.add_argument('--b-variant',choices=('B-open','B0','B1-0','B1-2','B1-4'),default=None,help='System1 scheme B variant (T02 v2); required with --controller sonic_planner')
p.add_argument('--device',default='cuda:0')
p.add_argument('--checkpoint-mass',choices=('evaluated','original'),default='evaluated',help='System1 T04 direct evaluation: with --object-mass-kg, validate the checkpoint contract against the original-mass experiment (checkpoint trained at the original mass) instead of the evaluated mass')
p.add_argument('--readback-only',action='store_true',help='System1 T04: build the env, write <output-dir>/object_physics_readback.json (mass, inertia, COM, material read back from PhysX) and exit without evaluating')
p.add_argument('--sonic-const-dq',default=None,help='System1 T03 C v2 diagnostic (scheme A only): constant joint offset added to every frame of SONIC\'s reference window, "joint=rad,joint=rad" (holosoma joint names); default none')
p.add_argument('--sonic-hip-pitch-actuator',choices=('ours','sonic'),default='ours',help='System1 sensitivity: SONIC hip_pitch effort/vel/armature')
p.add_argument('--finger-mode',choices=('locked','track','learn'),default=None,help='System1 T06/T07: G1 + Sharpa hands (bucket/table); policy checkpoints must be hand checkpoints of the same mode; off = rubber hands, unchanged')
p.add_argument('--hand-reference',type=Path,default=None,help='System1 T06/T07: 73-joint runtime reference (system1/hand/sharpa_reference.py); required with --finger-mode')
p.add_argument('--record-torques',action='store_true',help='System1 T04 v4: save every physics substep torque (post-clip, joint_control term) per episode to <state>.torques.npz and add per-joint saturation fractions (|tau| >= 0.95 x limit) to the episode result; off = unchanged')
a=p.parse_args()
if (a.finger_mode is None)!=(a.hand_reference is None):p.error('--finger-mode and --hand-reference go together')
if a.finger_mode is not None and a.case not in ('bucket','table'):p.error('--finger-mode: bucket/table for now (Push/Pull hands: plan5 path, separate)')
if a.finger_mode=='learn' and a.controller in ('sonic','sonic_planner'):p.error('A/B with hands: --finger-mode track (geometric grip) or locked')
if a.controller not in ('sonic','sonic_planner') and a.checkpoint is None:p.error('--checkpoint is required unless --controller sonic/sonic_planner')
if (a.controller=='sonic_planner')!=(a.b_variant is not None):p.error('--b-variant goes with --controller sonic_planner')
if a.sonic_const_dq is not None and a.controller!='sonic':p.error('--sonic-const-dq goes with --controller sonic')
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
        source_experiment=experiment
        experiment,mass_variant=core4d_experiment_with_mass(experiment,a.object_mass_kg)
    config=with_pair_experiment(g1_29dof_core4d_smalltable_smoke,experiment)
    if a.case in ('bucket','table'):
        config=replace(config,env_class='holosoma.envs.marl.core4d_bucket_manager.Core4DBucketManager',
                       simulator=replace(config.simulator,config=replace(config.simulator.config,enable_object_hand_contact=True)))
config=replace(config,observation=g1_29dof_core4d_smalltable_evaluation_observation,
               training=replace(config.training,num_envs=1,seed=241003,headless=True))
hand=None
if a.finger_mode is not None:  # System1 T06/T07 (off = unchanged)
    sys.path.insert(0,str(ROOT))
    from system1.hand.hand_config import with_sharpa_pair
    hand=dict(finger_mode=a.finger_mode,reference=str(a.hand_reference.resolve()),reference_sha256=sha(a.hand_reference))
    config=with_sharpa_pair(config,a.finger_mode,a.hand_reference)
config,placement_provenance=configure_push_placement(config,a.push_placement_proof,skill=a.case,root=ROOT)
if a.controller in ('sonic','sonic_residual','sonic_planner'):
    sys.path.insert(0,str(ROOT))
    from system1.scheme_a import with_sonic_pd
    config=with_sonic_pd(config,hip_pitch_actuator=a.sonic_hip_pitch_actuator)
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
        dev=a.device
        if a.controller!='policy':
            from system1 import sonic_controller as _sc
        if a.controller in ('sonic','sonic_planner'):
            from system1 import sonic_controller as _sc
            checkpoint=_sc.DEFAULT_CHECKPOINT;state={'iter':None,'plan5_skill':a.case,'placement_provenance':placement_provenance}
        else:
            checkpoint=a.checkpoint.resolve();state=torch.load(checkpoint,map_location=dev,weights_only=False)
        ablation=state.get('v3_ablation')
        if hand is not None and a.controller=='policy':  # hand MAPPO sizes before any checkpoint contract check
            from system1.hand.hand_mappo import patch_core4d_dims
            hand['mappo_dims']=patch_core4d_dims(a.finger_mode)
        if ablation is not None:
            from holosoma.agents.mappo.v3_plan5_ablation import ablation_contract
            if a.case!='pull' or ablation!=ablation_contract(from_scratch=ablation.get('from_scratch',False),zero_teammate_obs=ablation.get('zero_teammate_obs',False)):
                raise ValueError('Invalid V3 Pull ablation contract')
        zero_teammate_obs=bool(ablation and ablation['zero_teammate_obs'])
        if state.get('placement_provenance') != placement_provenance:
            raise ValueError('Checkpoint and evaluation placement differ')
        if state.get('plan5_skill',a.case) != a.case:
            raise ValueError('Checkpoint skill differs from evaluation case')
        if a.controller=='sonic_residual' and not state.get('system1_sonic_residual'):
            raise ValueError('--controller sonic_residual needs a scheme C checkpoint')
        if a.controller=='policy' and state.get('system1_sonic_residual'):
            raise ValueError('scheme C checkpoint requires --controller sonic_residual')
        if experiment and a.controller not in ('sonic','sonic_planner'):
            validate_core4d_smalltable_checkpoint(state,experiment_contract=(source_experiment if a.object_mass_kg is not None and a.checkpoint_mass=='original' else experiment).checkpoint_contract)
        if experiment:
            for path,expected in ((experiment.runtime_reference_file,experiment.runtime_reference_sha256),
                                  (experiment.object_urdf_file,experiment.object_urdf_sha256),
                                  (experiment.training_promotion_file,experiment.training_promotion_sha256)):
                if sha(ROOT/'src/holosoma'/path)!=expected:raise ValueError('Asset hash mismatch: '+path)
        elif a.controller not in ('sonic','sonic_planner'):
            c=state['plan5_mappo']
            if [c['num_agents'],c['actor_obs_dim'],c['critic_obs_dim'],c['action_dim']]!=[2,158,527,29]:raise ValueError('Plan5 shape contract')
        models=None
        if hand is not None and a.controller=='policy':
            if state.get('system1_finger_mode')!=a.finger_mode:raise ValueError('checkpoint finger mode differs from --finger-mode')
        if a.controller not in ('sonic','sonic_planner') and hand is not None:
            from system1.hand.hand_mappo import initialize_hand_model_bundle
            models=initialize_hand_model_bundle(config.algo.config,finger_mode=a.finger_mode,device=dev)
            models.actor.load_state_dict(state['actor_model_state_dict'],strict=True)
            models.actor_obs_normalizer.load_state_dict(state['actor_obs_normalizer_state_dict'],strict=True)
            models.actor.eval();models.actor_obs_normalizer.eval()
            def forbidden(*args,**kwargs):raise RuntimeError('Critic invoked during actor-only evaluation')
            models.critic.evaluate=forbidden
        elif a.controller not in ('sonic','sonic_planner'):
            models=initialize_core4d_smalltable_model_bundle(config.algo.config,device=dev)
            models.actor.load_state_dict(state['actor_model_state_dict'],strict=True)
            models.actor_obs_normalizer.load_state_dict(state['actor_obs_normalizer_state_dict'],strict=True)
            models.actor.eval();models.actor_obs_normalizer.eval()
            # A forbidden Critic call fails immediately rather than silently scoring with it.
            def forbidden(*args,**kwargs):raise RuntimeError('Critic invoked during actor-only evaluation')
            models.critic.evaluate=forbidden
        torch.manual_seed(241003)
        env=get_class(config.env_class)(get_tyro_env_config(config),device=dev)
        env.set_is_evaluating();command=env.command_manager.get_state('paired_motion_command')
        driver=None;cwrap=None
        if a.controller=='sonic':
            from system1.scheme_a import SonicDualDriver
            driver=SonicDualDriver(env,dev)
        elif a.controller=='sonic_planner':
            from system1.scheme_b_driver import SonicPlannerDualDriver
            driver=SonicPlannerDualDriver(env,dev,a.b_variant,a.case)
        elif a.controller=='sonic_residual':
            from system1.scheme_c import SonicResidualEnv
            cwrap=SonicResidualEnv(env,dev,tf32=False,kind=state.get('system1_sonic_residual_kind','v1'));driver=cwrap.driver  # T03 C v2: kind from the checkpoint (v1 if absent)
        const_dq=None
        if a.sonic_const_dq is not None:  # T03 C v2 diagnostic: constant dq on SONIC's input reference (scheme A path)
            const_dq=torch.zeros(env.num_envs,env.num_agents,len(env.dof_names),device=dev)
            for kv in a.sonic_const_dq.split(','):
                k,v=kv.split('=');const_dq[:,:,list(env.dof_names).index(k)]=float(v)
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
            if sensor is not None:result['hand_object_normal_force_w']=array(sensor.data.force_matrix_w[:,0].reshape(-1,2,sensor.data.force_matrix_w.shape[2]//2,3))  # rubber hands: [-1,2,2,3] as before
            return result
        captured=None;captured_sonic=None
        original_reward=env._compute_reward
        def capturing_reward():
            nonlocal captured,captured_sonic
            original_reward();captured=snapshot()
            if driver is not None:
                fr=min(int(env.episode_length_buf[0].item()),n-1)
                captured_sonic={k:v[0].detach().cpu().numpy().copy() for k,v in driver.tracking(command,fr).items()}
        env._compute_reward=capturing_reward
        _view=env.simulator._object.root_physx_view
        physics={k:v.detach().cpu().tolist() for k,v in dict(mass=_view.get_masses(),
                   material=_view.get_material_properties(),inertia_row_major=_view.get_inertias(),com_pose_xyzw=_view.get_coms()).items()}
        if a.readback_only:
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/'object_physics_readback.json',dict(case=a.case,object_mass_kg_arg=a.object_mass_kg,object_urdf=config.robot.object.object_urdf_path,
                       object_urdf_sha256=sha(ROOT/'src/holosoma'/config.robot.object.object_urdf_path),system1_object_mass_variant=mass_variant,physics=physics))
            return 0
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
        if hand is not None:manifest['system1_hand']=dict(hand,num_dof=int(env.num_dof),object_hand_contact_bodies=list(config.simulator.config.object_hand_contact_bodies))
        manifest['gpu_name']=torch.cuda.get_device_name(torch.device(dev)) if str(dev).startswith('cuda') else 'cpu'  # System1: GPU model of every evaluation
        if a.object_mass_kg is not None:manifest['system1_checkpoint_mass']=a.checkpoint_mass
        if cwrap is not None and cwrap.kind!='v1':manifest['system1_scheme_c_kind']=cwrap.kind  # T03 C v2
        if const_dq is not None:manifest['system1_sonic_const_dq']=dict(arg=a.sonic_const_dq,per_joint={n:float(v) for n,v in zip(env.dof_names,const_dq[0,0].tolist()) if v!=0.0})
        if driver is not None:
            manifest.update(controller=a.controller,inference=('sonic_release_teleop_encoder_action_mean+planner_onnx' if a.controller=='sonic_planner' else 'sonic_release_g1_encoder_action_mean') if cwrap is None else 'scheme_c_actor_mean_then_sonic',
                            sonic_checkpoint_sha256=sha(_sc.DEFAULT_CHECKPOINT),checkpoint_sha256=sha(checkpoint),
                            tf32=bool(torch.backends.cuda.matmul.allow_tf32),hip_pitch_actuator=a.sonic_hip_pitch_actuator,
                            frozen_action_scales=list(config.robot.control.frozen_action_scales),
                            effort_limits=list(config.robot.dof_effort_limit_list),vel_limits=list(config.robot.dof_vel_limit_list),
                            armature=list(config.robot.dof_armature_list),pd_target='joint_target_override (q* verbatim)',
                            sonic_spec=str(_sc.DEFAULT_SPEC),sonic_spec_sha256=sha(_sc.DEFAULT_SPEC),
                            sonic_pd=dict(hip_pitch_kp=float(env.p_gains.reshape(-1)[0].item()),p_gains=env.p_gains.reshape(-1).tolist(),
                                          d_gains=env.d_gains.reshape(-1).tolist(),action_scales=env.action_scales.reshape(-1).tolist()),
                            sonic_fall_rule='|dz pelvis|>0.25 or quat_err(pelvis)^2>1.0 or any ankle_roll/wrist_yaw |dz|>0.25 (SONIC eval terminations)')
        if a.controller=='sonic_planner':
            from system1 import scheme_b as _sb
            manifest.update(b_variant=a.b_variant,scheme_b=dict(_sb.preregistered_constants(),contacts_sha256=getattr(driver,'contacts_sha256',None),delta_m=driver.delta,position_feedback=driver.feedback))
        if a.record_torques:
            _tterm=env.action_manager.get_term('joint_control')
            _tlim=env.torque_limits.reshape(-1)[:env.num_dof].detach().cpu().numpy() if env.torque_limits.ndim<=2 else env.torque_limits[0,0].detach().cpu().numpy()
            manifest['system1_record_torques']=dict(source='joint_control.torques_substep (after clip_torques), every physics substep',
                                                    saturation_rule='|tau| >= 0.95 * torque limit',torque_limits=dict(zip(env.dof_names,_tlim.tolist())))
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
            start=snapshot();frames=[start];terms={};reward=0.;captured=None;sonic_frames=[];pd_diff=0.0;tq=[]
            if driver is not None:driver.reset()
            if cwrap is not None:
                cwrap._prev_t=command.time_steps.clone();cwrap._dones=torch.zeros_like(cwrap._dones)
            for step in range(1,n+1):
                if cwrap is not None:
                    actions=deterministic_core4d_smalltable_actions(models,observations,zero_teammate_obs=zero_teammate_obs)
                    observations,rewards,dones,extras=cwrap.step({'actions':actions})
                elif driver is not None:
                    actions,_=driver.actions(command) if const_dq is None else driver.actions(command,delta_q_h=const_dq)
                    observations,rewards,dones,extras=env.step({'actions':actions})
                else:
                    actions=deterministic_core4d_smalltable_actions(models,observations,zero_teammate_obs=zero_teammate_obs)
                    observations,rewards,dones,extras=env.step({'actions':actions})
                if captured is None:raise RuntimeError('Pre-reset state capture missing')
                frames.append(captured);reward+=float(rewards[0].item())
                if a.record_torques:tq.append(_tterm.torques_substep[0].detach().cpu().numpy().copy())
                if driver is not None:
                    sonic_frames.append(captured_sonic);pd_diff=max(pd_diff,driver.check_pd_target() or 0.0)
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
            if driver is not None:
                sch={k:np.stack([f[k] for f in sonic_frames]) for k in sonic_frames[0]}
                np.savez_compressed(out/(init['state_id']+'.sonic.npz'),**sch)
                fall=sch['fall']
                result['sonic_mpjpe_mm_per_robot']=sch['mpjpe_mm'].mean(axis=0).tolist()
                result['sonic_fall_step_per_robot']=[int(np.argmax(fall[:,r]))+1 if fall[:,r].any() else None for r in range(fall.shape[1])]
                result['sonic_fell_any']=bool(fall.any())
                result['sonic_mpjpe_pelvis_rel_mm_per_robot']=sch['mpjpe_pelvis_rel_mm'].mean(axis=0).tolist()
                result['sonic_pelvis_dxy_final_m_per_robot']=sch['pelvis_dxy_m'][-1].tolist()
                result['pd_target_minus_qstar_max_abs']=pd_diff
                if hasattr(driver,'episode_stats'):result.update(driver.episode_stats())
            result['trajectory']=str(out/(init['state_id']+'.trajectory.npz'))
            if a.record_torques:
                T_=np.stack(tq)  # [control step, substep, agent, dof]
                np.savez_compressed(out/(init['state_id']+'.torques.npz'),torques=T_.astype(np.float32),torque_limits=_tlim,dof_names=np.array(env.dof_names))
                sat=np.abs(T_)>=0.95*_tlim[None,None,None,:]
                result['torque_saturation_fraction_per_agent_joint']=sat.mean(axis=(0,1)).tolist()
                result['torque_substeps']=int(T_.shape[0]*T_.shape[1])
                result['torques']=str(out/(init['state_id']+'.torques.npz'))
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
