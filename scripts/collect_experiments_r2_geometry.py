#!/usr/bin/env python3
"""CPU reference metrics and factual evidence for the R2 handoff, no rendering."""
import argparse
import inspect
import shutil
import xml.etree.ElementTree as ET
from prepare_experiments_r2_cpu import *
from reference_geometry_math import BoxUnion, procrustes, map_robot
from reference_geometry_round2_correspondence import construct, fit_candidates

BATCH=ROOT/'logs/ReferenceGeometry/reference_geometry_pilot_20261017'
sys.path.insert(0,str(ROOT/'src/holosoma_retargeting'))
from holosoma_retargeting.dual_a1_layout import g1_sagittal_joint_map


def finite_distance(points, faces):
    patches=[p for f in faces for p in f['patches']]
    if not patches: raise ValueError('No finite exterior surface')
    lo=np.array([p['lo'] for p in patches]); hi=np.array([p['hi'] for p in patches])
    delta=np.asarray(points)[:,None,:]-np.clip(np.asarray(points)[:,None,:],lo,hi)
    return np.sqrt(np.sum(delta*delta,axis=-1).min(axis=1))


def u1(out,fk):
    rows=[]; details=[]; checks=[]
    for task in ['push','pull']:
        spec=read(GEOM/task/'correspondence_spec.json'); selection=read(GEOM/task/'selection.json')
        candidates={x['id']:x for x in spec['candidates']}
        source=np.load(BATCH/task/'preprocessed_source.npz'); q=source['robot_qpos']; obj=source['object_qpos']
        poses=fk.poses(q[:,:3],q[:,[4,5,6,3]],q[:,7:],HANDS)
        ro=Rotation.from_quat(obj[:,[4,5,6,3]]).as_matrix()
        local=np.einsum('tji,thj->thi',ro,poses[:,:,:3,3]-obj[:,:3][:,None])
        manifest=read(GEOM/task/'input_manifest.json'); sourcegeometry=BoxUnion(manifest['source_object']['path'])
        faces={f['id']:f for f in sourcegeometry.faces}
        interactions=read(BATCH/task/'source_interactions.json')['records']
        weighted=spec['weighted_source']; a=np.array(weighted['source_anchors']); w=np.array(weighted['weights'])
        for role,param in enumerate(selection['role_parameters']):
            candidate=candidates[param['id']]; region=spec['regions'][role]
            for placement in (['registered','prescribed'] if task=='push' else ['registered']):
                Q=np.array(param['Q']); c=np.array(param['c'])
                if placement=='prescribed':Q=np.eye(3);c=np.array([-.4 if role==0 else .4,0,0])
                residual=float(np.sqrt(np.sum(w*np.sum((a@Q.T+c-np.array(candidate['target_anchors']))**2,axis=1)))*1000)
                handrows=[]
                for h,name in enumerate(HANDS):
                    record=next(r for r in interactions if r['hand']==name)
                    allowed=np.zeros(len(q),bool)
                    for start,end in record['source_windows']: allowed[start:end]=True
                    distance=[]; frames=[]; omitted=[]
                    # Source windows stay frozen. Retain all competing finite faces active at the frame.
                    for t in np.flatnonzero(allowed):
                        ids={x['face_id'] for x in weighted['windows'] if x['hand']==name and x['start']<=t<x['end_exclusive']}
                        if not ids: omitted.append(int(t));continue
                        sf=[faces[x] for x in ids]
                        targets={candidate['face_map'][x]['target_face_id'] for x in ids}
                        tf=[x for x in region['faces'] if x['id'] in targets]
                        ds=finite_distance(local[t:t+1,h],sf)[0]
                        dt=finite_distance(local[t:t+1,h]@Q.T+c,tf)[0]
                        distance.append(abs(ds-dt));frames.append(int(t))
                    physical_hand=('right' if h==0 else 'left') if param['parity']<0 else ('left' if h==0 else 'right')
                    row=dict(task=task,role=role,parity=param['parity'],**{'placement(registered|prescribed)':placement},
                        separation_m=.8 if placement=='prescribed' else float(np.linalg.norm(np.array(selection['role_parameters'][1]['c'])-selection['role_parameters'][0]['c'])),
                        residual_rms_mm=residual,dd_mean_mm=float(np.mean(distance)*1000) if distance else '--',
                        dd_max_mm=float(max(distance)*1000) if distance else '--',n_contact_frames=len(distance),hand=physical_hand)
                    details.append(row);handrows.append(row)
                    checks.append(dict(task=task,role=role,placement=placement,source_hand=name,physical_hand=physical_hand,
                        valid_frames=frames,omitted_no_active_finite_face=omitted,delta_m=distance,
                        note='Source window is geometric proximity, not measured force/contact. Physical hands swap under parity -1.'))
                main={k:v for k,v in handrows[0].items() if k!='hand'}
                main.update(dd_mean_mm=float(np.mean([r['dd_mean_mm'] for r in handrows])) if all(r['n_contact_frames'] for r in handrows) else '--',
                    dd_max_mm=max(r['dd_max_mm'] for r in handrows) if all(r['n_contact_frames'] for r in handrows) else '--',
                    n_contact_frames=sum(r['n_contact_frames'] for r in handrows))
                rows.append(main)
    csv_write(out/'ref_role_synthesis.csv',rows);csv_write(out/'ref_role_synthesis_by_hand.csv',details)
    write(out/'checks/u1_finite_surface_checks.json',dict(records=checks,units='m internally, mm in CSV',
        scope='Existing source robot hand link origins; no additional wrist optimization during role construction',
        source_prior_note='Historical A1/learned rollout sources are preserved; this is not an ablation removing their historical wrist treatment',
        aggregation='Each role: equal two hand means, max of hand maxima, sum of hand-frame counts'))


def kick_check(out,fk):
    from holosoma_retargeting.dual_kick_reference import _symmetrize_object_qpos,_transport_robot_to_shared_object
    source=Path('/home/kevin/holosoma-rubber-hand-largetable/logs/WholeBodyTracking/20260807_034545-rubberhand_kick_sub16_028_8000_seed42-eval/kick_7999_seed42_physics_eval_v1.npz')
    data=PKG/'holosoma/data/motions/g1_29dof/whole_body_tracking'
    folder=data/'rubber_hand_largetable_v1/kick'
    viser=np.load(folder/'plan5_attempt09_dual_kick_mirrored_viser.npz')
    provenance=json.loads(str(viser['provenance']))
    assert sha(source)==provenance['source_sha256']
    with np.load(source) as z:
        a=provenance['source_attempt_start'];b=provenance['source_attempt_end_exclusive'];sl=slice(a,b)
        q=np.concatenate([z['root_pos'][sl],z['root_quat_xyzw'][sl][:,[3,0,1,2]],z['dof_pos'][sl]],axis=-1)
        obj=np.concatenate([z['object_pos_w'][sl],z['object_quat_xyzw'][sl][:,[3,0,1,2]]],axis=-1)
    shared=_symmetrize_object_qpos(obj); transported=_transport_robot_to_shared_object(q,obj,shared)
    src=BoxUnion(data/'objects_largetable.urdf');tgt=BoxUnion(data/'objects_widetable_plan5_pull_training.urdf')
    # This is a geometry-only equivalence test, NOT a new contact-window detector.
    # Five real tabletop faces (not the inner +Z cut plane) match both end regions.
    faces=[f for f in src.faces if 'tabletop' in f['part_id'] and not(f['axis']==2 and f['sign']==1)]
    anchors=[];windows=[]
    for f in faces:
        start=len(anchors);anchors.extend(f['corners'])
        windows.append(dict(face_id=f['id'],anchor_slice=[start,len(anchors)]))
    weighted=dict(source_anchors=np.array(anchors),weights=np.ones(len(anchors))/len(anchors),windows=windows)
    spec=construct('pull',src,tgt,weighted); candidates=fit_candidates(spec)
    # Explicit correspondence convention: preserve object local X/Y directions;
    # parity0=+1, parity1=-1 reflects object local Z. No old root offsets supplied.
    chosen=[]
    for role in range(2):
        want=np.eye(3) if role==0 else np.diag([1.,1.,-1.])
        matches=[c for c in candidates if c['role']==role and np.max(abs(c['fit']['Q']-want))<1e-10]
        assert len(matches)==1
        chosen.append(matches[0])
    joint_map,sign_mask=g1_sagittal_joint_map(np.array(JOINTS))
    new=np.stack([map_robot(transported,shared,c['fit']['Q'],c['fit']['c'],c['parity'],joint_map,sign_mask) for c in chosen],axis=1)
    runtime_path=folder/'plan5_attempt09_dual_kick_mirrored_runtime.npz'
    with np.load(runtime_path) as z:
        names=z['body_names'].tolist();pelvis=names.index('pelvis')
        runtime_root=z['agent_body_pos_w'][:,:,pelvis];runtime_quat=z['agent_body_quat_w'][:,:,pelvis][:,:,[1,2,3,0]]
        comparisons=dict(root_position_m=error(new[:,:,:3],runtime_root),root_rotation=quat_error(new[:,:,[4,5,6,3]],runtime_quat),
            joint_position_rad=error(new[:,:,7:],z['agent_joint_pos']),object_position_m=error(shared[:,:3],z['object_pos_w']),
            object_rotation=quat_error(shared[:,[4,5,6,3]],z['object_quat_w'][:,[1,2,3,0]]))
        ro=Rotation.from_quat(shared[:,[4,5,6,3]]).as_matrix()
        comparisons['object_local_root_m']=error(np.einsum('tji,taj->tai',ro,new[:,:,:3]-shared[:,None,:3]),np.einsum('tji,taj->tai',ro,runtime_root-shared[:,None,:3]))
    passed=all(x.get('max_abs',x.get('max_rad'))<2e-6 for x in comparisons.values())
    np.savez_compressed(out/'checks/kick_geometry_registered_reference.npz',robot_qpos=new,object_qpos=shared,fps=50,joint_names=JOINTS)
    write(out/'checks/kick_registration_equivalence.json',dict(status='geometry_pose_equivalent' if passed else 'not_equivalent',
        classification='Historical seed721 used legacy mirrored reference, now checked geometrically equivalent under stated finite-face correspondence; not retroactively trained with a registration pipeline.',
        source=identity(source),runtime=identity(runtime_path),viser=identity(folder/'plan5_attempt09_dual_kick_mirrored_viser.npz'),
        source_attempt=provenance,correspondence=spec,selected=chosen,comparisons=comparisons,tolerance_m_rad=2e-6,
        weighting='Uniform tabletop corner observations across five finite faces. Not the hand-proximity weights of Push/Pull; no Kick hand-window metric is claimed.',
        preprocessing='Existing object reflection-average + robot transport retained before O(3) map',
        training_decision='Keep the actual seed721 runtime, unchanged; no third retraining',
        velocity_comparison='Not replaced or recomputed; training keeps original runtime velocities',
        hand_window_status='Not measured yet; no foot value substituted into hand columns'))
    print('Kick geometry comparisons', comparisons,flush=True)


def facts(out,models,fk):
    from holosoma.config_values.marl.g1.reward import g1_29dof_plan5_push_reward
    from holosoma.config_values.marl.g1.core4d_smalltable_reward import g1_29dof_core4d_smalltable_reward
    from holosoma.config_values.marl.g1.termination import g1_29dof_plan5_push_termination
    from holosoma.config_values.marl.g1.core4d_smalltable_termination import g1_29dof_core4d_smalltable_termination
    from holosoma.managers.reward.terms.marl import UndesiredContacts
    from holosoma.managers.termination.terms.marl import JointBadTrackingZOnly
    terms={};links=[]
    for name,reward in [('Plan5',g1_29dof_plan5_push_reward),('CORE4D',g1_29dof_core4d_smalltable_reward)]:
        term=reward.terms['undesired_contacts'];pattern=term.params['undesired_contacts_body_names']
        terms[name]=asdict(term)
        for link in fk.u.link_map:
            if any(x in link for x in ('hand','wrist','knee','ankle','foot')):
                links.append(dict(family=name,link=link,regex_match=bool(re.match(pattern,link)),
                    weight=term.weight,force_threshold_N=term.params['threshold'],
                    availability='URDF link; runtime merging/contact-sensor membership requires actual simulator body list'))
    csv_write(out/'checks/contact_regularizer_link_mapping.csv',links)
    write(out/'checks/termination_and_contact_facts.json',dict(
        Plan5_termination=asdict(g1_29dof_plan5_push_termination),CORE4D_termination=asdict(g1_29dof_core4d_smalltable_termination),
        contact_terms=terms,termination_source=identity(inspect.getsourcefile(JointBadTrackingZOnly)),
        contact_source=identity(inspect.getsourcefile(UndesiredContacts)),
        hard_failure='any robot gravity_z_error >0.8 OR object world position error >0.25m OR object orientation angle >0.8rad',
        diagnostic_not_terminal=['torso low height','torso height error','selected body height error'],
        contact_force='agent_contact_forces_history: net contact force, no object-only filter; max history norm per body then body-count mean across robots',
        contact_status='soft reward penalty only, not a collision prohibition or early-stop condition',
        source_scope='Current sources; historical effective equivalence traced through existing unified evaluation source_audit, not inferred from git HEAD alone'))
    configs=[]
    for model in models:
        path=Path(model.get('config',Path(model['checkpoint']).parent/'run_config.json'));cfg=read(path)
        manifest=read(evaluation_result_dir(EVAL,model)/'run_manifest.json')
        scalar={k:v for k,v in cfg.items() if not isinstance(v,(list,dict))}
        configs.append(dict(model_id=model['model_id'],config=identity(path),scalar_fields=scalar,
            reward_terms=cfg.get('reward_terms'),reward_contract=cfg.get('bucket_reward_contract'),
            actual_evaluation_physics=manifest['physics'],reference_frames=manifest['reference_frames'],
            fps=1/manifest['dt'],first_last_sample_seconds=(manifest['reference_frames']-1)*manifest['dt'],
            full_horizon_seconds=manifest['reference_frames']*manifest['dt'],
            checkpoint_iteration=manifest['checkpoint_iteration'],body_names_to_track=BODY))
    write(out/'checks/per_model_settings.json',configs)
    # U2 dimensions are known. Distance rows stay explicitly missing until the
    # pre-wrist source-hand mapping and source proximity windows are checked.
    rows=[];scene=[]
    for task,exp in [('Chair','chair021'),('Small Table','smalltable5kg_A'),('Bucket','bucket003')]:
        e=get_core4d_pair_experiment(exp);directory=(PKG/e.runtime_reference_file).parent
        mp=directory/'source_manifest.json'
        if not mp.exists() and task=='Small Table':mp=directory.parent/'core4d_smalltable/source_manifest.json'
        m=read(mp);parent=None
        if task=='Chair':parent=Path(m['source_manifest'])
        if task=='Small Table':parent=Path(m['source_pair']).parent/'manifest.json'
        original=read(parent) if parent is not None else m
        heights=original['human_heights_m'];scales=m.get('source_human_to_robot_scales',m.get('human_to_robot_scales'))
        assert np.allclose(np.array(heights)*scales,1.32,atol=1e-10)
        scene.append(dict(task=task,experiment=asdict(e),source_manifest=identity(mp),original_manifest=identity(parent) if parent else identity(mp),
            h_i=heights,h_R=1.32,h_R_source=identity(ROOT/'src/holosoma_retargeting/holosoma_retargeting/config_types/robot.py'),
            scale_anchor_policy=original.get('scale_anchor_policy'),scale_anchors_m=original.get('scale_anchors_m'),
            source_hand_metric_status='pending pre-wrist source-frame/window alignment'))
        variants=[('nominal',1.),('re-optimized',1.)] if task=='Bucket' else [('one-sided',scales[0]),('mean',float(np.mean(scales)))]
        for variant,scale in variants:
            for i in range(2):
                rows.append(dict(task=task,participant=i,height_m=heights[i],s_i=scales[i],variant=variant,s_obj=scale,
                    size_mismatch_pct=abs(scale/scales[i]-1)*100,dd_mean_mm='--',dd_max_mm='--',n_contact_frames='--'))
    csv_write(out/'ref_shared_scene.csv',rows);write(out/'checks/shared_scene_sources.json',scene)
    audit=read(ROOT/'docs/experiments/retarget_method_comparison_20260925.json')
    wrist=[]
    for record in audit['chair']['hands']:
        wrist.append(dict(task='Chair',role=int(record['person'][-1])-1,hand=record['side'],target_source='source_SMPLX_palm_keypoints_archived_SO3_measurement',
            n_frames=audit['chair']['invariants']['shape'][0],before_mean_deg=record['baseline_orientation_deg']['mean'],
            after_mean_deg=record['refined_orientation_deg']['mean'],after_max_deg=record['refined_orientation_deg']['max']))
    csv_write(out/'wrist_orientation.csv',wrist)
    source=ROOT/audit['chair']['source_report'];assert sha(source)==audit['chair']['sha256']['comparison']
    write(out/'checks/wrist_sources.json',dict(audit=identity(ROOT/'docs/experiments/retarget_method_comparison_20260925.json'),
        archived_comparison=identity(source),mode='Archival matching full SO(3) metrics, not newly recomputed target frames',
        chair_overall_before_mean_deg=np.mean([x['before_mean_deg'] for x in wrist]),
        chair_overall_after_mean_deg=np.mean([x['after_mean_deg'] for x in wrist]),
        chair_worst_frame_deg=max(x['after_max_deg'] for x in wrist),
        remaining='Verify full target-frame axes and recompute metrics; other tasks not yet populated'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    assert (out/'cpu_stage_status.json').exists()
    if (out/'geometry_stage_status.json').exists():raise FileExistsError('Do not overwrite completed geometry stage')
    fk=BatchFK();u1(out,fk);kick_check(out,fk);facts(out,read(EVAL/'models.json'),fk)
    write(out/'geometry_stage_status.json',dict(return_code=0,script=identity(__file__),command=sys.argv,
        U1='Push/Pull complete',U2='dimensions complete, distance metrics pending',U6='geometry equivalence complete; training/eval not started',
        U8='partial facts and archived Chair metrics; full wrist recomputation pending'))


if __name__=='__main__':main()
