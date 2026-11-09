#!/usr/bin/env python3
"""Execute round2 offline correspondence, real-FK screens and paired export."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from reference_geometry_pilot import ROOT,RobotFK,write,sha,quaternion_wxyz_to_matrix,G1_29DOF_JOINT_NAMES,g1_sagittal_joint_map,draw_regions
from reference_geometry_math import BoxUnion,map_robot
from reference_geometry_round2_correspondence import tie_safe_weights,construct,fit_candidates
from reference_geometry_round2_checks import V0,CollisionProxy,surface_curves,support_metrics,motion_metrics,finite_max,summaries
from holosoma_retargeting.dual_pull_reference import DualPullReference
from holosoma_retargeting.dual_pull_runtime_reference import build_dual_pull_runtime_reference

BATCH=ROOT/'logs/ReferenceGeometry/reference_geometry_pilot_20261017'


def verify_batch():
    checked=[]
    for line in (BATCH/'SHA256SUMS').read_text().splitlines():
        digest,rel=line.split('  ',1)
        if sha(BATCH/rel)!=digest:raise ValueError('Batch1 changed: '+rel)
        checked.append(rel)
    for case in ['push','pull']:
        m=json.loads((BATCH/case/'input_manifest.json').read_text())
        records=[v for v in m.values() if isinstance(v,dict) and 'path' in v and v.get('sha256')]+m['dependencies']
        for r in records:
            if sha(r['path'])!=r['sha256']:raise ValueError('Actual input changed: '+r['path'])
    return dict(batch1=str(BATCH),verified_members=len(checked),all_declared_input_hashes_match=True,
        batch1_archive_sha256=sha(BATCH.with_suffix('.tar.gz')))


def save_curves(path,curves):
    np.savez_compressed(path.with_suffix('.npz'),**curves)
    cols=[k for k,v in curves.items() if np.asarray(v).ndim==1]
    with path.with_suffix('.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame']+cols)
        for t in range(len(curves[cols[0]])):
            w.writerow([t]+[float(curves[k][t]) if np.isfinite(curves[k][t]) else '' for k in cols])


def rotation_delta(a,b):
    shape=a.shape[:-1];ra=quaternion_wxyz_to_matrix(a.reshape(-1,4));rb=quaternion_wxyz_to_matrix(b.reshape(-1,4))
    return Rotation.from_matrix(ra@rb.transpose(0,2,1)).magnitude().reshape(shape)


def compare_array(a,b,quat=False):
    delta=rotation_delta(a,b) if quat else a-b
    return dict(max_abs=float(np.max(abs(delta))),rms=float(np.sqrt(np.mean(delta*delta))),
                representation='geodesic angle rad; quaternion signs equivalent' if quat else 'raw channel difference')


def audit_role(q,source,obj,fps,fk,geometry,engine,spec,candidate,source_curves,source_collision,source_poses):
    role=candidate['role'];fit=candidate['fit'];curves,poses=surface_curves(q,obj,fk,geometry,spec,role,candidate)
    curves.update(support_metrics(curves,source_curves,fps));curves.update(motion_metrics(q,source,obj,fps,fit['Q']))
    collision,witness=engine.evaluate([poses],obj)
    curves['nonhand_object_proxy_clearance_m']=collision['nonhand_object_clearance_m'][:,0]
    curves['hand_object_proxy_clearance_m']=collision['hand_object_convex_clearance_m'][:,0]
    curves['additional_nonhand_proxy_penetration_m']=np.maximum(0,-curves['nonhand_object_proxy_clearance_m'])-np.maximum(0,-source_collision['nonhand_object_clearance_m'][:,0])
    ro=quaternion_wxyz_to_matrix(obj[:,3:]);F=ro@fit['Q']@ro.transpose(0,2,1);errors=[]
    for name in fk.meshes:
        other=name
        if candidate['parity']<0:
            if name.startswith('left_'):other='right_'+name[5:]
            elif name.startswith('right_'):other='left_'+name[6:]
        ideal=obj[:,:3]+np.einsum('tij,tj->ti',F,source_poses[name][:,:3,3]-obj[:,:3])+np.einsum('tij,j->ti',ro,fit['c'])
        errors.append(np.linalg.norm(poses[other][:,:3,3]-ideal,axis=1))
    curves['actual_fk_link_origin_error_m']=np.max(errors,axis=0)
    limits=np.array([[j.limit.lower,j.limit.upper] for j in fk.u.actuated_joints]);curves['joint_limit_excess_rad']=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0).max(1)
    normal=max(np.degrees(np.arccos(np.clip(np.dot(fit['Q']@f['source_normal'],f['target_normal']),-1,1))) for f in candidate['face_map'].values())
    failures=[];checks=[]
    def bound(name,value,limit):
        maximum=finite_max(value);passed=maximum is not None and maximum<=limit+1e-10
        checks.append(dict(name=name,max=maximum,limit=limit,passed=passed))
        if not passed:failures.append(name if maximum is not None else name+':missing_support')
    bound('normal_mapping_deg',[normal],V0['normal_error_deg'])
    for key,lim in [('actual_fk_link_origin_error_m','fk_origin_error_m'),('joint_limit_excess_rad','numeric_tolerance'),
          ('quaternion_norm_error','numeric_tolerance'),('root_rotation_det_error','numeric_tolerance'),
          ('relative_gravity_deviation_deg','relative_gravity_deviation_deg'),
          ('root_position_step_added_m','max_added_root_step_m'),('root_rotation_step_added_rad','max_added_root_rotation_step_rad'),
          ('root_speed_added_m_s','additional_root_speed_m_s'),('root_acceleration_added_m_s2','additional_root_acceleration_m_s2'),
          ('additional_nonhand_proxy_penetration_m','additional_nonhand_proxy_penetration_m')]:bound(key,curves[key],V0[lim])
    for hand in fk.hands:
        for suffix in ['region_min_distance_m','region_near_fraction','penetration_m']:
            delta=curves[hand+'_'+suffix]-source_curves[hand+'_'+suffix]
            if suffix=='region_near_fraction':delta=-delta
            key=hand+'_added_'+suffix;curves[key]=delta
            bound(key,delta,V0[{'region_min_distance_m':'additional_region_min_distance_m','region_near_fraction':'region_near_fraction_drop','penetration_m':'additional_hand_penetration_m'}[suffix]])
    for side in ['left','right']:
        curves[side+'_foot_ground_penetration_m']=np.maximum(0,-curves[side+'_foot_bottom_z_m'])
        curves[side+'_foot_added_ground_penetration_m']=curves[side+'_foot_ground_penetration_m']-np.maximum(0,-source_curves[side+'_foot_bottom_z_m'])
        bound(side+'_foot_added_ground_penetration_m',curves[side+'_foot_added_ground_penetration_m'],V0['additional_foot_ground_penetration_m'])
        bound(side+'_support_added_speed_m_s',curves[side+'_support_added_speed_m_s'],V0['additional_support_speed_m_s'])
        bound(side+'_support_added_drift_m',curves[side+'_support_added_drift_m'],V0['additional_support_drift_m'])
    return dict(id=candidate['id'],role=role,fit=fit,passed=not failures,failures=failures,checks=checks,
         scope=V0['acceptance_scope'],summary=summaries(curves),collision_witnesses=witness),curves,poses


def run_case(case,out,fk,spec,m):
    folder=out/case;z=np.load(BATCH/case/'preprocessed_source.npz');source=z['robot_qpos'];obj=z['object_qpos'];fps=int(z['fps'])
    src=BoxUnion(m['source_object']['path']);target=BoxUnion(m['target_object']['path'])
    source_curves,source_poses=surface_curves(source,obj,fk,src,spec)
    source_engine=CollisionProxy(fk,src);source_collision,source_witness=source_engine.evaluate([source_poses],obj)
    source_curves.update(support_metrics(source_curves,source_curves,fps));source_curves.update(motion_metrics(source,source,obj,fps,np.eye(3)))
    source_curves['nonhand_object_proxy_clearance_m']=source_collision['nonhand_object_clearance_m'][:,0]
    source_curves['hand_object_proxy_clearance_m']=source_collision['hand_object_convex_clearance_m'][:,0]
    save_curves(folder/'source_checks',source_curves);write(folder/'source_checks_summary.json',dict(summary=summaries(source_curves),collision_witnesses=source_witness))
    engine=CollisionProxy(fk,target);write(folder/'collision_proxy_definition.json',dict(meshes=engine.mesh_info,
         method='Analytic separating axes: per-link collision-mesh local bounding box (OBB) vs exact object box; OBB vs OBB for cross-robot. Selected-frame convex refinement separately reported. No integration.',
         hand_collision_proxy='Not computed; use independent actual hand mesh surface samples with exact object boundary instead.',
         distance_cap_m=.1,cap_meaning='Positive SAT margin is a separation lower bound, capped at0.1m; negative is proxy penetration along a separating axis, not concave-mesh penetration or force.',
         self_collision_checked=False,scope='robot-object and cross-robot only; source internal joint/mesh self-overlap not assessed'))
    candidates=fit_candidates(spec);write(folder/'fitted_candidates.json',candidates)
    index,sign=g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES);cache={};audits=[]
    for i,c in enumerate(candidates):
        if c['fit']['status']!='solved':raise ValueError('Unexpected degenerate correspondence')
        q=map_robot(source,obj,c['fit']['Q'],c['fit']['c'],c['parity'],index,sign)
        audit,curves,poses=audit_role(q,source,obj,fps,fk,target,engine,spec,c,source_curves,source_collision,source_poses)
        name=f'candidate_{i:02d}';save_curves(folder/name,curves)
        np.savez_compressed(folder/(name+'_reference.npz'),robot_qpos=q,object_qpos=obj,fps=fps,joint_names=G1_29DOF_JOINT_NAMES)
        audit['files_prefix']=name;audits.append(audit);cache[c['id']]=(c,q,poses,curves,audit)
        print(case,name,c['id'],'PASS' if audit['passed'] else 'FAIL '+','.join(audit['failures']),flush=True)
    write(folder/'candidate_checks.json',audits)
    pairs=[];acceptable=[]
    left=[c for c in candidates if c['role']==0];right=[c for c in candidates if c['role']==1]
    for a,b in product(left,right):
        pair_id=a['id']+' || '+b['id'];reasons=[]
        for c in [a,b]:
            if not cache[c['id']][4]['passed']:reasons.append('role_screen_failed:'+c['id'])
        item=dict(id=pair_id,roles=[a['id'],b['id']],joint_weighted_rmse_m=float(np.sqrt((a['fit']['weighted_rmse_m']**2+b['fit']['weighted_rmse_m']**2)/2)),failures=reasons)
        if not reasons:
            collision,witness=engine.evaluate([cache[a['id']][2],cache[b['id']][2]],obj)
            item['robot_robot_min_proxy_clearance_m']=float(collision['robot_robot_clearance_m'].min())
            item['collision_witnesses']=witness
            if item['robot_robot_min_proxy_clearance_m']< -V0['pair_proxy_penetration_m']:item['failures'].append('robot_robot_convex_proxy_penetration')
            np.savez_compressed(folder/f'pair_{len(pairs):02d}_checks.npz',**collision)
        else:item['pair_check_skipped_reason']='At least one role already fails frozen single-role checks; not a feasible pair.'
        item['passed']=not item['failures'];pairs.append(item)
        if item['passed']:acceptable.append(item)
    if acceptable:
        chosen=sorted(acceptable,key=lambda p:(round(p['joint_weighted_rmse_m']/V0['fit_tie_tolerance_m']),p['id']))[0]
        status='passes_source_relative_geometry'
    else:
        chosen=sorted(pairs,key=lambda p:(sum(len(cache[k][4]['failures']) for k in p['roles'])+len(p['failures']),round(p['joint_weighted_rmse_m']/V0['fit_tie_tolerance_m']),p['id']))[0]
        status='diagnostic_only'
    selected=[cache[k] for k in chosen['roles']];pair=np.stack([v[1] for v in selected],axis=1)
    collision,witness=engine.evaluate([v[2] for v in selected],obj);save_curves(folder/'selected_pair_collision',dict(robot_robot_clearance_m=collision['robot_robot_clearance_m'],
        role0_nonhand_object_clearance_m=collision['nonhand_object_clearance_m'][:,0],role1_nonhand_object_clearance_m=collision['nonhand_object_clearance_m'][:,1]))
    write(folder/'pair_candidates.json',dict(total=len(pairs),evaluated_pair_geometry=sum('robot_robot_min_proxy_clearance_m' in p for p in pairs),records=pairs))
    selection=dict(status=status,selected=chosen,role_parameters=[dict(id=v[0]['id'],parity=v[0]['parity'],**v[0]['fit']) for v in selected],
        selection_rule='Passing pairs only: minimum joint weighted fit error; <=1e-10m quantized tie resolved by stable geometry ID. If none pass, diagnostic minimum failure count then same fit/ID rule.',
        collision_witnesses=witness,scope=V0['acceptance_scope'],correspondence_sha256=sha(folder/'correspondence_spec.json'))
    write(folder/'selection.json',selection)
    np.savez_compressed(folder/'new_pair_reference.npz',robot_qpos=pair,object_qpos=obj,fps=fps,joint_names=G1_29DOF_JOINT_NAMES,status=status)
    metadata={'case':case,'scope':status,'method':'round2 finite region Procrustes','lateral_axis':'table_local_x' if case=='push' else 'table_local_z',
              'mirror_plane':None if case=='push' else 'table_local_xy','lateral_leg_offset_m':None,
              'role_transforms':json.loads((folder/'selection.json').read_text())['role_parameters'],'note':'Generic pose expansion reused; no old shift parameter used.'}
    ref=DualPullReference(pair[:,0],pair[:,1],obj,fps);runtime=build_dual_pull_runtime_reference(ref,provenance=metadata)
    runtime.save(folder/'new_runtime.npz');ref.save_canonical_rollout(folder/'new_pair_viser.npz',provenance=metadata)
    # No old parameters were consumed by construction, fitting or selection.
    old=np.load(BATCH/case/'legacy_pair_reference.npz');oldq=old['robot_qpos']
    assert np.array_equal(obj,old['object_qpos'])
    comparisons=dict(object_trajectory_bit_exact=True,root_translation_m=compare_array(pair[:,:,:3],oldq[:,:,:3]),
        root_rotation_rad=compare_array(pair[:,:,3:7],oldq[:,:,3:7],True),joints_rad=compare_array(pair[:,:,7:],oldq[:,:,7:]))
    oldcurves=[];newcurves=[];oldposes=[]
    for role,v in enumerate(selected):
        # Evaluate legacy geometry against this same selected correspondence.
        c=dict(v[0]);oldQ=np.eye(3) if c['parity']>0 else np.diag([1.,1.,-1.]);ro=quaternion_wxyz_to_matrix(obj[:,3:])
        oldlocal=np.einsum('tji,tj->ti',ro,oldq[:,role,:3]-obj[:,:3]);srclocal=np.einsum('tji,tj->ti',ro,source[:,:3]-obj[:,:3])
        oldc=np.mean(oldlocal-srclocal@oldQ.T,axis=0)
        c['fit']=dict(c['fit'],Q=oldQ,c=oldc)
        audit,curves,poses=audit_role(oldq[:,role],source,obj,fps,fk,target,engine,spec,c,source_curves,source_collision,source_poses)
        write(folder/f'old_role{role}_screen.json',audit)
        save_curves(folder/f'old_role{role}_checks',curves);oldcurves.append(curves);newcurves.append(v[3]);oldposes.append(poses)
        save_curves(folder/f'new_role{role}_checks',v[3])
        write(folder/f'old_role{role}_summary.json',summaries(curves));write(folder/f'new_role{role}_summary.json',summaries(v[3]))
    oldcollision,_=engine.evaluate(oldposes,obj);save_curves(folder/'old_pair_collision',dict(robot_robot_clearance_m=oldcollision['robot_robot_clearance_m'],
        role0_nonhand_object_clearance_m=oldcollision['nonhand_object_clearance_m'][:,0],role1_nonhand_object_clearance_m=oldcollision['nonhand_object_clearance_m'][:,1]))
    before=np.load(BATCH/case/'legacy_runtime_recomputed.npz');after=np.load(folder/'new_runtime.npz');runtime_delta={}
    for key in before.files:
        if key not in after.files or not np.issubdtype(before[key].dtype,np.number):continue
        if before[key].shape!=after[key].shape:raise ValueError('Runtime schema mismatch:'+key)
        runtime_delta[key]=compare_array(after[key],before[key],'quat' in key)
    comparisons['runtime']=runtime_delta
    write(folder/'old_new_comparison.json',comparisons)
    rows=[]
    for role in [0,1]:
        for key,values in newcurves[role].items():
            if key not in oldcurves[role] or np.asarray(values).ndim!=1:continue
            a=np.asarray(oldcurves[role][key],float);b=np.asarray(values,float);valid=np.isfinite(a)&np.isfinite(b)
            if not valid.any():continue
            for scope,mask in [('all_available_frames',valid),('source_interaction_frames',valid & np.any([np.isfinite(newcurves[role][h+'_region_min_distance_m']) for h in fk.hands],axis=0))]:
                if not mask.any():continue
                delta=b[mask]-a[mask];rows.append(dict(role=role,metric=key,scope=scope,frames=int(mask.sum()),old_mean=float(a[mask].mean()),new_mean=float(b[mask].mean()),old_max=float(a[mask].max()),new_max=float(b[mask].max()),delta_max_abs=float(abs(delta).max()),delta_rms=float(np.sqrt(np.mean(delta*delta)))))
    with (folder/'old_new_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(case,'SELECTED',status,chosen['id'],comparisons['root_translation_m'],flush=True)
    return dict(case=case,status=status,role_candidate_counts=spec['accepted_per_role'],pair_count=len(pairs),passing_pairs=len(acceptable),selected_roles=chosen['roles'],old_new_pose_comparison={k:v for k,v in comparisons.items() if k!='runtime'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False);verification=verify_batch();write(out/'batch1_verification.json',verification)
    commit=subprocess.check_output(['git','rev-parse','private/main'],cwd=ROOT,text=True).strip()
    doc=subprocess.check_output(['git','show',commit+':docs/paper_handoff/ubuntu_reference_geometry_round2_20261017.md'],cwd=ROOT)
    (out/'REQUEST_FROM_MAIN.md').write_bytes(doc)
    write(out/'execution_identity.json',dict(instruction_commit=commit,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),training=False,physics_steps=0))
    write(out/'frozen_screen_thresholds.json',V0)
    prepared={}
    for case in ['push','pull']:
        folder=out/case;folder.mkdir();m=json.loads((BATCH/case/'input_manifest.json').read_text());write(folder/'input_manifest.json',m)
        interaction=json.loads((BATCH/case/'source_interactions.json').read_text());curves=np.load(BATCH/case/'source_distances.npz')
        src=BoxUnion(m['source_object']['path']);target=BoxUnion(m['target_object']['path']);weighted=tie_safe_weights(interaction,curves)
        spec=construct(case,src,target,weighted);write(folder/'correspondence_spec.json',spec);prepared[case]=(spec,m)
        print(case,'Frozen candidates:',spec['attempted_per_role'],'->',spec['accepted_per_role'],flush=True)
        points={f'role{r}':np.concatenate([c['target_anchors'] for c in spec['candidates'] if c['role']==r]) for r in [0,1]}
        draw_regions(folder/'target_anchors.png',target,points,title=case+' actual finite surface anchors; no cut walls')
    fk=RobotFK();results=[]
    for case in ['push','pull']:
        spec,m=prepared[case];results.append(run_case(case,out,fk,spec,m))
    write(out/'summary.json',results)

if __name__=='__main__':main()
