#!/usr/bin/env python3
"""Freeze region ambiguity, test real FK reflection, reproduce legacy references.

No old spacing/offset is consumed until correspondence_spec.json is saved.
The leg-only registration is an explicitly incomplete geometry diagnostic.
It is never exported as a full-role reference or presented as a selected pair.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from reference_geometry_pilot import (ROOT,DATA,ROBOT,MARL,RobotFK,write,sha,identity,artifact,
    transform,draw_regions,quaternion_wxyz_to_matrix,G1_29DOF_JOINT_NAMES,g1_sagittal_joint_map)
from reference_geometry_math import BoxUnion,procrustes,signed_permutations,map_robot,origin
from holosoma_retargeting.dual_pull_reference import synthesize_dual_pull_reference,DualPullReference
from holosoma_retargeting.dual_pull_runtime_reference import build_dual_pull_runtime_reference
import xml.etree.ElementTree as ET


def freeze(case,folder):
    manifest=json.loads((folder/'input_manifest.json').read_text())
    src=BoxUnion(manifest['source_object']['path']);target=BoxUnion(manifest['target_object']['path'])
    interactions=json.loads((folder/'source_interactions.json').read_text())
    srcids=sorted({w['part_id'] for r in interactions['records'] for w in r['stable_face_windows']})
    tabletop=next(p for p in src.parts if p['kind']=='tabletop');wide=next(p for p in target.parts if p['kind']=='tabletop')
    # Stable complete source tuple is deliberately retained, including transient
    # tabletop windows. There is no implicit change to a leg-only source task.
    roleparity=[1,1] if case=='push' else [1,-1]
    axis=0 if case=='push' else 2
    options=[]
    for option in ['two_half_surfaces','source_width_end_surfaces']:
        lo,hi=wide['lo'][axis],wide['hi'][axis];width=tabletop['size'][axis]
        intervals=[[lo,0.],[0.,hi]] if option=='two_half_surfaces' else [[lo,lo+width],[hi-width,hi]]
        options.append(dict(id=option,proposed_only=True,axis=axis,role_intervals_m=intervals,
             derivation='Target symmetry plane' if option=='two_half_surfaces' else 'Actual target outer boundary plus source tabletop metric width',
             no_robot_offset_specified=True,not_selected_or_evaluated=True))
    spec=dict(case=case,status='needs_role_regions',source_geometry_sha256=manifest['source_object']['sha256'],
        target_geometry_sha256=manifest['target_object']['sha256'],hand_identity_and_windows=interactions['records'],
        complete_source_part_tuple=srcids,role_parities=roleparity,
        reflection_hand_mapping={'left':'right','right':'left'} if case=='pull' else None,
        target_part_tuples=[{'largetable_tabletop':wide['id'],**({'largetable_leg_neg_x_neg_z':p['id']} if case=='pull' else {})}
                           for p in (target.parts if case=='pull' else [wide]) if case=='push' or p['kind']=='leg'],
        source_tabletop_size_m=tabletop['size'],target_tabletop_size_m=wide['size'],
        complete_role_candidate_count=None,complete_pair_candidate_count=None,
        why_count_undefined='Continuous tabletop region bounds not specified; no finite complete correspondence list exists yet. Not a truncated enumeration.',
        geometric_precheck='Whole source and target tabletop face widths differ; do not treat them as isometric or normalize their sizes.',
        target_face_corner_permutations='D4 only once per unique face; pending finite role-region bounds',
        weights='Each active hand total1/2; per-hand window duration normalized; equal weights for corners within window; pending complete region correspondence',
        thresholds=json.loads((folder/'source_thresholds.json').read_text()),
        manual_regions=None,region_options_for_author=options,
        source_geometry_observation=('Both hands have top/bottom slab proximity during original A1 penetration.' if case=='push' else
            'Both hands approach the same negative-X negative-Z source leg, but stable tabletop top/bottom/edge windows also occur. Full tuple is leg+tabletop, not two legs alone.'),
        blocked_exports=['full-role candidate paired reference','candidate runtime','candidate-vs-old metrics/images/replay','pair collision/gravity/support feasibility decision'],
        reason='Need explicit finite tabletop regions for both roles, or author decision to investigate a separately labelled leg-only fitting formulation with all other windows retained for validation.',
        old_parameters_used=False)
    write(folder/'correspondence_spec.json',spec)
    # Show exact candidate input choices without calculating robot translations.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,opt in zip(axs,options):
        ax.add_patch(Rectangle(wide['lo'][[0,2]],wide['size'][0],wide['size'][2],fill=False,linewidth=2))
        for i,(lo,hi) in enumerate(opt['role_intervals_m']):
            base=wide['lo'][[0,2]].copy();size=wide['size'][[0,2]].copy();j=0 if axis==0 else 1;base[j]=lo;size[j]=hi-lo
            ax.add_patch(Rectangle(base,*size,alpha=.2,color=['tab:blue','tab:orange'][i],label=f'Proposed role{i}'))
            ax.text(base[0]+size[0]/2,base[1]+size[1]/2,f'role {i}',ha='center',va='center')
        for p in target.parts:
            if p['kind']=='leg':
                ax.add_patch(Rectangle(p['lo'][[0,2]],p['size'][0],p['size'][2],color='gray'))
        ax.set_xlim(wide['lo'][0]-.15,wide['hi'][0]+.15);ax.set_ylim(wide['lo'][2]-.15,wide['hi'][2]+.15)
        ax.set_aspect('equal');ax.set_xlabel('Object X(m)');ax.set_ylabel('Object Z(m)');ax.set_title(opt['id']+' — NOT selected')
    fig.tight_layout();fig.savefig(folder/'target_region_options.png',dpi=180);plt.close(fig)
    return spec,src,target


def leg_subtuple(folder,src,target,interactions):
    sources=[];weights=[];entries=[]
    for record in interactions['records']:
        windows=[w for w in record['stable_face_windows'] if 'leg_' in w['part_id']]
        duration=sum(w['duration_s'] for w in windows)
        if not duration:return dict(status='no_stable_leg_subtuple')
        for w in windows:
            anchors=np.array(w['anchors']);sources.extend(anchors)
            weights.extend([.5*w['duration_s']/duration/len(anchors)]*len(anchors))
            entries.append(dict(hand=record['hand'],window=[w['start'],w['end_exclusive']],face_id=w['face_id']))
    source=np.array(sources);weights=np.array(weights)
    part=next(p for p in src.parts if p['id']=='largetable_leg_neg_x_neg_z')
    records=[]
    for parity in [1,-1]:
        for p in target.parts:
            if p['kind']!='leg':continue
            for index,orientation in enumerate(signed_permutations(parity)):
                if not np.allclose(abs(orientation)@part['size'],p['size'],atol=1e-10):continue
                b=(source-part['center'])@orientation.T+p['center']
                fit=procrustes(source,b,weights,parity)
                records.append(dict(id=f'{parity:+d}/{p["id"]}/axis_order{index:02d}',target_part=p['id'],parity=parity,
                                    corners_generated_from='Matched actual equal-sized leg box corners with one consistent signed-axis permutation',**fit))
    report=dict(scope='Leg subset only; other stable tabletop windows are intentionally NOT represented here. This is not a full-role candidate or success result.',
       source_windows=entries,source_anchors=source,weights=weights,candidate_count=len(records),
       candidates=records,selected=None,old_offsets_used=False)
    write(folder/'leg_subtuple_diagnostic.json',report)
    return report


def check_fk(folder,fk):
    z=np.load(folder/'preprocessed_source.npz',allow_pickle=False);q=z['robot_qpos'];obj=z['object_qpos']
    index,sign=g1_sagittal_joint_map(G1_29DOF_JOINT_NAMES);J=np.diag([1.,-1,1]);Q=np.diag([1.,1,-1])
    assert np.array_equal(index[index],np.arange(29));assert np.array_equal(sign*sign[index],np.ones(29));assert np.array_equal(J@J,np.eye(3))
    reflected=map_robot(q,obj,Q,np.zeros(3),-1,index,sign)
    p=fk.poses(q,fk.hands);r=fk.poses(reflected,fk.hands);ro=quaternion_wxyz_to_matrix(obj[:,3:]);F=ro@Q@ro.transpose(0,2,1)
    original_normals=[];records=[];data={}
    for i,name in enumerate(fk.hands):
        other=fk.hands[1-i];ideal_pos=obj[:,:3]+np.einsum('tij,tj->ti',F,p[name][:,:3,3]-obj[:,:3])
        errors=np.linalg.norm(ideal_pos-r[other][:,:3,3],axis=1);data[name+'_mirror_origin_error_m']=errors
        frames=np.linspace(0,len(q)-1,16,dtype=int);source_world=np.einsum('tij,pj->tpi',p[name][frames,:3,:3],fk.dense[name])+p[name][frames,:3,3][:,None]
        ideal=obj[frames,:3,None].transpose(0,2,1)+np.einsum('tij,tpj->tpi',F[frames],source_world-obj[frames,:3][:,None])
        target_local=np.einsum('tji,tpj->tpi',r[other][frames,:3,:3],ideal-r[other][frames,:3,3][:,None])
        _,distance,_=trimesh.proximity.closest_point(fk.meshes[other],target_local.reshape(-1,3))
        data[name+'_mirror_mesh_distance_m']=distance.reshape(len(frames),-1)
        records.append(dict(source_hand=name,target_hand=other,all_frame_link_origin_max_m=float(errors.max()),
             actual_mirrored_mesh_distance_mean_m=float(distance.mean()),actual_mirrored_mesh_distance_max_m=float(distance.max()),
             mesh_check_frames=frames,independent_samples_per_frame=1024))
    fixed=[];tree=ET.parse(ROBOT)
    for hand in fk.hands:
        joint=next(j for j in tree.findall('joint') if j.find('child').get('link')==hand)
        parent=joint.find('parent').get('link');tf=origin(joint.find('origin'));poses=fk.poses(q,[hand,parent])
        err=float(np.max(np.abs(poses[parent]@tf-poses[hand])))
        fixed.append(dict(hand=hand,parent=parent,fixed_joint_name=joint.get('name'),local_transform=tf,all_frames_matrix_error=err))
    limits=np.array([[j.limit.lower,j.limit.upper] for j in fk.u.actuated_joints]);excess=np.maximum(np.maximum(limits[:,0]-q[:,7:],q[:,7:]-limits[:,1]),0)
    report=dict(scope='Reflection/FK mechanism test on true source; zero translation, no candidate selected and no geometric feasibility claim',
         source_joint_limit_excess_max_rad=float(excess.max()),source_joint_limit_excess_max_deg=float(np.rad2deg(excess.max())),
         DP_double_application_exact=True,J_double_application_exact=True,
         root_rotation_determinant_max_error=float(np.max(abs(np.linalg.det(quaternion_wxyz_to_matrix(reflected[:,3:7]))-1))),
         actual_mesh_mirror_errors=records,fixed_hand_chain_checks=fixed)
    write(folder/'robot_fk_mechanism_check.json',report);np.savez_compressed(folder/'robot_fk_mechanism_check.npz',**data)


def baseline(case,folder):
    # Called only after the complete-role ambiguity/region-input rule is frozen.
    assert (folder/'correspondence_spec.json').exists()
    m=json.loads((folder/'input_manifest.json').read_text());c=json.loads(Path(m['formal_config']['path']).read_text())
    z=np.load(folder/'preprocessed_source.npz');q=z['robot_qpos'];obj=z['object_qpos'];fps=int(z['fps'])
    if case=='push':
        spacing=float(c['lateral_spacing_m']);ro=quaternion_wxyz_to_matrix(obj[:,3:]);pair=np.repeat(q[:,None,:],2,axis=1)
        pair[:,:,:3]+=spacing*.5*ro[:,:,0][:,None]*np.array([-1.,1.])[None,:,None]
        metadata=dict(method='Exact position formula of PairedA1Reference online implementation',lateral_spacing_m=spacing,
            implementation=identity(ROOT/'src/holosoma/holosoma/envs/marl/paired_a1_reference.py'))
    else:
        old=np.load(m['old_pair_reference']['path']);offset=float(old['lateral_leg_offset_m'])
        original=np.c_[z['raw_robot_qpos'],z['raw_object_qpos']]
        result=synthesize_dual_pull_reference(original,fps=fps,lateral_leg_offset_m=offset)
        pair=np.stack([result.robot_a_qpos,result.robot_b_qpos],axis=1)
        assert np.allclose(pair,np.stack([old['robot_a_qpos'],old['robot_b_qpos']],axis=1),atol=1e-12)
        assert np.allclose(obj,old['shared_object_qpos'],atol=1e-12)
        metadata=dict(method='Existing synthesis function vs actual saved pair',lateral_leg_offset_m=offset,
            qpos_max_abs_difference=float(np.max(abs(pair-np.stack([old['robot_a_qpos'],old['robot_b_qpos']],axis=1)))),
            original_pair_sha256=m['old_pair_reference']['sha256'])
    np.savez_compressed(folder/'legacy_pair_reference.npz',robot_qpos=pair,object_qpos=obj,fps=fps,joint_names=G1_29DOF_JOINT_NAMES)
    # Standard runtime interface with proper manifold differences, never mj_step.
    runtime=build_dual_pull_runtime_reference(DualPullReference(pair[:,0],pair[:,1],obj,fps),provenance={'scope':'Legacy reproduction only','case':case})
    runtime.save(folder/'legacy_runtime_recomputed.npz')
    metadata['runtime_note']='Common existing FK/velocity expansion; legacy Pull loader fields can be compared, Push online channels follow original runtime shift implementation.'
    metadata['correspondence_spec_frozen_before_old_parameters_sha256']=sha(folder/'correspondence_spec.json')
    write(folder/'legacy_reproduction.json',metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();fk=RobotFK()
    for case in ['push','pull']:
        folder=a.output/case;spec,src,target=freeze(case,folder)
        if case=='pull':leg_subtuple(folder,src,target,json.loads((folder/'source_interactions.json').read_text()))
        check_fk(folder,fk);baseline(case,folder)
        print(case,'regions frozen; actual FK checked; legacy reference reproduced',flush=True)

if __name__=='__main__':main()
