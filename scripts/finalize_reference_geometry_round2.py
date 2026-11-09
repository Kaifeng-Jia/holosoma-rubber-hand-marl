#!/usr/bin/env python3
"""Complete diagnostics, old screens, ambiguity-aware views and regression checks."""
import argparse
import json
import subprocess
import numpy as np
from pathlib import Path
from reference_geometry_pilot import ROOT,RobotFK,write,sha,quaternion_wxyz_to_matrix
from reference_geometry_math import BoxUnion
from reference_geometry_round2_correspondence import construct,tie_safe_weights,fit_candidates
from reference_geometry_round2_checks import CollisionProxy,rect_distance,hull_box_sat,V0,summaries
from run_reference_geometry_round2 import BATCH,audit_role,save_curves


def grouped_regions(case,folder,fk,spec,candidates,obj,oldq,newq):
    original=np.load(BATCH/case/'source_distances.npz');records=[]
    ro=quaternion_wxyz_to_matrix(obj[:,3:]);T=len(obj)
    for role,candidate in enumerate(candidates):
        for variant,q in [('old',oldq[:,role]),('new',newq[:,role])]:
            poses=fk.poses(q,fk.hands);curves={}
            for h,name in enumerate(fk.hands):
                actual=fk.hands[1-h] if candidate['parity']<0 else name;pose=poses[actual]
                world=np.einsum('tij,pj->tpi',pose[:,:3,:3],fk.dense[actual])+pose[:,:3,3][:,None]
                local=np.einsum('tji,tpj->tpi',ro,world-obj[:,:3][:,None]);groups={'certain':{},'edge':{}}
                for t in range(T):
                    active={w['face_id']:w for w in spec['weighted_source']['windows'] if w['hand']==name and w['start']<=t<w['end_exclusive']}
                    ids=sorted(active)
                    if not ids:continue
                    support=np.array([(abs(original[name+'_signed_distance_m'][t])<=.03)&((original[name+'_competing_face_bits'][t]&(np.uint64(1)<<np.uint64(active[f]['face_index'])))!=0) for f in ids])
                    count=support.sum(0)
                    for kind,mask in [('certain',count==1),('edge',count>1)]:
                        labels=tuple(ids[j] for j in range(len(ids)) if np.any(support[j]&mask))
                        if labels:groups[kind].setdefault(labels,[]).append(t)
                for kind,byfaces in groups.items():
                    d=np.full(T,np.nan);fraction=d.copy()
                    for labels,frames in byfaces.items():
                        target_ids={candidate['face_map'][fid]['target_face_id'] for fid in labels}
                        faces=[f for f in spec['regions'][role]['faces'] if f['id'] in target_ids]
                        distances=rect_distance(local[frames],faces);d[frames]=distances.min(1);fraction[frames]=(distances<=.03).mean(1)
                    curves[name+'_'+kind+'_region_min_m']=d;curves[name+'_'+kind+'_region_near_fraction']=fraction
                    records.append(dict(role=role,variant=variant,source_hand=name,actual_hand=actual,association=kind,
                        frames=int(np.isfinite(d).sum()),min_distance_mean_m=float(np.nanmean(d)) if np.isfinite(d).any() else None,
                        near_fraction_mean=float(np.nanmean(fraction)) if np.isfinite(d).any() else None))
            save_curves(folder/f'{variant}_role{role}_association_checks',curves)
    write(folder/'association_evaluation.json',dict(records=records,
        source_classification='First-batch near samples: exactly one active nearest face is certain; >1 competing active faces is edge. Labels are geometric, not physical contact truth.',
        evaluation='Distance of actual independent dense hand mesh samples to union of the corresponding real finite target faces. Certain and edge unions evaluated separately; a face may have both sorts of source support.',
        screening='Combined compatible-face union was used in screening; separated reports are descriptive, no thresholds were tuned.',
        no_palm_normal_multi_face_alignment=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;fk=RobotFK();regression=[]
    for case in ['push','pull']:
        folder=out/case;m=json.loads((folder/'input_manifest.json').read_text());selection=json.loads((folder/'selection.json').read_text())
        src=BoxUnion(m['source_object']['path']);target=BoxUnion(m['target_object']['path']);z=np.load(BATCH/case/'preprocessed_source.npz')
        source=z['robot_qpos'];raw=z['raw_robot_qpos'];obj=z['object_qpos'];fps=int(z['fps']);T=len(obj)
        interaction=json.loads((BATCH/case/'source_interactions.json').read_text());w=tie_safe_weights(interaction,np.load(BATCH/case/'source_distances.npz'));spec=construct(case,src,target,w)
        candidates={c['id']:c for c in fit_candidates(spec)};chosen=[candidates[k] for k in selection['selected']['roles']]
        old=np.load(BATCH/case/'legacy_pair_reference.npz')['robot_qpos'];new=np.load(folder/'new_pair_reference.npz')['robot_qpos']
        source_curves=dict(np.load(folder/'source_checks.npz'));source_poses=fk.poses(source,list(fk.meshes));engine=CollisionProxy(fk,target)
        source_collision=dict(nonhand_object_clearance_m=source_curves['nonhand_object_proxy_clearance_m'][:,None])
        for role,c in enumerate(chosen):
            ro=quaternion_wxyz_to_matrix(obj[:,3:]);Q=np.eye(3) if c['parity']>0 else np.diag([1.,1.,-1.])
            local=np.einsum('tji,tj->ti',ro,old[:,role,:3]-obj[:,:3]);slocal=np.einsum('tji,tj->ti',ro,source[:,:3]-obj[:,:3]);offset=np.mean(local-slocal@Q.T,0)
            legacy=dict(c);legacy['fit']=dict(c['fit'],Q=Q,c=offset)
            audit,curves,_=audit_role(old[:,role],source,obj,fps,fk,target,engine,spec,legacy,source_curves,source_collision,source_poses)
            save_curves(folder/f'old_role{role}_checks',curves);write(folder/f'old_role{role}_screen.json',audit)
        grouped_regions(case,folder,fk,spec,chosen,obj,old,new)
        # Refine worst selected non-hand OBB overlap, without changing selection.
        audits=json.loads((folder/'candidate_checks.json').read_text());refinements=[]
        for role,c in enumerate(chosen):
            record=next(a for a in audits if a['id']==c['id']);witnesses=record['collision_witnesses']
            if not witnesses:continue
            worst=min(witnesses,key=lambda r:r['distance_m']);t=worst['frame'];i,j=worst['geom_indices'];name=engine.names[i%engine.n];part=target.parts[j-2*engine.n]
            pose=fk.poses(new[t:t+1,role],[name])[name][0];vertices,normals,edges=engine.hulls[name];R=quaternion_wxyz_to_matrix(obj[t,3:])
            margin=hull_box_sat(vertices@pose[:3,:3].T+pose[:3,3],normals@pose[:3,:3].T,edges@pose[:3,:3].T,obj[t,:3]+R@part['center'],part['size']/2,R)
            refinements.append(dict(role=role,frame=t,link=name,object_part=part['id'],OBB_signed_margin_m=worst['distance_m'],convex_hull_signed_margin_m=margin,
                implication='Convex proxy is separated here' if margin>0 else 'Convex proxy overlaps; not exact concave-mesh penetration',used_for_selection=False))
        write(folder/'selected_overlap_refinement.json',refinements)
        # Foot source transport diagnostics: keep raw-vs-transport distinction.
        rawposes=fk.poses(raw,['left_ankle_roll_link','right_ankle_roll_link']);foot=[]
        for side in ['left','right']:
            name=side+'_ankle_roll_link';pose=rawposes[name];v=fk.meshes[name].vertices
            rawz=(np.einsum('tij,pj->tpi',pose[:,:3,:3],v)+pose[:,:3,3][:,None])[:,:,2].min(1)
            foot.append(dict(side=side,raw_min_bottom_z_m=float(rawz.min()),preprocessed_min_bottom_z_m=float(source_curves[side+'_foot_bottom_z_m'].min()),source_support_frames=int(source_curves[side+'_source_support_mask'].sum())))
        write(folder/'source_transport_ground_diagnostic.json',dict(world_ground_z_m=0,feet=foot,preprocessing=m['preprocessing'],
            note='Actual URDF mesh vertex extrema; preprocessed source is the existing symmetric-object transport contract. No ground offset or pose repair is applied.'))
        if case=='pull':
            oldc=np.load(folder/'old_pair_collision.npz');newc=np.load(folder/'selected_pair_collision.npz')
            residual=max(float(np.max(abs(oldc[k]-newc[k]))) for k in oldc.files)
            assert residual<1e-9,residual
            regression.append(dict(test='Full Pull old/new equivalent poses give same analytic proxy results',max_error_m=residual,passed=True))
        # Preserve exact root displacement norms separately from component RMS.
        delta=np.linalg.norm(new[:,:,:3]-old[:,:,:3],axis=-1)
        write(folder/'root_displacement_norm.json',dict(max_m=float(delta.max()),mean_m=float(delta.mean()),rms_m=float(np.sqrt(np.mean(delta*delta)))))
    tests=subprocess.run([__import__('sys').executable,'-B','-m','unittest','discover','-s','tests','-p','test_reference_geometry_round2.py','-v'],cwd=ROOT,text=True,capture_output=True)
    (out/'targeted_tests.txt').write_text(tests.stdout+tests.stderr+'\nExit code: '+str(tests.returncode)+'\n');assert tests.returncode==0
    regression.append(dict(test='Targeted clipping, tie weights, full correspondence, SAT sign/rigid-invariance tests',passed=True,count=5))
    write(out/'regression_checks.json',regression)

if __name__=='__main__':main()
