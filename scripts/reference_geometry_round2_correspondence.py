"""Round-2 finite actual-surface clipping, tie-safe weights and O(3) candidates.

No old pair, spacing or outward offset is accepted by this module.
"""
from itertools import product
import numpy as np
from reference_geometry_math import signed_permutations, procrustes


def clipped_faces(geometry, axis, interval):
    """Intersect actual exposed rectangles; NEVER create a cut-plane wall."""
    result=[]
    for f in geometry.faces:
        patches=[]
        for patch in f['patches']:
            lo=np.array(patch['lo']).copy();hi=np.array(patch['hi']).copy()
            lo[axis]=max(lo[axis],interval[0]);hi[axis]=min(hi[axis],interval[1])
            tangential=[j for j in range(3) if j!=f['axis']]
            if np.any(hi<lo-1e-12) or np.any((hi-lo)[tangential]<=1e-12):continue
            patches.append(dict(lo=lo,hi=hi))
        if patches:
            lo=np.min([p['lo'] for p in patches],0);hi=np.max([p['hi'] for p in patches],0)
            tangential=[j for j in range(3) if j!=f['axis']]
            area=sum(np.prod((p['hi']-p['lo'])[tangential]) for p in patches)
            # Present assets yield rectangles. Reject holes instead of using
            # their enclosing box as a fictitious interaction surface.
            if not np.isclose(area,np.prod((hi-lo)[tangential]),atol=1e-12):
                raise ValueError('Nonrectangular clipped face requires explicit patch correspondence')
            corners=[]
            for bits in product([0,1],repeat=2):
                point=lo.copy();point[tangential]=np.where(bits,hi[tangential],lo[tangential]);corners.append(point)
            result.append(dict(id=f['id'],part_id=f['part_id'],normal=f['normal'],axis=f['axis'],
                sign=f['sign'],patches=patches,corners=np.array(corners),area_m2=float(area)))
    return result


def tie_safe_weights(interactions, curves):
    """Equal hand totals; duration credit split over unique frame/sample support.

    One active frame has one unit of evidence per hand. Each eligible sample
    receives equal mass, then divides it among its competing stable windows.
    Thus a corner sample with 3 equally-near faces contributes once, not 3x.
    Original windows (including gap-filled spans) are all retained verbatim.
    """
    records=[];all_anchors=[];all_weights=[];all_windows=[]
    active=[r for r in interactions['records'] if r['stable_face_windows']]
    for record in active:
        h=record['hand'];windows=record['stable_face_windows'];count=len(windows)
        ties=curves[h+'_competing_face_bits'];near=abs(curves[h+'_signed_distance_m'])<=.03
        credit=np.zeros(count);unique_samples=0;ambiguous_samples=0;certain_samples=0;frames_used=0
        for t in range(len(ties)):
            win=[i for i,w in enumerate(windows) if w['start']<=t<w['end_exclusive']]
            if not win:continue
            support=np.array([near[t]&((ties[t]&(np.uint64(1)<<np.uint64(windows[i]['face_index'])))!=0) for i in win])
            multiplicity=support.sum(0);valid=multiplicity>0
            if not np.any(valid):continue
            credit[win]+=(support[:,valid]/multiplicity[valid]).sum(1)/valid.sum()
            unique_samples+=int(valid.sum());ambiguous_samples+=int((multiplicity>1).sum());certain_samples+=int((multiplicity==1).sum());frames_used+=1
        if not credit.sum():raise ValueError(f'No actual support for {h}')
        window_weights=credit/credit.sum()/len(active)
        for i,w in enumerate(windows):
            anchors=np.array(w['anchors']);start=len(all_anchors);all_anchors.extend(anchors)
            all_weights.extend([window_weights[i]/len(anchors)]*len(anchors))
            all_windows.append(dict(**w,hand=h,anchor_slice=[start,len(all_anchors)],
                effective_duration_frames=float(credit[i]),normalized_weight=float(window_weights[i])))
        records.append(dict(hand=h,unique_active_frames=frames_used,unique_support_samples=unique_samples,
            certain_support_samples=certain_samples,ambiguous_support_samples=ambiguous_samples,
            total_weight=float(window_weights.sum()),window_count=len(windows),zero_weight_window_count=int((credit==0).sum())))
    a=np.array(all_anchors);w=np.array(all_weights)
    if not np.isclose(w.sum(),1):raise ValueError('Invalid total weight')
    return dict(source_anchors=a,weights=w,windows=all_windows,hand_summaries=records,
        rule='Equal hand total; per-frame unique support samples share one duration credit, competing stable faces split each sample credit; equal corners within each retained window. No evidence credited to unsupported gap frames.')


def adjacency(geometry):
    parts=[];faces=[]
    for i,a in enumerate(geometry.parts):
        for b in geometry.parts[i+1:]:
            gap=np.maximum(np.maximum(a['lo']-b['hi'],b['lo']-a['hi']),0)
            parts.append(dict(parts=[a['id'],b['id']],center_delta=b['center']-a['center'],
                axis_separation_m=gap,minimum_gap_m=float(np.linalg.norm(gap))))
    for i,a in enumerate(geometry.faces):
        for b in geometry.faces[i+1:]:
            if a['part_id']!=b['part_id']:continue
            shared=[x for x in a['corners'] if np.any(np.linalg.norm(b['corners']-x,axis=1)<1e-10)]
            if shared:faces.append(dict(faces=[a['id'],b['id']],shared_corner_count=len(shared)))
    return dict(part_relations=parts,within_part_face_adjacency=faces,
        definition='Actual metric center relations and minimum box gap; shared corners/edges for same-part faces. No artificial contact for separated parts.')


def construct(case,src,target,weighted):
    axis=0 if case=='push' else 2
    slab=next(p for p in src.parts if p['kind']=='tabletop')
    wide=next(p for p in target.parts if p['kind']=='tabletop')
    width=slab['size'][axis];intervals=[[wide['lo'][axis],wide['lo'][axis]+width],[wide['hi'][axis]-width,wide['hi'][axis]]]
    parities=[1,1] if case=='push' else [1,-1]
    active_ids=sorted({w['face_id'] for w in weighted['windows']});source_faces={f['id']:f for f in src.faces}
    active_parts=sorted({source_faces[f]['part_id'] for f in active_ids})
    legs=[p for p in src.parts if p['id'] in active_parts and p['kind']=='leg']
    if len(legs)>1:raise ValueError('Unexpected new source leg tuple; do not silently change mapping')
    attempts=[];valid=[];regions=[]
    for role,(interval,parity) in enumerate(zip(intervals,parities)):
        rf=clipped_faces(target,axis,interval);region_center=wide['center'].copy();region_center[axis]=sum(interval)/2
        region_size=wide['size'].copy();region_size[axis]=width
        regions.append(dict(role=role,axis=axis,interval_m=interval,parity=parity,faces=rf))
        targetlegs=[p for p in target.parts if p['kind']=='leg' and p['lo'][axis]>=interval[0]-1e-10 and p['hi'][axis]<=interval[1]+1e-10] if legs else [None]
        for leg in targetlegs:
            for order,O in enumerate(signed_permutations(parity)):
                cid=f'role{role}/{leg["id"] if leg else wide["id"]}/axes{order:02d}'
                rejected=[];mapping={slab['id']:wide['id']}
                if leg:mapping[legs[0]['id']]=leg['id']
                if not np.allclose(abs(O)@slab['size'],region_size,atol=1e-10):rejected.append('tabletop_metric_dimensions')
                offset=region_center-O@slab['center']
                if leg:
                    if not np.allclose(abs(O)@legs[0]['size'],leg['size'],atol=1e-10):rejected.append('leg_metric_dimensions')
                    if not np.allclose(O@(legs[0]['center']-slab['center']),leg['center']-region_center,atol=1e-10):rejected.append('leg_tabletop_relative_geometry')
                face_map={};target_anchors=np.empty_like(weighted['source_anchors'])
                if not rejected:
                    for fid in active_ids:
                        sf=source_faces[fid];normal=O@sf['normal']
                        matches=[f for f in rf if f['part_id']==mapping[sf['part_id']] and np.allclose(f['normal'],normal,atol=1e-10)]
                        if len(matches)!=1:
                            rejected.append('missing_real_exterior_face:'+fid);continue
                        tf=matches[0];expected=sf['corners']@O.T+offset
                        distances=np.linalg.norm(expected[:,None]-tf['corners'][None],axis=-1);indices=distances.argmin(1)
                        if len(set(indices.tolist()))!=4 or distances[np.arange(4),indices].max()>1e-9:
                            rejected.append('finite_corner_or_adjacency_mismatch:'+fid);continue
                        face_map[fid]=dict(target_face_id=tf['id'],source_normal=sf['normal'],target_normal=tf['normal'],
                            source_corners=sf['corners'],target_corners=tf['corners'][indices],corner_permutation=indices,
                            target_patches=tf['patches'])
                    if not rejected:
                        for window in weighted['windows']:
                            start,end=window['anchor_slice'];target_anchors[start:end]=face_map[window['face_id']]['target_corners']
                attempts.append(dict(id=cid,role=role,parity=parity,part_map=mapping,rejection_reasons=rejected,
                    status='geometry_correspondence_accepted' if not rejected else 'geometry_pruned'))
                if not rejected:valid.append(dict(id=cid,role=role,parity=parity,part_map=mapping,face_map=face_map,
                    target_anchors=target_anchors,region_index=role))
    # Candidate identities/corners all arise from actual finite boundaries.
    # The consistent rectangular face tuple induces the finite signed-axis
    # orders; no per-face independent corner shuffle or nonrigid fit allowed.
    return dict(case=case,axis=axis,source_tabletop_width_m=float(width),regions=regions,
        source_adjacency=adjacency(src),target_adjacency=adjacency(target),weighted_source=weighted,
        attempts=attempts,candidates=valid,
        enumeration='All signed-axis orders at prescribed parity and all within-role leg tuples; metric shape and shared-corner consistency prefilter. D4 order derived consistently from tuple, not per-window.',
        attempted_per_role=[sum(a['role']==i for a in attempts) for i in [0,1]],
        accepted_per_role=[sum(a['role']==i for a in valid) for i in [0,1]],
        old_parameters_used=False,cut_plane_faces_created=False)


def fit_candidates(spec):
    ws=spec['weighted_source'];result=[]
    for candidate in spec['candidates']:
        fit=procrustes(ws['source_anchors'],candidate['target_anchors'],ws['weights'],candidate['parity'])
        result.append(dict(**candidate,fit=fit))
    return result
