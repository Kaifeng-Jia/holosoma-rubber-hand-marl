#!/usr/bin/env python3
"""Frozen fed77e6 U2 surface-distance supplement. CPU only, read-only inputs."""
from __future__ import annotations
import argparse,ast,csv,hashlib,importlib.metadata as md,json,os,platform,shutil,subprocess,sys,time,traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/holosoma_retargeting'))
from holosoma_retargeting.data_utils.core4d_adapter import load_canonical_core4d_sequence,resample_core4d_pair_sequence
from holosoma_retargeting.data_utils.core4d_retarget import CORE4D_A1_PALM_LANDMARKS,_physical_object_qpos,prepare_core4d_person_retarget_input

COMMIT='fed77e627081bd6116d364e8d2fe77617a9984de'
HANDS=['left_rubber_hand_link','right_rubber_hand_link']
PALM=[[20,25,28,31],[21,40,43,46]]
ROBOT=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
ROBOT_SHA='7ed217f28ed6e3b1fa864bf0aadd527ed5ad319361ecefaeccd56e81306a4a25'
FIELDS=['task','variant','participant','hand','n_window_frames','dist_mean_mm','dist_median_mm','dist_p90_mm','frac_inside']
PER_FIELDS=['task','variant','participant','hand','frame','time_s','in_source_window','source_dist_mm','dist_mm','inside']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n')
def csv_write(p,rows,fields):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def error(a,b):
    d=np.asarray(a)-np.asarray(b);return dict(max_abs=float(abs(d).max()),rms=float(np.sqrt(np.mean(d*d))))

def specs():
    chair=ROOT/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908'
    table=Path('/home/kevin/datasets/CORE4D-V1-full/retarget_results/method_comparison_20260904/20231030_001_desk001_move')
    bucket=ROOT/'logs/Core4DPreviews/bucket003_20231020_071_20260918'
    models=Path('/home/kevin/datasets/CORE4D-V1-full/CORE4D_Real/object_models')
    return [dict(task='Chair',slug='chair',folder=chair,stage=chair/'two_stage',canonical=chair/'source_canonical.npz',
       canonical_sha='e496e71901d50808b1d29d533b358117c64a1301b5244dfa134850fb8d729acc',
       pair=chair/'shared_small_preview/core4d_pair_reference.npz',pair_sha='d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586',
       mesh=models/'chair/chair021_m.obj',mesh_sha='7fb3283f1b67c8d08bb1f68c51d15aab5e6c940b8571e704bca265701b776161',
       frames=235,source_frames=118,scales=[.7616803111061523,.7267058142765197],variants=['one-sided','mean']),
      dict(task='Small Table',slug='small_table',folder=table,stage=table/'two_stage_fullnominal',
       canonical=Path('/home/kevin/datasets/CORE4D-V1-full/derived/20231030_001_desk001_v1_canonical.npz'),
       canonical_sha='5e27c929999bde64187ec5d71d4da02ea93bb12f78fedfcb36f7200e7990ed90',
       pair=table/'small_table_nominal_preview/core4d_pair_reference.npz',pair_sha='d9d17e96f5f0b73aa76a1bf32f1ec50cfb7fa7fe78a847dd890882add4bbfd05',
       mesh=models/'desk/desk001_m.obj',mesh_sha='f8f9dffa9f1ccdbd2ff90125cd39c96e79bf84ccebd8a81fcef656c4fbc7c00b',
       frames=413,source_frames=207,scales=[.7477371848535067,.7347195863845842],variants=['one-sided','mean']),
      dict(task='Bucket',slug='bucket',folder=bucket,stage=bucket/'two_stage',canonical=bucket/'source_canonical.npz',
       canonical_sha='1208deb13bf70daee2d42343c6728656cca8586ae46efc7a2bb718d514079083',
       pair=bucket/'two_stage/core4d_pair_reference.npz',pair_sha='089d3a36a39442e055f01a8111d6145475c062928c28727062f951887020ceec',
       mesh=models/'bucket/bucket003_m.obj',mesh_sha='2a95d824af8facbd363ce00eba0534cd7c341b51f9997e3b137990f5fb503bdf',
       frames=299,source_frames=150,scales=[.7614736204510514,.7222941753069272],variants=['nominal','re-optimized'])]

def literal_keyword(path,variable,keyword):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in node.targets):
            return ast.literal_eval(next(k.value for k in node.value.keywords if k.arg==keyword))
    raise ValueError((path,variable,keyword))

def make_fk():
    """Execute the unchanged BatchFK AST only, avoiding unrelated training imports."""
    helper=ROOT/'scripts/prepare_experiments_r2_cpu.py'
    names=literal_keyword(ROOT/'src/holosoma/holosoma/config_values/robot.py','g1_29dof','dof_names')
    body=literal_keyword(ROOT/'src/holosoma/holosoma/config_values/marl/g1/command.py','motion_config','body_names_to_track')
    node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BatchFK')
    namespace=dict(np=np,Rotation=Rotation,yourdfpy=yourdfpy,JOINTS=names,BODY=body,HANDS=HANDS,ROBOT=ROBOT,error=error)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),namespace)
    return namespace['BatchFK'](),dict(source=str(helper),sha256=sha(helper),
             extraction='Unmodified BatchFK class AST; constants read from actual robot/command declarations; no simulator-manager imports',joint_names=names,body_names=body)

def exact_distance(mesh,points):
    p=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if not np.isfinite(p).all():raise ValueError('Nonfinite query')
    result=[]
    for start in range(0,len(p),128):
        _,d,_=trimesh.proximity.closest_point(mesh,p[start:start+128]);result.append(d)
    d=np.concatenate(result) if result else np.empty(0)
    if not np.isfinite(d).all() or np.any(d<0):raise ValueError('Invalid exact distance')
    return d

def brute_distance(mesh,points):
    """Independent all-triangles candidate set, no BVH pruning/vertex sampling."""
    tri=np.asarray(mesh.triangles);out=[]
    for p in points:
        q=trimesh.triangles.closest_point(tri,np.broadcast_to(p,(len(tri),3)))
        out.append(float(np.linalg.norm(q-p,axis=1).min()))
    return np.asarray(out)

def winding_numbers(mesh,points):
    tri=np.asarray(mesh.triangles,dtype=np.float64);out=[]
    for k in range(0,len(points),32):
        v=tri[None]-np.asarray(points[k:k+32])[:,None,None]
        norm=np.linalg.norm(v,axis=-1)
        a,b,c=v[:,:,0],v[:,:,1],v[:,:,2]
        numerator=np.einsum('pti,pti->pt',a,np.cross(b,c))
        denominator=norm.prod(axis=-1)+np.einsum('pti,pti->pt',a,b)*norm[:,:,2]+np.einsum('pti,pti->pt',b,c)*norm[:,:,0]+np.einsum('pti,pti->pt',c,a)*norm[:,:,1]
        out.extend((2*np.arctan2(numerator,denominator).sum(axis=1)/(4*np.pi)).tolist())
    return np.asarray(out)

def inside_flags(mesh,points,dist_world):
    """Closed oriented mesh only: solid-angle winding and four ray parities agree."""
    p=np.asarray(points);flags=np.full(len(p),-1,dtype=np.int8);boundary=np.asarray(dist_world)<=1e-8
    info=dict(is_watertight=bool(mesh.is_watertight),is_winding_consistent=bool(mesh.is_winding_consistent),
              volume_source_units_m3=float(mesh.volume),boundary_count=int(boundary.sum()),boundary_tolerance_world_m=1e-8)
    if not (mesh.is_watertight and mesh.is_winding_consistent and np.isfinite(mesh.volume) and mesh.volume>0):
        return flags,dict(info,reliable_topology=False,reason='Nonclosed/inconsistent/nonpositive oriented volume; no geometry repair',unknown=len(p))
    flags[boundary]=0;ids=np.flatnonzero(~boundary);w=winding_numbers(mesh,p[ids])
    integer=np.rint(w);valid=np.isfinite(w)&(abs(w-integer)<1e-5)&np.isin(integer,[0,1]);parities=[]
    directions=np.array([[.4395064455,.617598629942,.652231566745],[-.7123,.3311,.6187]])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    for direction in [directions[0],-directions[0],directions[1],-directions[1]]:
        result=[]
        for k in range(0,len(ids),128):
            q=p[ids[k:k+128]]
            _,ray,_=mesh.ray.intersects_location(q,np.broadcast_to(direction,q.shape).copy(),multiple_hits=True)
            result.extend((np.bincount(ray,minlength=len(q))%2).tolist())
        parities.append(np.asarray(result))
    ray_ok=np.all(np.stack(parities)==integer[None],axis=0);valid &= ray_ok
    flags[ids[valid]]=integer[valid].astype(np.int8)
    return flags,dict(info,reliable_topology=True,method='generalized winding solid angle + two fixed bidirectional ray parity crosscheck',
          directions=directions,ray_backend=type(mesh.ray).__module__+'.'+type(mesh.ray).__name__,
          winding_integer_max_error=float(abs(w-integer).max()) if len(w) else 0,
          ray_winding_disagreement=int(np.count_nonzero(~ray_ok)),unknown=int(np.sum(flags<0)),
          caveat='Inside refers to the supplied triangulated closed solid, not a physical hollow-object/material contact assertion')

def local_points(points,qpos,scale=1.):
    rot=Rotation.from_quat(qpos[:,[4,5,6,3]]).as_matrix()
    return np.einsum('tji,t...j->t...i',rot,points-qpos[:,:3].reshape(len(qpos),*[1]*(points.ndim-2),3))/scale

def stat_row(task,variant,person,hand,dist,inside,mask):
    n=int(mask.sum());d=dist[mask]*1000;v=inside[mask]
    return dict(task=task,variant=variant,participant=person,hand=hand,n_window_frames=n,
        dist_mean_mm=float(d.mean()) if n else '--',dist_median_mm=float(np.median(d)) if n else '--',
        dist_p90_mm=float(np.percentile(d,90,method='linear')) if n else '--',
        frac_inside=float(v.mean()) if n and np.all(v>=0) else '--')

class Collector:
    def __init__(self,out):self.out=out;self.records=[];self.seen={}
    def add(self,path,kind='inputs',parents=None):
        path=Path(path).resolve();h=sha(path);key=(str(path),h)
        if key in self.seen:return self.seen[key]
        rel=Path('sources')/kind/(h[:12]+'_'+path.name);dest=self.out/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        record=dict(original_path=str(path),package_path=str(rel),sha256=h,parents=parents or [])
        self.records.append(record);self.seen[key]=record;return record

def mesh_transforms(urdf,xml,mesh_path):
    u=ET.parse(urdf);records=[]
    for visual in u.findall('.//visual'):
        node=visual.find('geometry/mesh')
        if node is None:continue
        p=Path(node.get('filename'));p=p if p.is_absolute() else Path(urdf).parent/p
        assert p.resolve()==mesh_path.resolve()
        origin=visual.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
        rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
        scale=np.fromstring(node.get('scale','1 1 1'),sep=' ')
        assert np.all(xyz==0) and np.all(rpy==0) and np.all(scale==1)
        records.append(dict(format='URDF source visual',xyz=xyz,rpy=rpy,scale=scale))
    tree=ET.parse(xml);matched=[]
    for node in tree.findall('.//asset/mesh'):
        if Path(node.get('file','')).resolve()==mesh_path.resolve():
            scale=np.fromstring(node.get('scale','1 1 1'),sep=' ');assert np.all(scale==1);matched.append(node.get('name'))
    assert records and matched
    for geom in tree.findall('.//geom'):
        if geom.get('mesh') in matched:
            assert np.all(np.fromstring(geom.get('pos','0 0 0'),sep=' ')==0)
            assert np.allclose(np.fromstring(geom.get('quat','1 0 0 0'),sep=' '),[1,0,0,0])
            records.append(dict(format='MJCF source geom',attributes=geom.attrib))
    return records

def preflight(s,c):
    for key in ['canonical','mesh','pair']:
        assert sha(s[key])==s[key+'_sha'],(s['task'],key,'hash mismatch');c.add(s[key])
    manifest_path=s['stage']/'manifest.json';manifest=read(manifest_path);c.add(manifest_path)
    assert manifest['method']=='two-stage' and manifest['wrist_mode']=='position_only_baseline; canonical wrist quaternions preserved for later A1',manifest['wrist_mode']
    assert manifest['input_sha256']==s['canonical_sha'] and manifest['object_mesh_sha256']==s['mesh_sha']
    source=load_canonical_core4d_sequence(s['canonical']);seq=resample_core4d_pair_sequence(source,30)
    assert source.fps==15 and len(source.human_joints_full)==s['source_frames']
    assert seq.human_joints_full.shape==(s['frames'],2,127,3) and Path(source.object_mesh_path).resolve()==s['mesh'].resolve()
    nominal=[];alignment=[]
    for i in range(2):
        f=s['stage']/f'person{i+1}/person{i+1}_nominal_scaled.npz';c.add(f,parents=[str(manifest_path)])
        z=load(f);assert z['qpos'].shape==(s['frames'],43) and int(z['fps'])==30
        nominal.append(z['qpos']);inp=prepare_core4d_person_retarget_input(seq,i)
        e=error(z['human_joints'],inp.human_joints);assert e['max_abs']<1e-9,e
        eobj=error(z['qpos'][:,36:],_physical_object_qpos(inp.nominal_object_poses));assert eobj['max_abs']<1e-9,eobj
        alignment.append(dict(participant=i,source_human_alignment=e,nominal_object_alignment=eobj,ground_offset=inp.ground_offset_m))
    q=np.stack(nominal,axis=1);pair=load(s['pair']);n=s['frames'];physical=_physical_object_qpos(seq.object_poses)
    assert pair['robot_qpos'].shape==(n,2,36) and pair['object_qpos'].shape==(n,7) and int(pair['fps'])==30
    assert np.allclose(pair['human_to_robot_scales'],s['scales'],atol=1e-14,rtol=0)
    assert np.allclose(1.32/source.human_heights,s['scales'],atol=1e-14,rtol=0)
    for values in [q,pair['robot_qpos'],pair['object_qpos']]:
        assert np.isfinite(values).all() and np.allclose(np.linalg.norm(values[...,3:7],axis=-1),1,atol=1e-6,rtol=0)
    if s['task']=='Bucket':
        assert np.array_equal(pair['object_qpos'],physical)
        variants=[(s['variants'][0],q[:,:,:36],pair['object_qpos'],1.),(s['variants'][1],pair['robot_qpos'],pair['object_qpos'],1.)]
    else:
        assert np.array_equal(pair['robot_qpos'],q[:,:,:36])
        assert np.allclose(pair['object_qpos'][:,:3],q[:,:,36:39].mean(1),atol=1e-12,rtol=0)
        for i in range(2):
            angles=(Rotation.from_quat(pair['object_qpos'][:,[4,5,6,3]]).inv()*Rotation.from_quat(q[:,i][:,[40,41,42,39]])).magnitude()
            assert angles.max()<1e-10
        assert abs(float(pair['shared_object_scale'])-np.mean(s['scales']))<1e-14
        variants=[('one-sided',q[:,:,:36],q[:,0,36:],s['scales'][0]),('mean',q[:,:,:36],pair['object_qpos'],float(pair['shared_object_scale']))]
        for f in [s['pair'].parent/'manifest.json']:
            if f.exists():c.add(f)
    urdf=Path(manifest['object_urdf']);xml=Path(manifest['scene_xml']);c.add(urdf);c.add(xml)
    transforms=mesh_transforms(urdf,xml,s['mesh'])
    mesh=trimesh.load(s['mesh'],force='mesh',process=False)
    assert isinstance(mesh,trimesh.Trimesh) and np.isfinite(mesh.vertices).all() and len(mesh.faces)>0
    return dict(source=source,seq=seq,variants=variants,mesh=mesh,physical=physical,
         checks=dict(input_sha_matches=True,source_frames=s['source_frames'],reference_frames=n,fps=30,
         human_alignment=alignment,stages_verified=True,source_scale=s['scales'],mesh_transform_audit=transforms,
         geometry_processing='trimesh.load(process=False,force=mesh); no repair, welding, hull, sample or vertex-neighbor approximation',
         variants=[dict(name=v[0],scale=v[3],robot_sha256_array=hashlib.sha256(v[1].tobytes()).hexdigest(),object_sha256_array=hashlib.sha256(v[2].tobytes()).hexdigest()) for v in variants]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--instruction',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.time();c=Collector(out)
    assert [i for _,i in CORE4D_A1_PALM_LANDMARKS]==sum(PALM,[])
    assert sha(ROBOT)==ROBOT_SHA;c.add(ROBOT)
    instruction=c.add(a.instruction,'code');fk,fk_source=make_fk();fk_check=fk.validate()
    for f in [Path(__file__),ROOT/'scripts/prepare_experiments_r2_cpu.py',ROOT/'scripts/export_core4d_small_table_pair.py',
        ROOT/'scripts/retarget_core4d_pair.py',ROOT/'src/holosoma/holosoma/config_values/robot.py',ROOT/'src/holosoma/holosoma/config_values/marl/g1/command.py',
        ROOT/'src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_adapter.py',ROOT/'src/holosoma_retargeting/holosoma_retargeting/data_utils/core4d_retarget.py',
        ROOT/'src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py',ROOT/'src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py']:
        c.add(f,'code')
    write(out/'checks_preflight.json',dict(fk=fk_check,instruction_commit=COMMIT))
    contexts={};failures={};windowinfo={};checks={}
    # Freeze all source-only windows before evaluating any robot-to-object distances.
    for s in specs():
        try:
            x=preflight(s,c);contexts[s['task']]=x;checks[s['task']]=x['checks']
            n=s['frames'];source=x['source'];seq=x['seq'];centroid=np.stack([seq.human_joints_full[:,:,idx,:].mean(2) for idx in PALM],axis=2)
            points=local_points(centroid,x['physical']);d=exact_distance(x['mesh'],points).reshape(n,2,2);mask=d<.050
            times=np.arange(n)/30;indices=np.arange(n)/2;left=np.floor(indices).astype(int);right=np.minimum(left+1,s['source_frames']-1);alpha=indices-left
            assert np.allclose(seq.human_joints_full,source.human_joints_full[left]*(1-alpha[:,None,None,None])+source.human_joints_full[right]*alpha[:,None,None,None],atol=1e-12,rtol=0)
            f=out/(s['slug']+'_source_windows.npz')
            np.savez_compressed(f,source_distance_m=d,in_source_window=mask,source_centroid_world=centroid,
                source_centroid_object_local=points,source_object_qpos_wxyz=x['physical'],reference_frame=np.arange(n),time_s=times,
                canonical_left_row=left,canonical_right_row=right,right_weight=alpha,
                aligned_frame_ids_left=source.aligned_frame_ids[left],aligned_frame_ids_right=source.aligned_frame_ids[right],
                original_aligned_frame_ids=source.aligned_frame_ids)
            windowinfo[s['task']]=dict(artifact=f.name,sha256=sha(f),source_sha256=s['canonical_sha'],mesh_sha256=s['mesh_sha'],
                frames=n,fps=30,interpolation='linear positions; original adapter SLERP rotations; no clipping/phase shift',
                per_hand=[dict(participant=i,hand=h,frames=np.flatnonzero(mask[:,i,j]),times_s=times[mask[:,i,j]],n_window_frames=int(mask[:,i,j].sum())) for i in range(2) for j,h in enumerate(['left','right'])])
            print('Source windows frozen',s['task'],[int(mask[:,i,j].sum()) for i in range(2) for j in range(2)],flush=True)
        except Exception as e:
            failures[s['task']]=dict(error=str(e),traceback=traceback.format_exc());contexts.pop(s['task'],None)
            print('PRECHECK FAILED',s['task'],str(e),flush=True)
    windows=dict(definition='equal-weight four-point source centroid; unsigned triangle surface distance strictly <0.050m',
        landmark_names_indices=CORE4D_A1_PALM_LANDMARKS,threshold_m=.05,strict=True,source_scale=1,
        source_no_extra_coordinate_or_ground_transform=True,smoothing_dilation_short_segment_filter=False,
        tasks=windowinfo,failures=failures)
    write(out/'windows.json',windows);windows_sha=sha(out/'windows.json');rows=[];per=[]
    for s in specs():
        if s['task'] not in contexts:
            for v in s['variants']:
                for i in range(2):
                    for h in ['left','right']:rows.append(dict(zip(FIELDS,[s['task'],v,i,h]+['--']*5)))
            continue
        x=contexts[s['task']];z=load(out/windowinfo[s['task']]['artifact']);mask=z['in_source_window'];n=s['frames'];mesh=x['mesh']
        checks[s['task']]['measurements']=[]
        for v,q,obj,scale in x['variants']:
            poses=fk.poses(q[...,:3],q[...,[4,5,6,3]],q[...,7:],HANDS);hands=poses[...,:3,3]
            local=local_points(hands,obj,scale);dist=(scale*exact_distance(mesh,local)).reshape(n,2,2)
            # Check FK on actual references independently of random unit cases.
            fkerr=0.
            for k in [0,n//2,n-1]:
                for i in range(2):
                    # Trimesh scene.add_edge skips updates within 1e-8; a fresh
                    # checker avoids previous-frame cached transforms. The
                    # measured BatchFK path does not use this scene cache.
                    checker=yourdfpy.URDF.load(str(ROBOT),load_meshes=False,load_collision_meshes=False)
                    checker.update_cfg(q[k,i,7:]);base=np.eye(4);base[:3,:3]=Rotation.from_quat(q[k,i,[4,5,6,3]]).as_matrix();base[:3,3]=q[k,i,:3]
                    for j,h in enumerate(HANDS):fkerr=max(fkerr,float(abs((base@checker.get_transform(h))[:3,3]-hands[k,i,j]).max()))
            assert fkerr<1e-10
            # Exact all-triangle/world checks at first/middle/last frames, all four hands.
            worlderr=0.;bruteerr=0.
            for k in [0,n//2,n-1]:
                r=Rotation.from_quat(obj[k,[4,5,6,3]]).as_matrix()
                world=trimesh.Trimesh(vertices=mesh.vertices@r.T*scale+obj[k,:3],faces=mesh.faces,process=False)
                worlderr=max(worlderr,float(abs(brute_distance(world,hands[k].reshape(-1,3))-dist[k].reshape(-1)).max()))
                bruteerr=max(bruteerr,float(abs(scale*brute_distance(mesh,local[k].reshape(-1,3))-dist[k].reshape(-1)).max()))
            assert max(worlderr,bruteerr)<1e-9
            inside,insideinfo=inside_flags(mesh,local.reshape(-1,3),dist.reshape(-1));inside=inside.reshape(n,2,2)
            f=out/(s['slug']+'_'+v.replace('-','_')+'_measurements.npz')
            np.savez_compressed(f,robot_qpos_wxyz=q,object_qpos_wxyz=obj,object_scale=scale,hand_link_origins_world=hands,
                hand_link_origins_object_local_unscaled=local,dist_m=dist,inside=inside,in_source_window=mask,
                source_dist_m=z['source_distance_m'],hand_names=HANDS,frame=np.arange(n),time_s=np.arange(n)/30)
            for i in range(2):
                for j,h in enumerate(['left','right']):
                    rows.append(stat_row(s['task'],v,i,h,dist[:,i,j],inside[:,i,j],mask[:,i,j]))
                    for k in range(n):per.append(dict(task=s['task'],variant=v,participant=i,hand=h,frame=k,time_s=k/30,
                        in_source_window=bool(mask[k,i,j]),source_dist_mm=float(1000*z['source_distance_m'][k,i,j]),
                        dist_mm=float(1000*dist[k,i,j]),inside='--' if inside[k,i,j]<0 else bool(inside[k,i,j])))
            checks[s['task']]['measurements'].append(dict(variant=v,array_file=f.name,array_sha256=sha(f),actual_frame_FK_max_m=fkerr,
                FK_checker='Fresh yourdfpy URDF per actual pose, avoiding scene add_edge 1e-8 reuse threshold; original BatchFK calculation unchanged',
                exact_local_world_max_error_m=worlderr,BVH_vs_all_triangles_max_error_m=bruteerr,inside=insideinfo,
                mesh_vertices=len(mesh.vertices),mesh_triangles=len(mesh.faces),scaled_volume_m3=float(mesh.volume*scale**3)))
            print('Measured',s['task'],v,flush=True)
    assert sha(out/'windows.json')==windows_sha
    csv_write(out/'ref_shared_scene_handdist.csv',rows,FIELDS);csv_write(out/'per_frame_handdist.csv',per,PER_FIELDS)
    aggregates=[]
    for s in specs():
        for v in s['variants']:
            groups=[r for r in rows if r['task']==s['task'] and r['variant']==v];assert len(groups)==4
            values=[r['dist_mean_mm'] for r in groups]
            aggregates.append(dict(task=s['task'],variant=v,method='equal_weight_participant_hand_means',
                task_variant_mean_mm=float(np.mean(values)) if all(isinstance(x,float) for x in values) else '--',components=groups))
    write(out/'aggregation.json',aggregates)
    assert len(rows)==24 and len({tuple(r[k] for k in FIELDS[:4]) for r in rows})==24
    manifest=dict(instruction_commit=COMMIT,instruction=instruction,checkpoint='not_used',command=[sys.executable,*sys.argv],
       code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
       python=sys.version,executable=sys.executable,platform=platform.platform(),
       packages={k:md.version(k) for k in ['numpy','scipy','trimesh','yourdfpy','rtree','embreex']},fk_source=fk_source,
       inputs_and_code=c.records,elapsed_seconds=time.time()-start,strict_cpu=True,
       configuration=dict(threshold_m=.050,boundary_world_m=1e-8,reference_fps=30,main_fields=FIELDS,hand_link_origins=HANDS),
       original_assets_modified=False,training_or_optimization=False,failures=failures)
    write(out/'sources/manifest.json',manifest)
    (out/'sources/worktree_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    files=[r['original_path'] for r in c.records if '/scripts/' in r['original_path'] or '/src/' in r['original_path']]
    (out/'sources/dirty_diff.patch').write_text(subprocess.check_output(['git','diff','HEAD','--',*files],cwd=ROOT,text=True))
    write(out/'checks.json',dict(status='complete' if not failures else 'partial',failures=failures,FK_random_validation=fk_check,tasks=checks,
       source_windows_sha256=windows_sha,exact_rows=24,per_frame_rows=len(per),no_checkpoint=True,
       distance_inside_not_clamped=True,independent_summary_recheck='to be recorded by separate verification script'))
    return 0 if not failures else 2
if __name__=='__main__':raise SystemExit(main())
