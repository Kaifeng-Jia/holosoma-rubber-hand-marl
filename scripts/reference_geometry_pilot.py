#!/usr/bin/env python3
"""Independent offline Push/Pull reference geometry pilot (2026-10-17)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import yourdfpy
from reference_geometry_math import BoxUnion,collision_meshes,surface_samples,continuous_windows

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/holosoma_retargeting'))
from holosoma_retargeting.dual_a1_layout import load_a1_motion,quaternion_wxyz_to_matrix,g1_sagittal_joint_map
from holosoma_retargeting.dual_pull_reference import (
    G1_29DOF_JOINT_NAMES, _symmetrize_object_qpos, _transport_source_robot_to_shared_object)

DATA=ROOT/'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking'
ROBOT=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
OLD=Path('/home/kevin/holosoma-rubber-hand-largetable')
MARL=Path('/home/kevin/holosoma-rubber-hand-marl')
SOURCES={
 'push':(DATA/'rubber_hand_largetable_v1/a1/sub6_largetable_033_a1_mj_fps50_w_obj.npz',
         'b669468ade12f8c271119d6c8c120463223ab65489805c486e43b0c3a4ba2207'),
 'pull':(OLD/'logs/WholeBodyTracking/20260807_212122-rubberhand_pull_sub3_010_8000_seed42-eval/pull_7999_seed42_attempt08_success_viser_qpos.npz',
         'd5c0705cfc3e643e75c9a9b1b28a21473b8b042134bd37ae935c7192c179bcdc')}
THRESHOLDS=dict(unsigned_near_m=.03,near_sample_fraction=.05,min_duration_s=.1,fill_gap_frames=1,
  fitting_samples_per_hand=256,independent_validation_samples_per_hand=1024,seed_fitting=1701,seed_validation=2701,
  nearest_face_tie_tolerance_m=1e-7,
  association_rule='Retain each face with >=5% of all hand samples within3cm for >=0.1s; one-frame gaps filled. Report competing nearest-face ties; do not treat proximity as contact.',
  source_only_freeze=True)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,Path):return str(v)
    raise TypeError(type(v))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,default=default,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def identity(path):
    p=Path(path);return dict(path=str(p),exists=p.exists(),sha256=sha(p) if p.exists() else None)
def artifact(path):
    result=identity(path)
    if path.suffix=='.npz' and path.exists():
        with np.load(path,allow_pickle=False) as z:result['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
    return result
def transform(q):
    a=np.eye(4);a[:3,:3]=quaternion_wxyz_to_matrix(q[3:7]);a[:3,3]=q[:3];return a


class RobotFK:
    def __init__(self):
        self.u=yourdfpy.URDF.load(str(ROBOT),load_meshes=False,load_collision_meshes=False)
        assert self.u.actuated_joint_names==G1_29DOF_JOINT_NAMES.tolist()
        self.meshes=collision_meshes(ROBOT)
        self.hands=[s+'_rubber_hand_link' for s in ('left','right')]
        self.fit={n:surface_samples(self.meshes[n],256,1701+i) for i,n in enumerate(self.hands)}
        self.dense={n:surface_samples(self.meshes[n],1024,2701+i) for i,n in enumerate(self.hands)}
    def poses(self,qpos,names):
        result={n:np.empty((len(qpos),4,4)) for n in names}
        for t,q in enumerate(qpos):
            self.u.update_cfg(q[7:]);base=transform(q)
            for n in names:result[n][t]=base@self.u.get_transform(n)
        return result


def draw_regions(path,geometry,points=None,faces=None,title=''):
    fig=plt.figure(figsize=(9,8));ax=fig.add_subplot(111,projection='3d')
    for i,f in enumerate(geometry.faces):
        color='tab:orange' if faces and i in faces else 'lightgray'
        for patch in f['patches']:
            other=[j for j in range(3) if j!=f['axis']];v=[]
            for a,b in [(0,0),(1,0),(1,1),(0,1)]:
                p=patch['lo'].copy();p[other]=[patch['hi' if a else 'lo'][other[0]],patch['hi' if b else 'lo'][other[1]]];v.append(p[[0,2,1]])
            ax.add_collection3d(Poly3DCollection([v],facecolor=color,edgecolor='gray',alpha=.23,linewidth=.3))
    for part in geometry.parts:
        c=part['center'];ax.text(c[0],c[2],c[1],part['id'].replace('largetable_','').replace('widetable_pull_',''),fontsize=7)
    if points:
        for label,p in points.items():ax.scatter(p[:,0],p[:,2],p[:,1],s=2,label=label)
        ax.legend()
    allv=np.vstack([p['lo'] for p in geometry.parts]+[p['hi'] for p in geometry.parts]);center=(allv.min(0)+allv.max(0))/2;rad=max(np.ptp(allv,axis=0))*.65
    ax.set_xlim(center[0]-rad,center[0]+rad);ax.set_ylim(center[2]-rad,center[2]+rad);ax.set_zlim(center[1]-rad,center[1]+rad)
    ax.set_xlabel('object X (m)');ax.set_ylabel('object Z (m)');ax.set_zlabel('object Y vertical (m)');ax.set_box_aspect([1,1,1]);ax.set_title(title)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


def analyze(case,out,fk):
    out.mkdir(parents=True,exist_ok=False)
    source,expected=SOURCES[case];assert sha(source)==expected
    if case=='push':
        q,p,quat,fps=load_a1_motion(source);obj=np.c_[p,quat];raw_obj=obj.copy();raw_q=q.copy()
        target_path=DATA/'objects_widetable_plan5_training.urdf'
        config=MARL/'logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/run_config.json'
        source_cfg=OLD/'logs/WholeBodyTracking/20260801_184233-rubberhand_a1_contact_semantic_8000_seed42-locomotion/holosoma_config.yaml'
        old_pair=None
    else:
        with np.load(source,allow_pickle=False) as z:allq=z['qpos'].astype(float);fps=int(z['fps'])
        raw_q,raw_obj=allq[:,:36],allq[:,36:];obj=_symmetrize_object_qpos(raw_obj)
        q=_transport_source_robot_to_shared_object(raw_q,raw_obj,obj)
        target_path=DATA/'objects_widetable_plan5_pull_training.urdf'
        config=MARL/'logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/run_config.json'
        source_cfg=source.parent/'holosoma_config.yaml'
        old_pair=MARL/'logs/Plan5Pull/reference_preview/plan5_attempt08_mirrored_pair_qpos.npz'
    conf=json.loads(config.read_text());assert Path(conf['object_urdf']).name==target_path.name
    text=source_cfg.read_text();assert 'object_urdf_path: holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largetable.urdf' in text
    assert 'urdf_file: g1/main_mesh_collision_rubberhand.urdf' in text and 'collapse_fixed_joints: true' in text
    historical_robot=OLD/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
    source_geom=BoxUnion(DATA/'objects_largetable.urdf');target_geom=BoxUnion(target_path)
    assert sha(historical_robot)==sha(ROBOT)
    deps=[]
    for urdf in [ROBOT,source_geom.path,target_path]:
        for node in ET.parse(urdf).findall('.//mesh'):
            p=Path(node.get('filename'));p=p if p.is_absolute() else urdf.parent/p
            deps.append(dict(**identity(p.resolve()),declaring_urdf=str(urdf),scale=node.get('scale','1 1 1')))
    manifest=dict(case=case,source=artifact(source),fps=fps,frames=len(q),source_config=identity(source_cfg),formal_config=identity(config),
      source_object=identity(source_geom.path),target_object=identity(target_path),source_robot=identity(historical_robot),actual_robot=identity(ROBOT),dependencies=deps,
      joint_names=G1_29DOF_JOINT_NAMES,quaternion_order='wxyz',world_axes='Z-up; object localY vertical',
      fixed_links=dict(collapse_fixed_joints=True,evidence=str(source_cfg),training_code='simulator/isaacsim/isaacsim.py UrdfFileCfg merge_fixed_joints',fk_method='Original unmerged URDF chain, including hand fixed joints; mathematically equivalent parent-to-hand transform'),
      wrists='Exactly original source coordinates; no new wrist optimization',
      preprocessing='Existing _symmetrize_object_qpos + _transport_source_robot_to_shared_object; no mirror or outward shift yet' if case=='pull' else 'None',
      preprocessing_source=identity(ROOT/'src/holosoma_retargeting/holosoma_retargeting/dual_pull_reference.py'),
      old_pair_reference=artifact(old_pair) if old_pair else dict(kind='Online source shift; no independent pair NPZ'),
      runtime_reference=artifact(ROOT/'src/holosoma'/conf['paired_reference_file']) if conf['paired_reference_file'] else artifact(source),
      old_parameters_loaded_for_generation=False)
    write(out/'input_manifest.json',manifest);write(out/'source_geometry.json',source_geom.manifest());write(out/'target_geometry.json',target_geom.manifest())
    write(out/'source_thresholds.json',THRESHOLDS)
    np.savez_compressed(out/'preprocessed_source.npz',robot_qpos=q,object_qpos=obj,raw_robot_qpos=raw_q,raw_object_qpos=raw_obj,fps=fps,joint_names=G1_29DOF_JOINT_NAMES)
    np.savez_compressed(out/'hand_samples.npz',**{n+'_fit':v for n,v in fk.fit.items()},**{n+'_validation':v for n,v in fk.dense.items()})
    transforms=fk.poses(q,fk.hands);rot=quaternion_wxyz_to_matrix(obj[:,3:]);curves={};records=[];cloud={}
    local_points=[]
    for hand in fk.hands:
        tf=transforms[hand];world=np.einsum('tij,pj->tpi',tf[:,:3,:3],fk.fit[hand])+tf[:,:3,3][:,None]
        local=np.einsum('tji,tpj->tpi',rot,world-obj[:,:3][:,None]);local_points.append(local)
        unsigned,signed,face,near,ties=source_geom.distances(local,with_ties=True)
        frac=np.mean(unsigned<=.03,axis=1);windows=continuous_windows(frac>=.05,int(np.ceil(.1*fps)))
        face_records=[]
        for idx,f in enumerate(source_geom.faces):
            fraction=np.mean((unsigned<=.03)&((ties & (np.uint64(1)<<np.uint64(idx)))!=0),axis=1)
            for a,b in continuous_windows(fraction>=.05,int(np.ceil(.1*fps))):
                face_records.append(dict(face_index=idx,face_id=f['id'],part_id=f['part_id'],start=a,end_exclusive=b,
                     duration_s=(b-a)/fps,mean_sample_fraction=float(fraction[a:b].mean()),anchors=f['corners']))
        records.append(dict(hand=hand,source_windows=windows,stable_face_windows=face_records,
          nearest_face_tie_fraction=float(np.mean((ties & (ties-np.uint64(1)))!=0)),
          min_unsigned_distance_m=float(unsigned.min()),max_penetration_m=float(max(0,-signed.min())),max_near_fraction=float(frac.max()),
          closest_faces_all_sequence=[dict(id=f['id'],fraction=float(np.mean(face==i))) for i,f in enumerate(source_geom.faces) if np.any(face==i)]))
        curves[hand+'_min_distance_m']=unsigned.min(1);curves[hand+'_near_fraction']=frac;curves[hand+'_max_penetration_m']=np.maximum(0,-signed.min(1))
        curves[hand+'_nearest_face_index']=face;curves[hand+'_signed_distance_m']=signed
        curves[hand+'_competing_face_bits']=ties
        chosen=np.concatenate([np.arange(a,b) for a,b in windows]) if windows else np.arange(len(q))
        cloud[hand]=local[chosen][::max(1,len(chosen)//10)].reshape(-1,3)
    np.savez_compressed(out/'source_distances.npz',**curves,hand_points_object_frame=np.stack(local_points,axis=1),fps=fps)
    write(out/'source_interactions.json',dict(records=records,labels_are='Candidate geometric interaction, not contact force truth',thresholds=THRESHOLDS))
    fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for hand in fk.hands:
        for ax,key in zip(axs,['min_distance_m','near_fraction','max_penetration_m']):ax.plot(np.arange(len(q))/fps,curves[hand+'_'+key],label=hand)
    for ax,title in zip(axs,['Unsigned distance to external union boundary (m)','Fraction of hand samples within3cm','Maximum sampled penetration (m)']):ax.set_ylabel(title);ax.grid(alpha=.25)
    axs[0].axhline(.03,color='gray',linestyle='--');axs[1].axhline(.05,color='gray',linestyle='--');axs[0].legend();axs[-1].set_xlabel('Source time (s)')
    fig.tight_layout();fig.savefig(out/'source_distance_curves.png',dpi=160);plt.close(fig)
    associated=sorted({x['face_index'] for record in records for x in record['stable_face_windows']})
    draw_regions(out/'source_regions.png',source_geom,cloud,associated,case+' source surface proximity')
    draw_regions(out/'target_regions.png',target_geom,title=case+' actual target parts; roles not assigned')
    print(case,json.dumps(records,default=default),flush=True)
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(args.output/'worktree_identity.json',dict(head=head,branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),instruction_commit='b84464ba212c7dc26bafe4771580caf7a4f3455c',
        instruction_path='docs/paper_handoff/ubuntu_reference_geometry_pilot_20261017.md',no_training=True,no_dynamic_simulation=True))
    doc=subprocess.check_output(['git','show','b84464ba212c7dc26bafe4771580caf7a4f3455c:docs/paper_handoff/ubuntu_reference_geometry_pilot_20261017.md'],cwd=ROOT)
    (args.output/'REQUEST_FROM_MAIN.md').write_bytes(doc)
    fk=RobotFK()
    for case in ('push','pull'):
        analyze(case,args.output/case,fk)

if __name__=='__main__':main()
