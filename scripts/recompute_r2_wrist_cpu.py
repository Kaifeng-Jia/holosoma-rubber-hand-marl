#!/usr/bin/env python3
"""Recompute full palm-frame SO(3) errors; no wrist solver, physics or renderer."""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
import yourdfpy
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/holosoma_retargeting'))
from holosoma_retargeting.data_utils.core4d_adapter import load_canonical_core4d_sequence,resample_core4d_pair_sequence
from holosoma_retargeting.data_utils.core4d_retarget import build_core4d_a1_palm_orientation_targets


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ident(p):return dict(path=str(p),sha256=digest(p))
def read(p):return json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output
    code=ROOT/'src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py'
    tree=ast.parse(code.read_text());normals=None
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='palm_normals' for t in node.targets):
            assert isinstance(node.value,ast.Dict)
            normals={ast.literal_eval(k):np.array(ast.literal_eval(v.args[0]),dtype=float) for k,v in zip(node.value.keys,node.value.values)}
    assert normals is not None
    bases=[]
    for side in ['left','right']:
        normal=normals[side];normal/=np.linalg.norm(normal)
        finger=np.array([1.,0.,0.]);finger-=np.dot(finger,normal)*normal;finger/=np.linalg.norm(finger)
        across=np.cross(normal,finger);across/=np.linalg.norm(across)
        bases.append(np.column_stack([finger,across,normal]))
    robot=ROOT/'src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf'
    urdf=yourdfpy.URDF.load(str(robot),load_meshes=False,load_collision_meshes=False)
    audit=read(ROOT/'docs/experiments/retarget_method_comparison_20260925.json')
    cases=[('Chair',ROOT/audit['chair']['baseline'],ROOT/audit['chair']['refined']),
        ('Bucket',ROOT/'logs/Core4DPreviews/bucket003_20231020_071_20260918/two_stage/core4d_pair_reference.npz',
         ROOT/'logs/Core4DPreviews/bucket003_20231020_071_20260918/a1_wrist_only/core4d_pair_reference.npz')]
    rows=[];records=[]
    for task,before,after in cases:
        m=read(after.parent/'manifest.json');canonical=Path(m['input'])
        assert digest(canonical)==m['input_sha256']
        seq=load_canonical_core4d_sequence(canonical)
        with np.load(before) as z:qb=z['robot_qpos'];fps=int(z['fps']);ob=z['object_qpos']
        with np.load(after) as z:qa=z['robot_qpos'];oa=z['object_qpos']
        assert np.array_equal(ob,oa)
        nonwrist=[i for i in range(36) if i not in [26,27,28,33,34,35]]
        assert np.array_equal(qb[:,:,nonwrist],qa[:,:,nonwrist])
        if seq.fps!=fps:seq=resample_core4d_pair_sequence(seq,fps)
        assert len(seq.human_joints_full)==len(qb)
        scene=Path(m['scene_xml']); model=mujoco.MjModel.from_xml_path(str(scene)); data=mujoco.MjData(model)
        bodyids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,s+'_rubber_hand_link') for s in ['left','right']]
        assert all(x>=0 for x in bodyids) and model.nq==43
        arrays={};calibration=[];urdf_max_difference=0.
        for role in range(2):
            targets,cal=build_core4d_a1_palm_orientation_targets(seq.human_joints_full[:,role],seq.wrist_quat_xyzw[:,role])
            calibration.append(cal.tolist());arrays[f'role{role}_target_palm_world']=targets
            curves=[]
            for variant,q in [('before',qb[:,role]),('after',qa[:,role])]:
                result=np.empty((len(q),2));actual=np.empty((len(q),2,3,3))
                for t,pose in enumerate(q):
                    urdf.update_cfg(pose[7:]);root=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                    for h,side in enumerate(['left','right']):
                        actual[t,h]=root@urdf.get_transform(side+'_rubber_hand_link')[:3,:3]@bases[h]
                urdf_actual=actual.copy()
                # Match the original retarget MJCF; kinematics only, no mj_step.
                # URDF/MJCF fixed rotations have tiny independent decimal rounding.
                for t,pose in enumerate(q):
                    data.qpos[:]=np.r_[pose,ob[t]];mujoco.mj_kinematics(model,data)
                    for h,bid in enumerate(bodyids): actual[t,h]=data.xmat[bid].reshape(3,3)@bases[h]
                cross=np.degrees(Rotation.from_matrix((actual@urdf_actual.swapaxes(-1,-2)).reshape(-1,3,3)).magnitude())
                urdf_max_difference=max(urdf_max_difference,float(cross.max()))
                relative=targets@actual.swapaxes(-1,-2)
                # acos(trace) as specified, with stable geodesic cross-check near zero.
                result=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
                stable=np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude()).reshape(len(q),2)
                assert np.max(abs(result-stable))<3e-6
                arrays[f'role{role}_{variant}_error_deg']=result
                arrays[f'role{role}_{variant}_stable_error_deg']=stable
                curves.append(result)
            for h,side in enumerate(['left','right']):
                rows.append(dict(task=task,role=role,hand=side,target_source='source_SMPLX_landmarks_calibrated_wrist_frames',n_frames=len(qb),
                    before_mean_deg=float(curves[0][:,h].mean()),after_mean_deg=float(curves[1][:,h].mean()),after_max_deg=float(curves[1][:,h].max())))
        np.savez_compressed(out/f'checks/{task.lower()}_wrist_so3_arrays.npz',**arrays)
        records.append(dict(task=task,before=ident(before),after=ident(after),canonical=ident(canonical),manifest=ident(after.parent/'manifest.json'),
            fps=fps,frames=len(qb),calibration_p90_deg=calibration,root_object_nonwrist_bit_exact=True,
            original_retarget_scene=ident(scene),URDF_vs_original_MJCF_max_rotation_deg=urdf_max_difference))
    chair=[r for r in rows if r['task']=='Chair']
    for r in chair:
        archived=next(h for h in audit['chair']['hands'] if int(h['person'][-1])-1==r['role'] and h['side']==r['hand'])
        assert abs(r['before_mean_deg']-archived['baseline_orientation_deg']['mean'])<3e-6
        assert abs(r['after_mean_deg']-archived['refined_orientation_deg']['mean'])<3e-6
        assert abs(r['after_max_deg']-archived['refined_orientation_deg']['max'])<3e-6
    with (out/'wrist_orientation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(script=ident(__file__),python=sys.executable,command=sys.argv,source_axes=ident(code),robot=ident(robot),
        normal_vectors=normals,robot_palm_bases=bases,records=records,window='All original retarget frames, both hands separately',
        formula='degrees(acos(clip((trace(R_target @ R_actual.T)-1)/2,-1,1)))',
        source_axes_definition='forward wrist->middle1; across index1->pinky1 left positive, right negative; normal cross(forward,across)',
        independence='Full rotations from original retarget MJCF via mj_kinematics; separate real URDF cross-check; no mj_step, solver, renderer or policy run',
        archived_chair_match_tolerance_deg=3e-6,return_code=0,
        pending_tasks='OMOMO Push/Pull/Kick and smalltable target-frame applicability not yet audited; not silently filled')
    (out/'checks/wrist_recomputed.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist())+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
