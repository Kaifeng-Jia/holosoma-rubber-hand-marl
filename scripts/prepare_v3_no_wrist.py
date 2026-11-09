"""Generate explicitly versioned no-wrist assets; no retargeting or physics step."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/holosoma'),str(ROOT/'src/holosoma_retargeting')]
from holosoma_retargeting.core4d_pair_runtime_reference import build_core4d_pair_runtime_reference_file
from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
from holosoma_retargeting.interaction_mesh_geometry import landmarks_world

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def main():
    a=argparse.ArgumentParser();a.add_argument('--task',choices=['chair','bucket'],required=True);args=a.parse_args()
    task=args.task;name='chair021' if task=='chair' else 'bucket003';base=get_core4d_pair_experiment(name)
    pkg=ROOT/'src/holosoma';full_dir=(pkg/base.runtime_reference_file).parent
    before=ROOT/'logs/Core4DPreviews'/('chair021_20231020_074_baseline_20260908/shared_small_preview' if task=='chair' else 'bucket003_20231020_071_20260918/two_stage')/'core4d_pair_reference.npz'
    expected='d8e945b4ec08cc9d22d6493942cf05ed787cd42b87b5e18ac4e6d7d6423dd586' if task=='chair' else '089d3a36a39442e055f01a8111d6145475c062928c28727062f951887020ceec'
    assert sha(before)==expected
    full_compact=full_dir/'core4d_pair_compact_fps30.npz'
    assert sha(full_compact)==base.source_pair_sha256
    pre=load(before);full=load(full_compact)
    wrists=[26,27,28,33,34,35];mask=np.ones(36,bool);mask[wrists]=False
    assert np.array_equal(pre['robot_qpos'][:,:,mask],full['robot_qpos'][:,:,mask])
    assert np.array_equal(pre['object_qpos'],full['object_qpos']) and np.array_equal(pre['fps'],full['fps'])
    dest=full_dir.parent/'v3_20261102'/f'{task}_no_wrist_v2';dest.mkdir(parents=True,exist_ok=False)
    compact=dest/'core4d_pair_compact_fps30.npz';shutil.copyfile(before,compact)
    dump(dest/'source_manifest.json',{'pair_sha256':sha(compact),'variant':'no-wrist','parent_experiment':name,'source_path':str(before.relative_to(ROOT)),'source_sha256':expected,'full_source_sha256':base.source_pair_sha256,'authority_commit':'6d3836a8328b32c901b8140d3e0f41aff4ef01d6'})
    runtime_file=dest/'core4d_pair_runtime_fps50.npz'
    runtime=build_core4d_pair_runtime_reference_file(compact,runtime_file,expected_source_sha256=expected,source_manifest_path=dest/'source_manifest.json')
    new=load(runtime_file);old=load(pkg/base.runtime_reference_file)
    assert np.array_equal(new['joint_names'],old['joint_names']) and np.array_equal(new['body_names'],old['body_names'])
    joint_mask=np.array(['wrist' not in str(n) for n in new['joint_names']]);assert (~joint_mask).sum()==6
    body_mask=np.array([not any(s in str(n) for s in ['wrist','rubber_hand','thumb','pinky']) for n in new['body_names']])
    # mj_objectVelocity uses a tree COM frame, whose rounding can change when
    # distal wrists move, even for mathematically unchanged ancestor velocities.
    # Keep exact Full values on unchanged channels after an independent numerical
    # check. Changed wrist descendants always use fresh FK/velocity, never caches.
    rounding={}
    for k in ('agent_body_lin_vel_w','agent_body_ang_vel_w'):
        x,y=new[k][:,:,body_mask],old[k][:,:,body_mask]
        err=float(np.max(np.abs(x-y)));bound=64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(y))))
        assert err<=bound,(k,err,bound)
        rounding[k]={'recomputed_max_abs_difference':err,'machine_roundoff_bound':bound,'preserved_full_unchanged_channels':True}
        new[k][:,:,body_mask]=y
    np.savez_compressed(runtime_file,**new)
    diff={}
    for k in new:
        if k=='provenance':continue
        x,y=new[k],old[k];assert x.shape==y.shape
        if k.startswith('agent_joint'):x,y=x[:,:,joint_mask],y[:,:,joint_mask]
        elif k.startswith('agent_body'):x,y=x[:,:,body_mask],y[:,:,body_mask]
        equal=np.array_equal(x,y);err=float(np.max(np.abs(x-y))) if x.dtype.kind in 'fiu' else None
        diff[k]={'unchanged_channels_bitwise_equal':equal,'max_abs_difference':err}
        assert equal,(task,k,err)
    # Byte-identical original object and dependent mesh; no mass/inertia change.
    promotion=json.loads((pkg/base.training_promotion_file).read_text())
    for p in [pkg/base.object_urdf_file,full_dir/promotion['object_mesh_file']]:shutil.copyfile(p,dest/p.name)
    newurdf=dest/Path(base.object_urdf_file).name
    promotion={k:v for k,v in promotion.items() if not k.startswith('physics_loading')}
    promotion.update(source_pair_sha256=sha(compact),source_manifest_sha256=sha(dest/'source_manifest.json'),runtime_reference_sha256=sha(runtime_file),training_object_urdf_sha256=sha(newurdf),training_ready=True,formal_training_authorized=True,physics_loading_check='pending_variant_smoke; unchanged object geometry and physical parameters',preparation_scope='explicit V3 variant authorized for bounded smoke; formal queue requires smoke evidence',variant='no-wrist',parent_runtime_sha256=base.runtime_reference_sha256)
    if task=='bucket':
        artifact=load(full_dir/'interaction_vectors_v1.npz');meta=json.loads(str(artifact['metadata_json'].item()))
        specs=[{'body':str(n),'local_xyz':off.tolist()} for n,off in zip(artifact['point_body_names'],artifact['point_offsets'])]
        points=landmarks_world(runtime.agent_body_pos_w,runtime.agent_body_quat_wxyz,runtime.body_names,specs)
        # Same mapped landmarks are compared on Full to validate this recomputation path.
        check=landmarks_world(old['agent_body_pos_w'],old['agent_body_quat_w'],old['body_names'],specs)
        mapping_error=float(np.max(np.abs(check-artifact['reference_body_points_world'])))
        assert mapping_error<1e-6,mapping_error
        artifact['reference_body_points_world']=points.astype(np.float32)
        meta.update(source_pair_sha256=sha(compact),runtime_reference_sha256=sha(runtime_file),v3_variant='no-wrist',v3_full_landmark_equivalence_max_m=mapping_error)
        artifact['metadata_json']=np.asarray(json.dumps(meta,sort_keys=True))
        np.savez_compressed(dest/'interaction_vectors_v1.npz',**artifact)
        promotion['interaction_artifact_sha256']=sha(dest/'interaction_vectors_v1.npz')
        promotion['allowed_reward_variant']='bucket_A_no_rel_no_height'
    dump(dest/'training_asset_manifest.json',promotion)
    dump(dest/'diff_to_full.json',{'passed':True,'scope':'only twelve wrist channels and their FK/velocity/landmark derivatives','changed_qpos_columns_per_robot':wrists,'unchanged_runtime':diff,'roundoff_audit':rounding,'mass_material_inertia_urdf_unchanged':sha(newurdf)==base.object_urdf_sha256,'full_contract':base.checkpoint_contract})
    descriptor={'schema':'v3_reference_variant_v1','variant':'no-wrist','parent_experiment':name,'parent_contract':base.checkpoint_contract,'source_pair_file':str(compact.relative_to(pkg)),'source_pair_sha256':sha(compact),'runtime_reference_file':str(runtime_file.relative_to(pkg)),'runtime_reference_sha256':sha(runtime_file),'object_urdf_file':str(newurdf.relative_to(pkg)),'object_urdf_sha256':sha(newurdf),'training_promotion_file':str((dest/'training_asset_manifest.json').relative_to(pkg)),'training_promotion_sha256':sha(dest/'training_asset_manifest.json'),'diff_file':str((dest/'diff_to_full.json').relative_to(pkg)),'diff_sha256':sha(dest/'diff_to_full.json'),'allowed_reward_variant':'baseline' if task=='chair' else 'bucket_A_no_rel_no_height'}
    dump(dest/'variant.json',descriptor)
    print(json.dumps({'task':task,'variant_manifest':str(dest/'variant.json'),'frames':len(new['agent_joint_pos']),'unchanged_channels_exact':True,'status':'offline_ready_smoke_pending'}))
if __name__=='__main__':main()
