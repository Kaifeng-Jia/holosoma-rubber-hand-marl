"""U3a: fixed C-01 robot trajectories, participant-1 object scale/translation."""
import hashlib,json,shutil,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'src/holosoma'),str(R/'src/holosoma_retargeting'),str(R/'scripts')]
from export_core4d_small_table_pair import _load_nominal_qpos,_write_scaled_preview_urdf
from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
from holosoma_retargeting.core4d_pair_runtime_reference import build_core4d_pair_runtime_reference_file
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    e=get_core4d_pair_experiment('chair021');pkg=R/'src/holosoma';base=(pkg/e.runtime_reference_file).parent
    compact=base/'core4d_pair_compact_fps30.npz';assert sha(compact)==e.source_pair_sha256
    pair=load(compact);full=load(pkg/e.runtime_reference_file)
    source=R/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/two_stage'
    p1=source/'person1/person1_nominal_scaled.npz';p2=source/'person2/person2_nominal_scaled.npz'
    q1,f1=_load_nominal_qpos(p1);q2,f2=_load_nominal_qpos(p2)
    assert f1==f2==int(pair['fps']) and q1.shape==q2.shape and len(q1)==len(pair['robot_qpos'])
    scales=pair['human_to_robot_scales'];s1,s2=map(float,scales);mean=float(pair['shared_object_scale'])
    assert np.array_equal(scales,load(source/'core4d_pair_reference.npz')['human_to_robot_scales'])
    assert abs(s1-.7616803111061523)<1e-14 and abs(s2-.7267058142765197)<1e-14 and mean==(s1+s2)/2
    assert np.allclose(pair['object_qpos'][:,:3],.5*(q1[:,36:39]+q2[:,36:39]),rtol=0,atol=1e-14)
    assert np.allclose(pair['object_qpos'][:,3:],q1[:,39:43],rtol=0,atol=1e-14)
    dest=base.parent/'v3_20261102/chair_one_sided';dest.mkdir(parents=True,exist_ok=False)
    original_robot=pair['robot_qpos'].copy();original_rotation=pair['object_qpos'][:,3:].copy()
    pair['object_qpos'][:,:3]=q1[:,36:39];pair['shared_object_scale']=np.asarray(s1)
    if 'provenance_json' in pair:
        prior=json.loads(str(pair['provenance_json'].item()));prior.update(v3_variant='one-sided-scale',scale_choice='participant1',translation_choice='participant1_nominal',robot_choice='C-01 with-wrist unchanged');pair['provenance_json']=np.asarray(json.dumps(prior,sort_keys=True))
    assert np.array_equal(pair['robot_qpos'],original_robot) and np.array_equal(pair['object_qpos'][:,3:],original_rotation)
    output=dest/'core4d_pair_compact_fps30.npz';np.savez_compressed(output,**pair)
    dump(dest/'source_manifest.json',{'pair_sha256':sha(output),'variant':'one-sided-scale','parent_contract':e.checkpoint_contract,'participant1_nominal':{'path':str(p1),'sha256':sha(p1)},'participant2_nominal':{'path':str(p2),'sha256':sha(p2)},'scale1':s1,'scale2':s2,'full_mean_scale':mean,'robot_qpos_fixed':True,'rotation_fixed':True})
    runtime=dest/'core4d_pair_runtime_fps50.npz'
    build_core4d_pair_runtime_reference_file(output,runtime,expected_source_sha256=sha(output),source_manifest_path=dest/'source_manifest.json')
    new=load(runtime);diff={}
    for k in full:
        if k.startswith('agent_') or k in ['fps','joint_names','body_names','object_quat_w','object_ang_vel_w']:
            assert np.array_equal(full[k],new[k]),k;diff[k]='bitwise_equal'
    prom=json.loads((pkg/e.training_promotion_file).read_text());urdf=dest/Path(e.object_urdf_file).name
    _write_scaled_preview_urdf(pkg/e.object_urdf_file,urdf,s1/mean)
    mesh=base/prom['object_mesh_file'];shutil.copyfile(mesh,dest/mesh.name)
    oldtree=ET.parse(pkg/e.object_urdf_file);tree=ET.parse(urdf)
    assert tree.find('.//mass').get('value')==oldtree.find('.//mass').get('value')=='5'
    geom={'mass_kg':5,'length_ratio_to_full':s1/mean,'inertia_multiplier_to_full':(s1/mean)**2,'rule':'uniform geometry and COM scaling; fixed mass; inertia scales as length squared','same_mesh_sha256':sha(mesh)}
    prom={k:v for k,v in prom.items() if not k.startswith('physics_loading')}
    prom.update(source_pair_sha256=sha(output),source_manifest_sha256=sha(dest/'source_manifest.json'),runtime_reference_sha256=sha(runtime),training_object_urdf_sha256=sha(urdf),shared_object_scale=s1,training_ready=True,variant='one-sided-scale',physics_loading_check='pending_variant_smoke',geometry_inertia_rule=geom)
    dump(dest/'training_asset_manifest.json',prom)
    audit={'passed':True,'scope':'only object uniform scale, COM/inertia and nominal participant1 translation changed; all robot channels, object rotation/time fixed','unchanged_runtime':diff,'geometry':geom,'max_size_mismatch_percent':{'one_sided':100*max(abs(s1/s-1) for s in [s1,s2]),'shared_mean':100*max(abs(mean/s-1) for s in [s1,s2])}}
    dump(dest/'diff_to_full.json',audit)
    descriptor={'schema':'v3_reference_variant_v1','variant':'one-sided-scale','parent_experiment':'chair021','parent_contract':e.checkpoint_contract,'allowed_reward_variant':'baseline'}
    for prefix,p in [('source_pair',output),('runtime_reference',runtime),('object_urdf',urdf),('training_promotion',dest/'training_asset_manifest.json'),('diff',dest/'diff_to_full.json')]:descriptor[prefix+'_file']=str(p.relative_to(pkg));descriptor[prefix+'_sha256']=sha(p)
    dump(dest/'variant.json',descriptor)
    print(json.dumps({'output':str(dest),'status':'offline_asset_ready_pending_loader_and_physics_smoke','geometry':geom,'audit':audit}))
if __name__=='__main__':main()
