"""Promote a fresh source-size Stage1/Stage2/A1 Chair solve as an explicit V3 variant."""
from pathlib import Path
import hashlib,json,shutil,sys,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/holosoma'),str(ROOT/'src/holosoma_retargeting'),str(ROOT/'scripts')]
from holosoma.config_values.marl.g1.core4d_pair_experiments import get_core4d_pair_experiment
from holosoma_retargeting.core4d_pair_runtime_reference import build_core4d_pair_runtime_reference_file
from export_core4d_small_table_pair import _write_scaled_preview_urdf
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def dump(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def main():
 source=ROOT/'logs/V3_20261102/U3_assets/chair_source_size_reopt_20261103'
 decision=json.loads((source/'rev112_preview_complete.json').read_text())
 assert decision['passed_rule_a'] and decision['all_solves_converged'] and decision['over6_at_limit_and_converged_fraction']>=.9
 stage=source/'two_stage';refined=source/'a1_wrist_only_rev112_preview';sm=json.loads((stage/'manifest.json').read_text());am=json.loads((refined/'manifest.json').read_text())
 assert sm['input_sha256']=='e496e71901d50808b1d29d533b358117c64a1301b5244dfa134850fb8d729acc'
 assert sm['solver']['stage2_nominal_tracking_indices']==list(range(36))
 assert sm['solver']['stage2_nominal_tracking_scope']=='full_robot_qpos_soft_prior'
 assert sm['method']=='two-stage' and sm['shared_physical_object_qpos_exact']
 a=load(stage/'core4d_pair_reference.npz');b=load(refined/'core4d_pair_reference.npz');mask=np.ones(36,dtype=bool);mask[[26,27,28,33,34,35]]=False
 assert np.array_equal(a['robot_qpos'][:,:,mask],b['robot_qpos'][:,:,mask])
 assert np.array_equal(a['object_qpos'],b['object_qpos']) and np.array_equal(a['fps'],b['fps'])
 assert am['a1']['non_wrist_qpos_exact'] and am['a1']['shared_object_qpos_exact']
 pkg=ROOT/'src/holosoma';full=get_core4d_pair_experiment('chair021');parent=(pkg/full.runtime_reference_file).parent
 dest=parent.parent/'v3_20261102/chair_source_size_reopt';dest.mkdir(parents=True,exist_ok=False)
 pair=dest/'core4d_pair_compact_fps30.npz';shutil.copy2(refined/'core4d_pair_reference.npz',pair)
 manifest={'variant':'source-size-reopt','pair_sha256':sha(pair),'source_canonical_sha256':sm['input_sha256'],'stage2_manifest_sha256':sha(stage/'manifest.json'),'a1_manifest_sha256':sha(refined/'manifest.json'),'source_job':str(source),'solver':sm['solver'],'a1':am['a1'],'shared_object_scale':1.0,'reoptimization_completed':True}
 dump(dest/'source_manifest.json',manifest)
 rt=dest/'core4d_pair_runtime_fps50.npz';build_core4d_pair_runtime_reference_file(pair,rt,expected_source_sha256=sha(pair),source_manifest_path=dest/'source_manifest.json')
 before=load(pkg/full.runtime_reference_file);after=load(rt)
 for k in ['fps','joint_names','body_names','object_quat_w','object_ang_vel_w']:assert np.array_equal(before[k],after[k]),k
 assert before['agent_joint_pos'].shape==after['agent_joint_pos'].shape
 assert not np.array_equal(before['agent_joint_pos'],after['agent_joint_pos'])
 oldprom=json.loads((pkg/full.training_promotion_file).read_text());ratio=1/oldprom['shared_object_scale'];urdf=dest/Path(full.object_urdf_file).name
 _write_scaled_preview_urdf(pkg/full.object_urdf_file,urdf,ratio)
 tree=ET.parse(urdf);assert float(tree.find('.//mass').get('value'))==5.0
 for mesh in tree.findall('.//mesh'):assert np.allclose([float(x) for x in mesh.get('scale','1 1 1').split()],1,atol=1e-10)
 shutil.copy2(parent/oldprom['object_mesh_file'],dest/oldprom['object_mesh_file'])
 geom={'mass_kg':5.0,'source_scale':1.0,'length_ratio_to_full':ratio,'inertia_multiplier_to_full':ratio**2,'material':[.5,.5,0.0]}
 diff={'passed':True,'variant':'source-size-reopt','fresh_two_stage_and_a1':True,'source_object_trajectory_preserved':True,'full_robot_nominal_prior_indices':list(range(36)),'a1_only_wrist_change':True,'geometry':geom,'unchanged_runtime':{k:'bitwise_equal' for k in ['fps','joint_names','body_names','object_quat_w','object_ang_vel_w']},'robot_and_translation_change_expected':True,'max_joint_change_vs_full_rad':float(np.max(np.abs(before['agent_joint_pos']-after['agent_joint_pos'])))}
 diff['rev112_residual_decision']=decision
 dump(dest/'diff_to_full.json',diff)
 prom={k:v for k,v in oldprom.items() if not k.startswith('physics_loading') and k not in ['reviewed_on','review_decision','formal_training_authorized_on','authorized_training','source_manifest_status_preserved']}
 prom.update(variant='source-size-reopt',shared_object_scale=1.0,source_pair_sha256=sha(pair),source_manifest_sha256=sha(dest/'source_manifest.json'),runtime_reference_sha256=sha(rt),training_object_urdf_sha256=sha(urdf),training_ready=True,formal_training_authorized=True,physics_loading_check='pending_variant_smoke',geometry_inertia_rule=geom,inertia_basis='fixed5kg source-size geometry, Full inertia scaled by inverse mean scale squared',authority_commit='434bedb692ad2f4e2922f417829f4b1d47f7cfc7',rev112_residual_decision=decision)
 dump(dest/'training_asset_manifest.json',prom)
 d={'schema':'v3_reference_variant_v1','variant':'source-size-reopt','parent_experiment':'chair021','parent_contract':full.checkpoint_contract,'allowed_reward_variant':'baseline'}
 for key,path in [('source_pair',pair),('runtime_reference',rt),('object_urdf',urdf),('training_promotion',dest/'training_asset_manifest.json'),('diff',dest/'diff_to_full.json')]:d[key+'_file']=str(path.relative_to(pkg));d[key+'_sha256']=sha(path)
 dump(dest/'variant.json',d);print(json.dumps({'variant':str(dest/'variant.json'),'offline_checks_passed':True,'physics_smoke_pending':True}))
if __name__=='__main__':main()
