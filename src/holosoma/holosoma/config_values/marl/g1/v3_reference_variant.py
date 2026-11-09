"""Explicit opt-in reference variants; old experiment defaults are untouched."""
from dataclasses import replace
import hashlib,json
from pathlib import Path

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def with_reference_variant(experiment, manifest_path, *, reward_variant=None):
    if manifest_path is None:return experiment
    path=Path(manifest_path).resolve();d=json.loads(path.read_text())
    package=Path(__file__).resolve().parents[4]
    if d.get('schema')!='v3_reference_variant_v1' or d.get('variant') not in ('no-wrist','one-sided-scale','source-size-reopt'):raise ValueError('Unknown V3 reference schema/variant')
    if d['parent_experiment']!=experiment.experiment_id:raise ValueError('Variant parent differs')
    if json.dumps(d['parent_contract'],sort_keys=True)!=json.dumps(experiment.checkpoint_contract,sort_keys=True):raise ValueError('Variant Full parent identity differs')
    if reward_variant is not None and d['allowed_reward_variant']!=reward_variant:raise ValueError('Variant reward must match its Full control')
    for prefix in ['source_pair','runtime_reference','object_urdf','training_promotion','diff']:
        p=(package/d[prefix+'_file']).resolve()
        if not p.is_relative_to(package) or not p.is_file() or _sha(p)!=d[prefix+'_sha256']:raise ValueError('V3 asset hash/path mismatch: '+prefix)
    diff=json.loads((package/d['diff_file']).read_text())
    if diff.get('passed') is not True:raise ValueError('Variant field audit missing')
    if d['variant']=='no-wrist':
        if d['object_urdf_sha256']!=experiment.object_urdf_sha256:raise ValueError('No-wrist object changed')
        if diff.get('changed_qpos_columns_per_robot')!=[26,27,28,33,34,35]:raise ValueError('No-wrist field audit missing')
    elif d['variant']=='one-sided-scale':
        if experiment.experiment_id!='chair021' or d['allowed_reward_variant']!='baseline':raise ValueError('One-sided scale is restricted to Chair baseline')
        required=['agent_body_ang_vel_w','agent_body_lin_vel_w','agent_body_pos_w','agent_body_quat_w','agent_joint_pos','agent_joint_vel','body_names','joint_names','fps','object_ang_vel_w','object_quat_w']
        if any(diff.get('unchanged_runtime',{}).get(k)!='bitwise_equal' for k in required):raise ValueError('One-sided scale must retain all robot, time and object rotation channels')
        if diff.get('geometry',{}).get('mass_kg')!=experiment.object_mass_kg:raise ValueError('One-sided mass changed')
    else:
        if experiment.experiment_id!='chair021' or d['allowed_reward_variant']!='baseline':raise ValueError('Source-size reoptimization is Chair baseline only')
        if diff.get('full_robot_nominal_prior_indices')!=list(range(36)):raise ValueError('Full-configuration Stage2 prior required')
        for key in ['fresh_two_stage_and_a1','source_object_trajectory_preserved','a1_only_wrist_change']:
            if diff.get(key) is not True:raise ValueError('Missing source-size audit: '+key)
        if diff.get('geometry',{}).get('mass_kg')!=experiment.object_mass_kg or diff.get('geometry',{}).get('source_scale')!=1.0:raise ValueError('Source-size geometry/mass mismatch')
        for key in ['fps','joint_names','body_names','object_quat_w','object_ang_vel_w']:
            if diff.get('unchanged_runtime',{}).get(key)!='bitwise_equal':raise ValueError('Source-size time/rotation changed: '+key)
    promotion=json.loads((package/d['training_promotion_file']).read_text())
    for pkey,dkey in [('source_pair_sha256','source_pair_sha256'),('runtime_reference_sha256','runtime_reference_sha256'),('training_object_urdf_sha256','object_urdf_sha256')]:
        if promotion.get(pkey)!=d[dkey]:raise ValueError('Variant promotion mismatch')
    if promotion.get('training_ready') is not True:raise ValueError('Variant not ready for smoke')
    # Existing checkpoint contract already includes all three distinct input hashes,
    # and the promotion hash includes the explicit variant. Resume requires equality.
    return replace(experiment,source_pair_sha256=d['source_pair_sha256'],runtime_reference_file=d['runtime_reference_file'],runtime_reference_sha256=d['runtime_reference_sha256'],object_urdf_file=d['object_urdf_file'],object_urdf_sha256=d['object_urdf_sha256'],training_promotion_file=d['training_promotion_file'],training_promotion_sha256=d['training_promotion_sha256'])
