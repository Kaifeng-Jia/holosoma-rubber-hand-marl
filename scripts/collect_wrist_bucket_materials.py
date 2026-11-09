#!/usr/bin/env python3
"""Collect original bucket evidence and reaggregate existing episodes; no simulator.

The changed-file interpretations below are the explicit human-reviewed decisions
for these two frozen archives, not a generic automatic source-equivalence test.
Unknown differences fail closed. This script does not restore or modify sources.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from paper_evaluation_utils import sha, evaluation_result_dir
from summarize_paper_evaluation import summarize

REPO = Path(__file__).resolve().parents[1]
REL = 'src/holosoma/holosoma/'
REVIEWED = {
    'scripts/train_core4d_smalltable.py': 'Adds other allowed experiments/ablations; selected bucket variant and reference directory resolve identically; collector/update loop unchanged.',
    'scripts/smoke_core4d_smalltable.py': 'Adds table experiment; selected bucket reference path and physics checks unchanged.',
    'scripts/evaluate_core4d_smalltable.py': 'Adds table experiment; selected bucket branch and evaluation policy unchanged.',
    REL+'agents/mappo/core4d_smalltable_ppo.py': 'Only metadata validation supports new table branch; bucket contact target unchanged; PPO update implementation byte-identical.',
    REL+'config_values/marl/g1/core4d_bucket_contract.py': 'Additional variants/table target validation; complete selected bucket contract equal; original A weights remain [1,1,1,2], RZ0 [1,1,0,0].',
    REL+'config_values/marl/g1/core4d_pair_experiments.py': 'Adds preceding smalltable5kg_A branch; selected bucket003 experiment unchanged.',
    REL+'config_values/marl/g1/core4d_bucket_reward.py': 'Generalizes variant validation only; selected reward manager and sensing unchanged.',
    REL+'managers/reward/terms/core4d_bucket.py': 'Adds inactive table validation and (A archive only) generalized fixed block weights; selected A/RZ0 formulas and nongated contact behavior equivalent.',
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def pure_contract(source, variant, contract):
    """Execute only literal globals and pure contract functions, not import modules."""
    import re
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
             all(isinstance(t, ast.Name) for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and
              n.name in ('bucket_block_weights', 'bucket_reward_contract')]
    scope = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pure_original_contract>', 'exec'), scope)
    return scope['bucket_reward_contract'](variant, contract['artifact_sha256'],
            contract['runtime_reference_sha256'], contract['training_robot_urdf_sha256'])


def collect(output):
    bucket = output/'bucket'; evidence = bucket/'source_evidence'
    root = REPO/'logs/PaperSupplement20260929/unified_evaluation_20261003'
    new_manifest_path = REPO/'logs/PaperSupplement20260929/autodl_received/operations/verified_source/package_manifest.json'
    new_manifest = json.loads(new_manifest_path.read_text())
    shutil.copy2(new_manifest_path, evidence/'new_training_verified_package_manifest.json')
    shutil.copy2(REPO/'logs/PaperSupplement20260929/autodl_received/operations/source_verified.json', evidence/'new_training_source_verified.json')
    models = json.loads((root/'models.json').read_text())
    records = []
    cases = [
        ('A', 'B-A', Path('/home/kevin/CORE4D_BUCKET_AB_TRANSFER_20260918/core4d_bucket_transfer.tar.gz'),
         'f25df6c05dc370cf425c7f36524284e782e61c4628e8a6a1aeba18356687c254', 'A'),
        ('RZ0', 'B-RZ0', Path('/home/kevin/CORE4D_BUCKET_BOTH_REMOVED_TRANSFER_20260919/core4d_bucket_transfer.tar.gz'),
         '2d7b7620039dc3c25138153f18019ce0bf052c42d118f1337c1215de4d187050', 'A_no_rel_no_height'),
    ]
    for label, conf, archive, expected, variant in cases:
        assert sha(archive) == expected
        config_path = root/f'training_evidence/{conf}_seed721/stage0_run_config.json'
        config = json.loads(config_path.read_text())
        record = dict(model_id=conf+'_seed721', archive=str(archive), archive_sha256=expected,
                      config_sha256=sha(config_path), base_git_head=config['git_commit'],
                      status='historical_effective_code_verified', missing_critical_items=[],
                      caveat='Historical training final process exit remains unknown; unified evaluation has explicit validated exit0. Source audit does not prove deterministic equality across hardware.')
        model = next(m for m in models if m['model_id'] == record['model_id'])
        assert sha(model['checkpoint']) == model['checkpoint_sha256']
        record['checkpoint_sha256'] = model['checkpoint_sha256']
        with tarfile.open(archive) as tar:
            manifest = json.load(tar.extractfile('package_manifest.json'))
            assert manifest == json.loads((evidence/f'{label}_package_manifest.json').read_text())
            assert manifest['base_git_head'] == config['git_commit']
            payload_checked = 0; changed = []; identical = []; compared_to_new = []; excluded = []
            for rel, info in manifest['files'].items():
                blob = tar.extractfile('payload/'+rel).read()
                digest = hashlib.sha256(blob).hexdigest()
                assert digest == info['sha256'], rel
                assert len(blob) == info['size_bytes'], rel
                payload_checked += 1
                if not rel.endswith('.py'):
                    continue
                current = (REPO/rel).read_bytes()
                if rel in new_manifest['files']:
                    assert hashlib.sha256(current).hexdigest() == new_manifest['files'][rel]['sha256'], rel
                    compared_to_new.append(rel)
                else:
                    assert blob == current, rel
                    excluded.append(rel)
                if blob == current:
                    identical.append(rel)
                else:
                    assert rel in REVIEWED, rel
                    dest = evidence/f'{label}_original'/rel
                    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(blob)
                    dest = evidence/'current_comparison'/rel
                    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(current)
                    patch = ''.join(difflib.unified_diff(blob.decode().splitlines(True), current.decode().splitlines(True),
                                    fromfile='historical/'+rel, tofile='current/'+rel))
                    (evidence/f'{label}_{rel.replace("/", "_")}.patch').write_text(patch)
                    changed.append(dict(path=rel, historical_sha256=digest, current_sha256=sha(REPO/rel),
                                        interpretation=REVIEWED[rel], active_bucket_behavior_changed=False))
                if rel == REL+'config_values/marl/g1/core4d_bucket_contract.py':
                    assert pure_contract(blob.decode(), variant, config['bucket_reward_contract']) == config['bucket_reward_contract']
                    assert pure_contract(current.decode(), variant, config['bucket_reward_contract']) == config['bucket_reward_contract']
                if rel.endswith(('/core4d_smalltable_runner.py','/core4d_smalltable_manager.py',
                        '/core4d_bucket_manager.py','/core4d_pair_manager.py','/mappo/ppo.py',
                        '/mappo/storage.py','/core4d_smalltable_initialization.py',
                        '/core4d_smalltable_evaluation.py','/simulator/isaacsim/isaacsim.py',
                        '/managers/reward/terms/core4d_smalltable.py','/managers/reward/terms/marl.py',
                        '/envs/marl/plan5_push_manager.py','/managers/action/terms/marl.py',
                        '/managers/action/terms/joint_control.py','/managers/observation/terms/marl.py',
                        '/managers/termination/terms/core4d_smalltable.py','/managers/termination/terms/marl.py',
                        '/managers/reward/terms/interaction_vectors.py','/config_values/marl/g1/action.py',
                        '/config_values/marl/g1/core4d_smalltable_observation.py',
                        '/config_values/marl/g1/core4d_smalltable_action.py',
                        '/config_values/marl/g1/core4d_smalltable_termination.py')):
                    dest = evidence/f'{label}_original'/rel
                    dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(blob)
            record.update(payload_files_verified=payload_checked, byte_identical_python_files=identical,
                          changed_python_files_reviewed=changed, current_matches_verified_new_deployment=compared_to_new,
                          preparation_only_files_not_in_new_payload_but_byte_identical=excluded,
                          contract_function_outputs_equal=True)
            asset_checks = []
            for rel, info in manifest['files'].items():
                if 'core4d_bucket003_20231020_071_a1/' in rel or rel.endswith('main_mesh_collision_rubberhand.urdf'):
                    actual = sha(REPO/rel); assert actual == info['sha256'], rel
                    asset_checks.append(dict(path=rel, expected_sha256=info['sha256'], actual_sha256=actual))
            record['asset_checks'] = asset_checks
        if label == 'A':
            launch = (evidence/'A_actual_launch_CN.md').read_text()
            assert expected in launch and 'A_fresh12000_env2048_seed721' in launch
            record['historical_source_to_run_link'] = 'source_evidence/A_actual_launch_CN.md: actual A command, archive SHA and463 matching launch hashes'
        else:
            proof = json.loads((evidence/'RZ0_source_verified.json').read_text())
            assert proof['verified'] and proof['archive_sha256'] == expected
            assert json.loads((evidence/'RZ0_source_package_manifest.json').read_text()) == json.loads((evidence/'RZ0_package_manifest.json').read_text())
            record['historical_source_to_run_link'] = 'source_evidence/RZ0_source_verified.json + formal.started.json + formal.start_verified.json'
        comparisons = []
        for seed in (722, 723):
            target_dir = root/f'training_evidence/{conf}_seed{seed}'
            target = json.loads((target_dir/'stage0_run_config.json').read_text())
            differences = [k for k in sorted(set(config)|set(target)) if config.get(k) != target.get(k)]
            assert set(differences) <= {'bucket_reference','git_commit','seed','started_unix_time'}, differences
            stages = json.loads((target_dir/'stages.json').read_text())
            intervals = [s['included_iteration_interval'] for s in stages]
            assert intervals[0][0] == 1 and intervals[-1][1] == 12000
            for x, y in zip(intervals, intervals[1:]): assert x[1]+1 == y[0]
            resumed_config_checks = []
            for stage in stages[1:]:
                path = target_dir/f'stage{stage["stage"]}_run_config.json'
                actual = json.loads(path.read_text())
                changed_keys = [k for k in sorted(set(actual)|set(target)) if actual.get(k)!=target.get(k)]
                allowed = {'actor_learning_rate','initialization','iterations_requested','resume_checkpoint',
                           'resume_checkpoint_sha256','start_iteration','started_unix_time'}
                assert set(changed_keys)<=allowed, changed_keys
                resumed_config_checks.append(dict(config_sha256=sha(path), changed_keys=changed_keys,
                            learning_rate_change='Checkpoint-restored adaptive KL actor LR; same learner implementation, not new reward/physics parameters.'))
            shutil.copy2(target_dir/'stages.json', evidence/f'{conf}_seed{seed}_stages.json')
            comparisons.append(dict(seed=seed, original_config_sha256=sha(target_dir/'stage0_run_config.json'),
                                    non_effective_metadata_differences=differences,
                                    all_other_configuration_values_exactly_equal=True,
                                    included_training_intervals=intervals,
                                    resumed_config_checks=resumed_config_checks))
        record['new_seed_comparisons'] = comparisons
        record['selected_recipe'] = config['bucket_reward_contract']
        record['training_settings'] = {k:config[k] for k in ('initialization','iterations_requested','num_envs','steps_per_env',
                'actor_input_dim','critic_input_dim','action_dim_per_agent','physics_hz','control_hz','save_interval','reward_variant','reward_terms','termination_terms')}
        record['object_physics_arrays_identical_across_seeds'] = True
        records.append(record)
    audit = dict(version='bucket_original_source_recovery_20261005', current_git_head='2b6668b1205ee5c5b81ae9251744b16c799c7173',
                 scope='Read original archives/configs; manually review selected effective paths; reuse existing evaluations only.',
                 new_verified_deployment_manifest_sha256=sha(new_manifest_path), records=records,
                 previous_pending_status_reason='10-04 search had not found the original /home/kevin frozen archives and local A launch record; this review adds originals, not reconstructed snapshots.',
                 simulations_or_training_started=False, rewards_or_physics_changed=False)
    write(bucket/'bucket_source_audit.json', audit)
    updated = bucket/'updated_results'; updated.mkdir(exist_ok=False)
    shutil.copy2(root/'models.json', updated/'models_before_source_recovery.json')
    adopted = []
    for model in models:
        if model['model_id'] in {r['model_id'] for r in records}:
            model['reuse_status'] = 'historical_effective_code_verified'
            model['source_audit_reference'] = '../bucket_source_audit.json'
        source = evaluation_result_dir(root, model)/'summary.json'
        dest = evaluation_result_dir(updated, model)/'summary.json'; dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        exit_path = root/(model['model_id']+'.exit.json')
        shutil.copy2(exit_path, updated/exit_path.name)
        summary = json.loads(source.read_text())
        assert summary['complete'] and summary['episode_count']==51
        exit_record = json.loads(exit_path.read_text())
        assert exit_record['validated'] and exit_record['process_returncode']==0
        state_ids = [e['state_id'] for e in summary['episodes']]
        assert len(state_ids)==len(set(state_ids))==51
        adopted.append(dict(model_id=model['model_id'], checkpoint_sha256=model['checkpoint_sha256'],
                            summary_original_path=str(source), summary_sha256=sha(source),
                            process_exit_record_path=str(exit_path), process_exit_record_sha256=sha(exit_path),
                            state_ids=state_ids, episode_count=51,
                            excluded_attempts=model.get('excluded_evaluation_attempts', [])))
    write(updated/'models.json', models); write(updated/'adopted_evaluations.json', adopted)
    assert summarize(updated)==21
    shutil.copy2(REPO/'scripts/summarize_paper_evaluation.py', updated/'summarize_paper_evaluation.py')
    shutil.copy2(REPO/'scripts/paper_evaluation_utils.py', updated/'paper_evaluation_utils.py')
    media = root/'media/B-RZ0_seed721'
    manifest = json.loads((media/'manifest.json').read_text())
    write(bucket/'existing_nominal_media.json', dict(model_id='B-RZ0_seed721',
          reuse_status='historical_effective_code_verified', selection='Existing preselected nominal; not reselected or rerendered',
          media_relative_directory_in_previous_package='media/B-RZ0_seed721', original_manifest=manifest,
          files=[dict(path=str(p),name=p.name,sha256=sha(p)) for p in sorted(media.iterdir()) if p.is_file()],
          seed722_failure_not_rerendered='Conditional fallback not needed now that721 source recovered; original failure remains in original results.',
          failure_states_available=[dict(path=str(root/f'results/B-RZ0_seed{s}/nominal.trajectory.npz'),sha256=sha(root/f'results/B-RZ0_seed{s}/nominal.trajectory.npz')) for s in (722,723)]))
    print('Both old721 source identities verified;21 models/1050 perturbations/21 nominal reaggregated offline.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    collect(parser.parse_args().output.resolve())
