#!/usr/bin/env python3
"""Prepare a source/assets bundle and NON-executing run grid for paper supplements.

Does not connect, install, train, evaluate, push, or approve historical reuse.
Only the two hash-verified single-agent priors are included from legacy logs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from package_core4d_bucket_transfer import (
    TRAINING_ROBOT, asset_dependencies, git, identity, safe_file,
)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    audit_dir = args.audit_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Select a new output directory; packages are never overwritten')
    audit = json.loads((audit_dir / 'run_manifest.json').read_text())
    selected = set()
    for package in ('src/holosoma/holosoma', 'src/holosoma_retargeting/holosoma_retargeting'):
        selected.update(p.relative_to(root).as_posix() for p in (root / package).rglob('*.py')
                        if not any(part.startswith('converted') or part == '__pycache__' for part in p.parts))
    selected.update('scripts/' + name for name in (
        'train_core4d_smalltable.py', 'train_plan5_push.py', 'evaluate_core4d_smalltable.py',
        'smoke_core4d_smalltable.py', 'smoke_plan5_environment.py', 'smoke_plan5_ppo_update.py',
        'package_core4d_bucket_transfer.py', 'package_paper_supplement.py', 'audit_paper_supplement.py',
        'setup_isaacsim.sh', 'source_isaacsim_setup.sh', 'source_common.sh',
    ))
    selected.update(('src/holosoma/pyproject.toml', 'src/holosoma_retargeting/pyproject.toml',
                     'src/holosoma/MANIFEST.in', 'src/holosoma/README.md', 'src/holosoma/setup.py',
                     'docs/experiments/paper_supplement_20260929.md'))
    selected.update(asset_dependencies(root, (TRAINING_ROBOT,)))
    priors = {}
    configs = []

    def include(relative, expected=None):
        path = safe_file(root, relative)
        if expected and identity(root, relative)['sha256'] != expected:
            raise ValueError(f'Asset differs from historical hash: {relative}')
        if path.suffix in ('.urdf', '.xml', '.obj', '.mtl'):
            selected.update(asset_dependencies(root, (relative,)))
        else:
            selected.add(relative)

    for row in audit['historical_candidates']:
        config = row['stages'][-1]['config']
        for field, hash_field in (('runtime_reference', 'runtime_reference_sha256'),
                                  ('object_urdf', 'object_urdf_sha256'),
                                  ('training_promotion', 'training_promotion_sha256'),
                                  ('motion_file', None), ('paired_reference_file', None)):
            raw = config.get(field)
            if not raw:
                continue
            if Path(raw).is_absolute():
                raise ValueError(f'Unexpected absolute {field}: {raw}')
            rel = raw if (root / raw).is_file() else 'src/holosoma/' + raw
            include(rel, config.get(hash_field) if hash_field else None)
        bucket = config.get('bucket_reward_contract')
        if bucket:
            artifact = (Path(config['runtime_reference']).parent / 'interaction_vectors_v1.npz').as_posix()
            include(artifact, bucket['artifact_sha256'])
        if row['case'] in ('Push', 'Pull'):
            prior = next(a for a in row['asset_checks'] if a['field'] == 'source_checkpoint')
            if prior['match'] is not True:
                raise ValueError('Single-agent prior must have a matching recorded SHA256')
            p = Path(prior['path'])
            if p.is_symlink() or not p.is_file() or p.stat().st_size > 100 * 1024**2:
                raise ValueError(f'Invalid prior file: {p}')
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != prior['expected_sha256']:
                raise ValueError(f'Prior changed since audit: {p}')
            destination = config['source_checkpoint']
            # Exact legacy prior whitelist: no arbitrary logs or trained dual policies.
            expected_destinations = {
                'logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt',
                'logs/WholeBodyTracking/marl_compat_pull_v1/model_07999_actor158.pt',
            }
            if destination not in expected_destinations:
                raise ValueError(f'Unapproved checkpoint destination: {destination}')
            priors[destination] = {'source_path': str(p), 'sha256': actual, 'size_bytes': p.stat().st_size,
                                  'role': 'fixed_single_agent_WBT_prior_not_a_new_training_seed'}
        configs.append({'case': row['case'], 'configuration': row['configuration'], 'phase': row['phase'],
                        'budget_endpoint': row['target_final_iteration'],
                        'training_experiment': config.get('experiment', config.get('skill', row['case'].lower())),
                        'reward_variant': config.get('reward_variant', 'baseline'),
                        'source_candidate': row['checkpoint'],
                        'seeds': [721, 722, 723]})
    records = {name: identity(root, name) for name in sorted(selected)}
    tasks = []
    for config in configs:
        for seed in config['seeds']:
            tasks.append({'run_id': f"{config['configuration']}_seed{seed}",
                          'case': config['case'], 'configuration': config['configuration'],
                          'phase': config['phase'], 'training_seed': seed,
                          'budget_endpoint': config['budget_endpoint'],
                          'status': 'reuse_audit_pending' if seed == 721 else 'planned_not_started',
                          'resource': 'unassigned', 'num_envs': 2048, 'steps_per_env': 24,
                          'evaluations': {'nominal': 1, 'perturbed': 50},
                          'launch_enabled': False})
    plan = {'schema': 'paper_supplement_grid_v1',
            'resources_authorized': {'cloud_gpu': '2 x RTX 4090', 'local_gpu': 'RTX 5070 Laptop',
                                     'same_gpu_concurrency': 'benchmark aggregate throughput before enabling'},
            'resources_connected': False, 'training_started': False,
            'not_an_executable_queue': True,
            'launch_prerequisites': ['effective training protocol/source audit', 'host environment preflight',
                                     'source and asset verification', 'resource assignment and budget'],
            'configurations': configs, 'runs': tasks,
            'counts': {'target_main_models': 15, 'target_all_models': 21,
                       'new_trainings_if_seven_candidates_approved': 14,
                       'formal_perturbation_episodes': 1050, 'nominal_episodes': 21},
            'evaluation_status': 'existing evaluators bundled, new 50-relative-state protocol not implemented yet'}
    manifest = {'format': 'paper_supplement_source_assets_v1', 'base_git_head': git(root, 'rev-parse', 'HEAD'),
                'source_is_current_worktree': True, 'files': records, 'single_agent_priors': priors,
                'training_launch_ready': False, 'simulation_precheck_done': False,
                'exclusions': ['full datasets', 'old dual-agent models', 'training directories',
                               'Python environments', 'Isaac caches', 'credentials'],
                'note': 'Source snapshot + selected assets, not approval of historical comparability'}
    output.mkdir(parents=True)
    extras = {'package_manifest.json': encoded(manifest), 'run_plan.json': encoded(plan),
              'request.md': (audit_dir / 'request.md').read_bytes(),
              'source_tracked_diff.patch': git(root, 'diff', 'HEAD').encode()}
    hashes = [f"{value['sha256']}  payload/{name}" for name, value in records.items()]
    hashes += [f"{value['sha256']}  payload/{name}" for name, value in priors.items()]
    hashes += [f"{hashlib.sha256(raw).hexdigest()}  {name}" for name, raw in extras.items()]
    extras['SHA256SUMS'] = ('\n'.join(hashes) + '\n').encode()
    archive = output / 'paper_supplement_source_assets.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        def add(name, raw):
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            item.mode = 0o644
            item.mtime = 0
            tar.addfile(item, io.BytesIO(raw))
        for name, info in records.items():
            raw = safe_file(root, name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise RuntimeError('Source changed during packaging')
            add('payload/' + name, raw)
        for name, info in priors.items():
            raw = Path(info['source_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != info['sha256']:
                raise RuntimeError('Prior changed during packaging')
            add('payload/' + name, raw)
        for name, raw in extras.items():
            add(name, raw)
    for name, raw in extras.items():
        (output / name).write_bytes(raw)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / (archive.name + '.sha256')).write_text(checksum + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'sha256': checksum,
                      'MiB': archive.stat().st_size / 1024**2, 'runs': len(tasks),
                      'training_launch_ready': False}, indent=2))


if __name__ == '__main__':
    main()
