#!/usr/bin/env python3
"""Read trusted local model metadata; write a new, simulator-free audit directory.

This is an inventory, not automatic approval of historical comparability.
No model tensors, environments or full training logs are copied into the report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git(root, *args):
    proc = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    return {'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def compact(value):
    # Historical physics records duplicate values for all 2048 environments.
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 16:
            same = all(item == value[0] for item in value)
            return {'count': len(value), 'all_equal': same,
                    'first': compact(value[0]), 'last': compact(value[-1])}
        return [compact(item) for item in value]
    return value


def inventory(root, legacy):
    return [
        ('Push', 'Push-livePD', 'main', legacy, 'Plan5Push', [
            'a1_livepd_20kg_critic50_seed721_env2048',
            'a1_livepd_20kg_full8000_from_critic50_seed721_env2048',
            'a1_livepd_20kg_continue7000_from_08050_seed721_env2048'], 13050),
        ('Pull', 'Pull-baseline', 'main', legacy, 'Plan5Pull', [
            'pull_20kg_critic50_seed721_env2048',
            'pull_20kg_full8000_from_critic50_seed721_env2048'], 8050),
        ('Chair', 'C-01', 'main', root, 'Core4DChair', [
            'chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910'], 12000),
        ('Bucket', 'B-RZ0', 'main', root, 'Core4DBucket', [
            'A_no_rel_no_height_fresh12000_env2048_seed721_cloud_20260919'], 12000),
        ('Small Table', 'T-07', 'main', root, 'Core4DSmallTableA', [
            'A_no_rel_no_height_5kg_fresh12000_env2048_seed721_cloud_20260920'], 12000),
        ('Bucket', 'B-A', 'contrast', root, 'Core4DBucket', [
            'A_fresh12000_env2048_seed721'], 12000),
        ('Small Table', 'T-06', 'contrast', root, 'Core4DSmallTableA', [
            'A_5kg_fresh12000_env2048_seed721_cloud_20260919'], 12000),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--legacy-root', type=Path, default=Path('/home/kevin/holosoma-rubber-hand-marl'))
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Choose a new audit directory; historical reports are never overwritten')
    if not args.request.is_file():
        raise FileNotFoundError(args.request)
    # Deliberately no holosoma/Isaac imports or GPU tensor allocations.
    import torch
    output.mkdir(parents=True)
    (output / 'request.md').write_bytes(args.request.read_bytes())
    current = {'root': str(root), 'head': git(root, 'rev-parse', 'HEAD'),
               'branch': git(root, 'branch', '--show-current'),
               'status': git(root, 'status', '--short')}
    source_files = sorted(root.glob('src/holosoma/**/*.py')) + sorted(root.glob('scripts/*.py'))
    current['source_sha256'] = {str(p.relative_to(root)): sha(p) for p in source_files}
    dump(output / 'current_source.json', current)
    (output / 'current_tracked_diff.patch').write_text(git(root, 'diff', 'HEAD')['stdout'])
    rows = []
    for case, identity, phase, repo, project, stages, endpoint in inventory(root, args.legacy_root):
        row = {'case': case, 'configuration': identity, 'phase': phase, 'training_seed': 721,
               'target_final_iteration': endpoint, 'reuse_status': 'pending_source_and_protocol_audit',
               'stages': [], 'missing': [], 'asset_checks': [], 'source_packages': []}
        for stage in stages:
            folder = repo / 'logs' / project / stage
            record = {'directory': str(folder)}
            config_path = folder / 'run_config.json'
            if not config_path.is_file():
                row['missing'].append(str(config_path))
                row['stages'].append(record)
                continue
            config = json.loads(config_path.read_text())
            record.update(config_sha256=sha(config_path), config=compact(config))
            status = folder / 'status.json'
            record['status'] = json.loads(status.read_text()) if status.is_file() else None
            record['process_exit_code'] = 'unknown_unless_explicit_exit_record_below'
            durations = []
            eligible_durations = []
            iterations = []
            metrics = folder / 'metrics.jsonl'
            if metrics.is_file():
                with metrics.open() as stream:
                    for line in stream:
                        item = json.loads(line)
                        iterations.append(item['iteration'])
                        if isinstance(item.get('elapsed_seconds'), (int, float)):
                            durations.append(item['elapsed_seconds'])
                            if item['iteration'] <= endpoint:
                                eligible_durations.append(item['elapsed_seconds'])
                record['metrics'] = {
                    'path': str(metrics), 'sha256': sha(metrics), 'count': len(iterations),
                    'first_iteration': iterations[0] if iterations else None,
                    'last_iteration': iterations[-1] if iterations else None,
                    'consecutive': all(b == a + 1 for a, b in zip(iterations, iterations[1:])),
                    'recorded_update_hours': sum(durations) / 3600,
                    'update_hours_through_selected_endpoint': sum(eligible_durations) / 3600,
                    'timing_note': 'Existing host only; excludes startup and is not a cloud GPU estimate',
                }
            row['stages'].append(record)
            for path_key, hash_key in (
                ('runtime_reference', 'runtime_reference_sha256'), ('object_urdf', 'object_urdf_sha256'),
                ('training_promotion', 'training_promotion_sha256'),
                ('source_checkpoint', 'source_checkpoint_sha256'),
            ):
                raw = config.get(path_key)
                if not raw:
                    continue
                candidates = [Path(raw)] if Path(raw).is_absolute() else [repo / raw, repo / 'src/holosoma' / raw]
                path = next((p for p in candidates if p.is_file()), candidates[0])
                actual = sha(path) if path.is_file() else None
                expected = config.get(hash_key)
                row['asset_checks'].append({'stage': stage, 'field': path_key, 'path': str(path),
                    'expected_sha256': expected, 'actual_sha256': actual,
                    'match': actual == expected if actual and expected else None})
            for package in sorted(folder.glob('**/*manifest.json')):
                if not any(s in package.name for s in ('package_manifest', 'source_package_manifest')):
                    continue
                data = json.loads(package.read_text())
                files = data.get('files', {})
                if not isinstance(files, dict):
                    continue
                matched, different, missing = [], [], []
                for name, meta in files.items():
                    if not name.endswith('.py'):
                        continue
                    historical_hash = meta.get('sha256') if isinstance(meta, dict) else meta
                    local = root / name
                    if not local.is_file():
                        missing.append(name)
                    elif sha(local) == historical_hash:
                        matched.append(name)
                    else:
                        different.append(name)
                row['source_packages'].append({'path': str(package), 'sha256': sha(package),
                    'matching_python_files': len(matched), 'different_python_files': different,
                    'missing_python_files': missing,
                    'note': 'Byte equality only; differences require effective-code analysis'})
            record['explicit_exit_records'] = [
                {'path': str(p), 'record': json.loads(p.read_text())}
                for p in sorted(folder.glob('**/formal.exit.json'))
            ]
        final = repo / 'logs' / project / stages[-1] / f'model_{endpoint:05d}.pt'
        row['checkpoint'] = {'path': str(final), 'exists': final.is_file()}
        if final.is_file():
            state = torch.load(final, map_location='cpu', weights_only=False)
            row['checkpoint'].update(sha256=sha(final), bytes=final.stat().st_size,
                iteration=state.get('iter'),
                contract=state.get('core4d_smalltable_mappo', state.get('plan5_mappo')),
                actor_parameter_shapes={k: list(v.shape) for k, v in state['actor_model_state_dict'].items()})
            config = row['stages'][-1].get('config', {})
            contract = row['checkpoint']['contract'] or {}
            row['checkpoint']['configuration_matches'] = {
                k: contract[k] == config[k] for k in ('runtime_reference_sha256', 'object_urdf_sha256',
                                                     'training_promotion_sha256')
                if k in contract and k in config}
            if 'bucket_reward' in contract:
                row['checkpoint']['configuration_matches']['bucket_reward'] = (
                    contract['bucket_reward'] == config.get('bucket_reward_contract'))
            del state
        else:
            row['missing'].append(str(final))
        rows.append(row)
        print(identity, 'checkpoint_present=' + str(final.is_file()), 'reuse=pending', flush=True)
    manifest = {'schema': 'paper_supplement_audit_v1',
        'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'request_sha256': sha(args.request), 'cloud_training_only': True,
        'execution': {'new_training_runs_started': 0, 'formal_evaluations': 0,
                      'cloud_connection': 'awaiting_current_connection_and_budget'},
        'targets': {'main_models': 15, 'all_models': 21, 'training_seeds': [721, 722, 723],
                    'perturbed_episodes_per_model': 50, 'nominal_episodes_per_model': 1,
                    'new_runs_if_all_reusable': 14, 'approved_reused_models': 0},
        'historical_candidates': rows}
    dump(output / 'run_manifest.json', manifest)
    report = ['# 补实验第一轮盘点', '', '这是候选模型审计，不是复用批准，也不是新增实验结果。', '',
              '| 配置 | 目标终点 | 历史模型 | 复用状态 |', '|---|---:|---|---|']
    for row in rows:
        report.append(f"| {row['configuration']} | {row['target_final_iteration']} | "
                      f"{'已找到' if row['checkpoint']['exists'] else '缺失'} | 待代码/协议审计 |")
    report += ['', '## 已明确的执行条件', '',
               '- 用户指定云端训练；当前连接与可用时长/费用上限待提供。本轮未连接旧云端，也未启动本地训练。',
               '- 七个旧配置均先作为 seed 721 候选，不能把模型存在、status passed 或同一评测 seed 当作可比性证明。',
               '- Push 候选链为 50 + 8000 + 5000；保留 8050 处进程重启/续训事实，后续 2000 轮不计入 13050 预算。',
               '- Pull 为独立 critic 预热 50 + 完整训练 8000；CORE4D 为 fresh 12000。',
               '- B-RZ0/T-07 嵌套奖励为 wz=2、独立高度=0、关系=0、门控关闭；旧顶层 wz=1 不代表实际桶式奖励。',
               '- 完整 A 对照同时加入高度与关系，不能把差异单独归因于一项。',
               '- 旧桶 A 的提交号不足以复原当时未提交代码；需继续核对同期源文件快照。',
               '- 旧评测不能填入新 50 状态协议；所有正式评测、真实 reset 和统计汇总仍待执行。',
               '- 详见 run_manifest.json：模型哈希、元数据、配置、资产核对、历史更新时间和代码包差异。',
               '- 资产哈希缺失记为未知；未捕获的历史工作区差异不会用当前代码补写。', '']
    (output / 'REPORT_CN.md').write_text('\n'.join(report))
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(output)}\n' for p in files))
    print('REPORT', output)


if __name__ == '__main__':
    main()
