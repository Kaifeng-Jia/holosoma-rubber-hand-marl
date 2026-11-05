#!/usr/bin/env python3
"""Copy a bounded, portable evidence snapshot; never run a simulator or policy.

Outputs only beneath this script's directory. Original logs are read-only.
Run on the originating Ubuntu checkout; Mac readers need not run this script.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

DEST = Path(__file__).resolve().parent
REPO = DEST.parents[2]
HISTORY = Path('/home/kevin/holosoma-rubber-hand-marl')
FILES: list[dict] = []
MISSING: list[str] = []


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(relative: str, value: object) -> Path:
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return target


def copy(source: Path, relative: str, compressed: bool = False) -> dict | None:
    if not source.is_file():
        MISSING.append(str(source))
        return None
    raw = source.read_bytes()
    content = gzip.compress(raw, compresslevel=9, mtime=0) if compressed else raw
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    record = dict(source_absolute=str(source), copied_relative=relative,
                  source_sha256=sha(raw), copy_sha256=sha(content),
                  source_bytes=len(raw), copy_bytes=len(content),
                  transformation='lossless_gzip' if compressed else 'byte_identical')
    FILES.append(record)
    return record


def abbreviated(value: object) -> object:
    """Keep every key; abbreviate long arrays in the *derived* readable extract."""
    if isinstance(value, dict):
        return {k: abbreviated(v) for k, v in value.items()}
    if isinstance(value, list) and len(value) > 32:
        serialized = [json.dumps(v, sort_keys=True) for v in value]
        return {'_derived_large_array_summary': True, 'length': len(value),
                'all_entries_equal': len(set(serialized)) == 1,
                'first_entry': abbreviated(value[0]),
                'last_entry': abbreviated(value[-1]),
                'note': 'Full original array is preserved in run_config.json.gz.'}
    if isinstance(value, list):
        return [abbreviated(v) for v in value]
    return value


def config_copy(source: Path, subdir: str) -> dict:
    original = copy(source, f'{subdir}/run_config.json.gz', compressed=True)
    if original is None:
        return {}
    config = json.loads(source.read_text())
    write_json(f'{subdir}/config_extract.json', {
        '_artifact_kind': 'derived_readable_config_not_the_original',
        '_source_sha256': original['source_sha256'],
        '_original_lossless_copy': 'run_config.json.gz',
        '_transformation': 'All keys retained; arrays longer than 32 entries summarized.',
        'config': abbreviated(config),
    })
    return config


def completion(summary: dict) -> dict:
    episodes = summary.get('episodes', [])
    if not isinstance(episodes, list):
        return {'definition': 'This file does not use CORE4D reference-completion episodes.'}
    return {'completed_reference': sum(e.get('completed_reference') is True for e in episodes),
            'episodes': len(episodes), 'checkpoint_iteration': summary.get('checkpoint_iteration'),
            'evaluation_seed': summary.get('seed'),
            'inference': summary.get('inference'), 'observation_noise': summary.get('observation_noise'),
            'definition': 'Reached reference horizon in this recorded evaluation; not a hand-use or lifting success label.'}


def core4d() -> list[dict]:
    text = (REPO / 'DEMO_AND_EXPERIMENT_INVENTORY.md').read_text()
    section = text.split('## 9. 实验身份与文件入口', 1)[1]
    rows = [line for line in section.splitlines() if re.match(r'^\| (?:B-|T-|C-)\w+', line)]
    assert len(rows) == 13, f'Expected exactly 13 registry rows, got {len(rows)}'
    experiments = []
    for row in rows:
        columns = row.split('|')
        identifier, label, physics = (columns[i].strip() for i in (1, 2, 3))
        config_path = re.search(r'\[配置\]\(([^)]+)\)', row).group(1)
        summary_path = re.search(r'\[summary\]\(([^)]+)\)', row).group(1)
        config = config_copy(REPO / config_path, identifier)
        copy(REPO / summary_path, f'{identifier}/summary.json')
        copy((REPO / config_path).with_name('status.json'), f'{identifier}/status.json')
        summary = json.loads((REPO / summary_path).read_text())
        experiments.append({
            'id': identifier, 'label_cn': label, 'physics_registry_cn': physics,
            'source_run_directory': str((REPO / config_path).parent),
            'config_extract': f'{identifier}/config_extract.json',
            'config_original_gzip': f'{identifier}/run_config.json.gz',
            'primary_summary': f'{identifier}/summary.json',
            'reward_variant_raw': config.get('reward_variant'),
            'reward_terms_raw': config.get('reward_terms'),
            'bucket_reward_contract_raw': config.get('bucket_reward_contract'),
            'interaction_contract_raw': config.get('interaction_contract'),
            'object_position_tracking_contract_raw': config.get('object_position_tracking_contract'),
            'top_level_object_z_error_weight_raw': config.get('object_z_error_weight'),
            'reward_contract_warning': 'For bucket-style rewards the nested bucket_reward_contract is authoritative; the generic top-level z field can be stale.',
            'training_seed': config.get('seed'), 'num_envs': config.get('num_envs'),
            'steps_per_env': config.get('steps_per_env'),
            'declared_git_commit': config.get('git_commit'),
            'declared_commit_note': 'Historical run metadata, not proof of a clean worktree or the full historical source snapshot.',
            'evaluation': completion(summary),
        })
    chair = next(e for e in experiments if e['id'] == 'C-01')
    run_dir = Path(chair['source_run_directory'])
    chair['additional_evaluations'] = []
    for seed in (722, 723):
        source = run_dir / f'evaluation/model12000_seed{seed}/summary.json'
        copy(source, f'C-01/summary_seed{seed}.json')
        chair['additional_evaluations'].append(completion(json.loads(source.read_text())))
    chair['pooled_recorded_completion'] = {'completed_reference': 9, 'episodes': 9,
        'note': '3 evaluations x 3 episodes; deterministic traces were identical, not 9 independent training runs or randomized scenarios.'}
    t04 = next(e for e in experiments if e['id'] == 'T-04')
    copy(Path(t04['source_run_directory']) / 'evaluation_20260917/model12000_seed721/summary.json',
         'T-04/summary_recheck_20260917.json')
    return experiments


def reports() -> None:
    selections = {
        'bucket_AB_20260919.md': 'logs/Core4DBucket/comparison_20260919/RESULTS_CN.md',
        'bucket_ablations_20260919.md': 'logs/Core4DBucket/ablation_comparison_20260919/RESULTS_CN.md',
        'bucket_ablations_final_20260920.md': 'logs/Core4DBucket/ablation_comparison_20260920/RESULTS_CN.md',
        'smalltable_5kg_20260920.md': 'logs/Core4DSmallTableA/comparison_20260920/RESULTS_CN.md',
        'smalltable_5kg_analysis_20260920.json': 'logs/Core4DSmallTableA/comparison_20260920/ANALYSIS.json',
        'smalltable_height_penalty_20260917.md': 'logs/Core4DSmallTable/interaction_mesh_z2_height1_scale005_fresh12000_save2000_actor158_seed721_env2048_20260916/evaluation_20260917/RESULTS_CN.md',
        'chair_training_review.md': 'logs/Core4DChair/chair021_a1_fresh12000_save2000_actor158_seed721_env2048_20260910/evaluation/TRAINING_REVIEW.md',
    }
    for filename, source in selections.items():
        copy(REPO / source, 'reports/' + filename)


def history() -> list[dict]:
    items = []
    configs = {
        'Push-livePD': 'logs/Plan5Push/a1_livepd_20kg_continue7000_from_08050_seed721_env2048/run_config.json',
        'Pull': 'logs/Plan5Pull/pull_20kg_full8000_from_critic50_seed721_env2048/run_config.json',
        'Kick': 'logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/run_config.json',
        'Demo3': 'logs/Demo3Tug/square_table_diagonal_tug_full15000_seed721_env2048/run_config_resume_from_06800.json',
        'Demo4': 'logs/Demo4Rotate/rectangular_pull_pull_rotate90_full10000_save1000_seed721_env2048/run_config.json',
    }
    for name, source in configs.items():
        config_copy(HISTORY / source, f'historical/{name}')
        items.append({'id': name, 'source_config': str(HISTORY / source),
                      'config_extract': f'historical/{name}/config_extract.json'})
    for name, path in {
        'Demo3': 'logs/Demo3Tug/eval_model10100_seed721/evaluation.json',
        'Demo4': 'logs/Demo4Rotate/eval_model10000_seed721/evaluation.json',
    }.items():
        copy(HISTORY / path, f'historical/{name}/evaluation.json')
    source = HISTORY / 'MULTI_AGENT_EMERGENCE_ROADMAP.md'
    raw = source.read_bytes()
    lines = raw.decode().splitlines()
    excerpts = [(1448, 1502), (1650, 1672)]
    output = ['# 历史记录原文摘录', '', f'来源：`{source}`', f'原文件 SHA256：`{sha(raw)}`', '',
              '以下保留原行号；这是当时记录，不是本次重新评测。', '']
    for start, end in excerpts:
        output.extend([f'## 原文件第 {start}–{end} 行', '', '```text'])
        output.extend(f'{i + 1}: {lines[i]}' for i in range(start - 1, min(end, len(lines))))
        output.extend(['```', ''])
    target = DEST / 'historical/rollout_history_excerpts.md'
    target.write_text('\n'.join(output))
    FILES.append({'source_absolute': str(source), 'source_sha256': sha(raw),
                  'copied_relative': str(target.relative_to(DEST)),
                  'copy_sha256': sha(target.read_bytes()), 'copy_bytes': target.stat().st_size,
                  'transformation': 'verbatim_line_numbered_excerpts', 'line_ranges': excerpts})
    # Extract only small structured outcome fields from existing Push logs.
    push_rows = []
    decoder = json.JSONDecoder()
    for iteration in (13050, 14050, 15050):
        directory = HISTORY / f'logs/Plan5Push/eval_livepd_{iteration}_multiseed_20260827'
        for source in sorted(directory.glob('*.log')):
            raw = source.read_bytes()
            text = raw.decode(errors='replace')
            result = None
            for match in re.finditer(r'^\{', text, re.MULTILINE):
                try:
                    value, _ = decoder.raw_decode(text[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and 'object_reference_metrics' in value:
                    result = value
            if result is None:
                MISSING.append(f'No structured outcome decoded from {source}')
                continue
            push_rows.append({'source_absolute': str(source), 'source_sha256': sha(raw),
                'checkpoint_iteration': iteration,
                'fields_copied_verbatim': {key: result.get(key) for key in (
                    'mappo_checkpoint', 'seed', 'rollout_steps', 'object_reference_metrics',
                    'termination_counts', 'first_termination') if key in result}})
    write_json('historical/Push-livePD/evaluation_outcomes_27runs.json', {
        'artifact_kind': 'selected_fields_from_existing_logs_no_re_evaluation',
        'protocol': 'Fixed frame-0 scene, actor mean, 3 seeds x 3 process launches per checkpoint; no scenario randomization.',
        'runs': push_rows,
    })
    for item in items:
        if item['id'] in ('Pull', 'Kick'):
            item['evaluation_evidence'] = 'historical/rollout_history_excerpts.md'
            item['missing'] = 'No standalone summary JSON found in the archived selected rollout directory; existing NPZ and historical record were not replaced by invented summary.'
    return items


def main() -> None:
    experiments = core4d()
    reports()
    historical = history()
    index = {'schema': 'portable_experiment_evidence_v1', 'snapshot_date': '2026-09-25',
             'purpose': 'Mac-side code/material inspection, not a portable simulator installation.',
             'original_logs_modified': False, 'training_or_simulation_started': False,
             'checkpoint_dataset_environment_included': False,
             'core4d_experiments': experiments, 'historical_experiments': historical,
             'copied_sources': FILES, 'missing_requested_files': MISSING}
    write_json('index.json', index)
    checks = []
    for record in FILES:
        target = DEST / record['copied_relative']
        assert sha(target.read_bytes()) == record['copy_sha256'], target
        if record['transformation'] == 'lossless_gzip':
            assert sha(gzip.decompress(target.read_bytes())) == record['source_sha256'], target
        elif record['transformation'] == 'byte_identical':
            assert record['source_sha256'] == record['copy_sha256'], target
    for path in sorted(DEST.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            checks.append(f'{sha(path.read_bytes())}  {path.relative_to(DEST)}')
    (DEST / 'SHA256SUMS').write_text('\n'.join(checks) + '\n')
    print(json.dumps({'core4d_runs': len(experiments), 'historical_runs': len(historical),
                      'files_hashed': len(checks), 'bytes': sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()),
                      'missing': MISSING}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
