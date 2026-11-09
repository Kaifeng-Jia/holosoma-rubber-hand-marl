#!/usr/bin/env python3
"""Organize existing training/evaluation evidence; never train or run physics."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from paper_evaluation_utils import sha, write_json, evaluation_result_dir
from summarize_paper_evaluation import summarize, save_csv

ROOT = Path(__file__).resolve().parents[1]


def training_evidence(root, models):
    target = root / 'training_evidence'
    target.mkdir(exist_ok=True)
    historical = json.loads((ROOT / 'logs/PaperSupplement20260929/audit_initial/run_manifest.json').read_text())['historical_candidates']
    runs = ROOT / 'logs/PaperSupplement20260929/autodl_received/runs'
    inventory = []
    for m in models:
        seed = m['training_seed']
        if seed == 721:
            candidate = next(c for c in historical if c['configuration'] == m['configuration'])
            stages = [Path(s['directory']) / 'run_config.json' for s in candidate['stages']]
        elif m['case'] in ('push', 'pull'):
            stages = sorted(runs.glob(f'{m["case"].title()}_seed{seed}_stage*/run_config.json'))
        else:
            stages = [Path(m['config'])]
            while True:
                config = json.loads(stages[0].read_text())
                resumed = config.get('resume_checkpoint')
                if not resumed:
                    break
                previous = runs / Path(resumed).parent.name / 'run_config.json'
                if not previous.exists() or previous in stages:
                    raise ValueError(f'Cannot resolve training chain for {m["model_id"]}: {previous}')
                stages.insert(0, previous)
        budget = 13050 if m['case'] == 'push' else 8050 if m['case'] == 'pull' else 12000
        rows = []
        records = []
        directory = target / m['model_id']
        directory.mkdir(exist_ok=True)
        for index, config_path in enumerate(stages):
            c = json.loads(config_path.read_text())
            start = c.get('start_iteration', 0)
            end = min(budget, json.loads(stages[index + 1].read_text()).get('start_iteration', budget) if index + 1 < len(stages) else budget)
            shutil.copyfile(config_path, directory / f'stage{index}_run_config.json')
            metrics_path = config_path.with_name('metrics.jsonl')
            accepted = []
            if metrics_path.exists():
                for line in metrics_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    if start < raw['iteration'] <= end:
                        accepted.append(dict(source_stage=index, **{k: v for k, v in raw.items() if isinstance(v, (int, float))}))
            rows.extend(accepted)
            records.append(dict(stage=index, original_config=str(config_path), config_sha256=sha(config_path),
                                metrics_path=str(metrics_path), metrics_sha256=sha(metrics_path) if metrics_path.exists() else None,
                                included_iteration_interval=[start + 1, end], recorded_rows=len(accepted),
                                initialization=c.get('initialization', c.get('actor_update_mode')),
                                resume_checkpoint=c.get('resume_checkpoint'), git_commit=c.get('git_commit')))
        save_csv(directory / 'curve.csv', rows)
        write_json(directory / 'stages.json', records)
        inventory.append(dict(model_id=m['model_id'], checkpoint_sha256=m['checkpoint_sha256'],
                              reuse_status=m['reuse_status'], selected_budget=budget, stages=records,
                              note='Discard rollback-tail updates after a resumed checkpoint. Historical process exit may be unknown. Training reward is not an evaluation success metric.'))
    write_json(target / 'index.json', inventory)
    # Individual curves only: different reward versions do not share a comparable return scale.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for m in models:
        directory = target / m['model_id']
        path = directory / 'curve.csv'
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        for ax, key in zip(axes, ('reward_mean', 'last_step_object_position_error_mean_m')):
            available = [r for r in rows if r.get(key)]
            ax.plot([int(r['iteration']) for r in available], [float(r[key]) for r in available], linewidth=.5)
            ax.set_ylabel(key.replace('_', ' '), fontsize=8)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel('PPO iteration (selected optimization chain)')
        fig.suptitle(m['model_id'] + ' | training diagnostics, not evaluation')
        fig.tight_layout(); fig.savefig(directory / 'curve.png', dpi=130); plt.close(fig)


def latex(root):
    for name in ('main_results', 'reward_comparison'):
        path = root / (name + '.csv')
        if not path.exists():
            continue
        lines = [r'\begin{tabular}{lccc}', r'\hline', r'Configuration & Completion (\%) & Progress & Object RMSE (m) \\', r'\hline']
        for r in csv.DictReader(path.open()):
            values = []
            for metric, scale in (('completion_rate', 100), ('progress_mean', 1), ('completed_object_rmse_3d_m', 1)):
                mean, sd = r.get(metric + '_mean'), r.get(metric + '_sample_sd')
                value = f'${float(mean)*scale:.3f} \\pm {float(sd)*scale:.3f}$' if mean and sd and r['complete_three_seed_result'] == 'True' else '--'
                values.append(value)
            lines.append(r['configuration'].replace('_', r'\_') + ' & ' + ' & '.join(values) + r' \\')
        lines += [r'\hline', r'\end{tabular}', '% Missing or incomplete three-seed results are --. RMSE uses completed episodes only.']
        (root / (name + '.tex')).write_text('\n'.join(lines) + '\n')


def report(root):
    models = json.loads((root / 'models.json').read_text())
    finished = [m for m in models if (root / (m['model_id']+'.exit.json')).exists()
                and json.loads((root / (m['model_id']+'.exit.json')).read_text()).get('validated')]
    pending = [m['model_id'] for m in models if m['reuse_status'] == 'source_audit_pending']
    status = json.loads((root / 'batch_status.json').read_text()) if (root / 'batch_status.json').exists() else {}
    text = ['# 统一评测结果（动态整理）', '', '更新UTC：' + dt.datetime.now(dt.timezone.utc).isoformat(), '',
            f'14个新增训练已完成并本地校验；评测完整输出 {len(finished)}/21 个模型身份。',
            '当前队列状态：' + status.get('state', 'unknown') + '；当前模型：' + status.get('current_model', '无') + '。', '',
            '## 已完成单模型结果', '', '| 模型 | 来源资格 | 扰动完成 | 平均进度 | 完成子集3D RMSE(m) |', '|---|---|---:|---:|---:|']
    for m in finished:
        s = json.loads((evaluation_result_dir(root,m) / 'summary.json').read_text())
        rs = [r for r in s['episodes'] if r['kind'] == 'perturbed']
        good = [r for r in rs if r['completed_reference']]
        error = f'{sum(r["object_rmse_3d_m"] for r in good)/len(good):.5f}' if good else 'NA'
        text.append(f'| {m["model_id"]} | {m["reuse_status"]} | {len(good)}/{len(rs)} | {sum(r["progress"] for r in rs)/len(rs):.4f} | {error} |')
    text += ['', '## 统计与证据说明', '',
             '- 主表使用每seed50扰动回合的汇总，再计算独立训练seed间样本SD；标准初态另外列出。',
             '- 未通过来源核查的旧模型仍可保存回放，但不计入正式均值/SD；不虚构第三个训练seed。',
             '- 完成率受冻结的跟踪终止条件约束，不等于双人均匀承重或严格用手完成。',
             '- 完成子集RMSE与失败前缀误差分列；不能因失败轨迹短、误差低就认为更好。',
             '- 桶/小桌对照同时添加独立高度与相对关系，不据此拆分两项单独的因果作用。',
             '- T-07及桶A新seed有欠费中断恢复；恢复优化器/模型，不恢复全部仿真和RNG状态，非逐位续跑。', '',
             '待来源核查：' + (', '.join(pending) if pending else '无') + '。', '',
             '全部性能判断以完整表格为准；评测完成与历史来源资格分别记录，不据此预先宣布某奖励普遍胜出。']
    (root / 'REPORT_CN.md').write_text('\n'.join(text) + '\n')
    write_json(root / 'run_manifest.json', dict(
        version='paper_supplement_unified_evaluation_v1', research_commit='2b6668b1205ee5c5b81ae9251744b16c799c7173',
        new_training_models=14, target_model_identities=21, validated_evaluations=len(finished),
        models=models, pending_historical_audits=pending,
        initial_states_sha256={p.name: sha(p) for p in root.glob('*_states.json')},
        evaluation_scripts_sha256={name: sha(ROOT/'scripts'/name) for name in
            ('evaluate_paper_supplement.py','paper_evaluation_utils.py','summarize_paper_evaluation.py','run_paper_evaluation_batch.py','organize_paper_evaluation.py')},
        note='New training completion is supported by received operations exit/metric validation; old source eligibility is independent. Evaluations require actual exit0 and51 unique episodes.'))
    latex(root)


def media(root, models):
    import imageio_ffmpeg
    media_dir = root / 'media'
    media_dir.mkdir(exist_ok=True)
    index = []
    for m in models:
        if m['training_seed'] != 721:
            continue
        src = evaluation_result_dir(root,m)
        if not (src / 'summary.json').exists():
            index.append(dict(model_id=m['model_id'], state='no_complete_evaluation')); continue
        manifest = json.loads((src / 'run_manifest.json').read_text())
        object_path = ROOT / 'src/holosoma' / manifest['object_urdf']
        dest = media_dir / m['model_id']
        record = dict(model_id=m['model_id'], rule='preselected seed721 nominal, including failure',
                      reuse_status=m['reuse_status'], rollout=str(src / 'nominal_viser.npz'))
        if (dest / 'manifest.json').exists():
            record['state'] = 'exported'; index.append(record); continue
        if dest.exists():
            record['state'] = 'partial_export_retained'; index.append(record); continue
        argv = ['/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python', '-B',
                str(ROOT / 'scripts/export_recorded_demo_media.py'), '--rollout', str(src / 'nominal_viser.npz'),
                '--object-urdf', str(object_path), '--output', str(dest), '--title', m['configuration'],
                '--checkpoint-label', 'seed721 nominal | ' + m['reuse_status'], '--ffmpeg', imageio_ffmpeg.get_ffmpeg_exe()]
        with (media_dir / (m['model_id'] + '.log')).open('w') as log:
            rc = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT).returncode
        record.update(process_returncode=rc, state='exported' if rc == 0 and (dest / 'manifest.json').exists() else 'export_failed')
        index.append(record)
    write_json(media_dir / 'index.json', index)


def portable(root, package_name='PAPER_SUPPLEMENT_EVALUATION.tar.gz'):
    if Path(package_name).name != package_name or not package_name.endswith('.tar.gz'):
        raise ValueError('Package name must be a local .tar.gz basename')
    archive = root / package_name
    if archive.exists():
        raise FileExistsError(f'Preserve delivered archives; choose a new package name: {archive}')
    dest = root / 'portable'
    dest.mkdir(exist_ok=True)
    for name in ('REPORT_CN.md', 'run_manifest.json', 'models.json', 'episodes.csv', 'per_seed_summary.csv', 'nominal_results.csv',
                 'main_results.csv', 'reward_comparison.csv', 'main_results.tex', 'reward_comparison.tex', 'aggregation_status.json'):
        if (root / name).exists(): shutil.copyfile(root / name, dest / name)
    shutil.copyfile(ROOT / 'docs/experiments/unified_evaluation_20261003.md', dest / 'evaluation_protocol.md')
    review = ROOT / 'docs/experiments/unified_results_review_20261004.md'
    if review.exists():
        shutil.copyfile(review, dest / 'RESULTS_REVIEW_CN.md')
    for folder in ('source_audit', 'training_evidence', 'media'):
        if (root / folder).exists(): shutil.copytree(root / folder, dest / folder, dirs_exist_ok=True)
    for file in root.glob('*_states.json'): shutil.copyfile(file, dest / file.name)
    exits = dest / 'evaluation_exit_records'; exits.mkdir(exist_ok=True)
    for file in root.glob('*.exit.json'): shutil.copyfile(file, exits / file.name)
    for file in root.glob('*.interruption.json'): shutil.copyfile(file, exits / file.name)
    manifests = dest / 'evaluation_manifests'; manifests.mkdir(exist_ok=True)
    for file in (root/'results').glob('*/run_manifest.json'):
        shutil.copyfile(file, manifests / (file.parent.name+'.json'))
    operations = ROOT / 'logs/PaperSupplement20260929/autodl_received/operations'
    training_records = dest / 'training_process_records'; training_records.mkdir(exist_ok=True)
    for pattern in ('*.exit.json', '*.launch.json', '*.status.json'):
        for file in operations.glob(pattern):
            shutil.copyfile(file, training_records / file.name)
    provenance = dest / 'scripts'; provenance.mkdir(exist_ok=True)
    for name in ('evaluate_paper_supplement.py', 'paper_evaluation_utils.py', 'summarize_paper_evaluation.py',
                 'run_paper_evaluation_batch.py', 'organize_paper_evaluation.py', 'export_recorded_demo_media.py'):
        shutil.copyfile(ROOT / 'scripts' / name, provenance / name)
    files = sorted(p for p in dest.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (dest / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(dest)}\n' for p in files))
    with tarfile.open(archive, 'w:gz') as tf: tf.add(dest, arcname='paper_supplement_evaluation')
    archive.with_suffix(archive.suffix + '.sha256').write_text(sha(archive) + '  ' + archive.name + '\n')
    return archive


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--watch', action='store_true', help='Wait for existing queue; no new evaluation or training')
    p.add_argument('--export-media', action='store_true')
    p.add_argument('--package-name', default='PAPER_SUPPLEMENT_EVALUATION.tar.gz',
                   help='Use a new filename to preserve previously delivered archives')
    a = p.parse_args(); root = a.root.resolve()
    if Path(a.package_name).name != a.package_name or not a.package_name.endswith('.tar.gz'):
        p.error('--package-name must be a local .tar.gz basename')
    if (root / a.package_name).exists():
        p.error('Package already exists; choose a new --package-name without overwriting it')
    models = json.loads((root / 'models.json').read_text())
    training_evidence(root, models)
    while True:
        report(root)
        status = json.loads((root / 'batch_status.json').read_text())
        write_json(root / 'organization_status.json', dict(state='waiting_for_evaluation', pid=os.getpid(), updated_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        if not a.watch or status.get('state') in ('complete', 'needs_review'): break
        time.sleep(30)
    # Running queues own their CSV outputs. Recompute only after the queue ends.
    if status.get('state') in ('complete', 'needs_review'):
        # Load the final aggregation implementation from disk after a long wait.
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/summarize_paper_evaluation.py'), '--root', str(root)], check=True)
        report(root)
        models = json.loads((root / 'models.json').read_text())
        if a.export_media: media(root, models)
        archive = portable(root, a.package_name)
        write_json(root / 'organization_status.json', dict(state='organized', batch_state=status['state'],
                   audit_pending=[m['model_id'] for m in models if m['reuse_status'] == 'source_audit_pending'],
                   package=str(archive), ended_utc=dt.datetime.now(dt.timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
