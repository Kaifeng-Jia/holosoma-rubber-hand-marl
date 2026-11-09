#!/usr/bin/env python3
"""Bounded deployment continuation: T-07 seeds 722/723 only, no automatic repairs."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import traceback

OPS = Path('/root/autodl-tmp/paper_supplement_20260929_ops')
ROOT = Path('/root/autodl-tmp/paper_supplement_20260929')
PY = '/root/autodl-tmp/core4d_deps/hssim/bin/python'
VARIANT = 'bucket_A_no_rel_no_height'
ARCHIVE_SHA = '243732c543e6e0c78f27ce5f9cee6ae01061e7d2df8d6ed3fba4e2ab0d8aac8e'
EXPECTED_CONFLICT = re.compile(
    r'isaacsim-kernel 5\.1\.0\.0 has requirement typing[-_]extensions==4\.12\.2, '
    r'but you have typing[-_]extensions 4\.16\.0\.', re.I)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def save(path, record):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)

def acceptable_pip_check(returncode, output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return ((returncode == 0 and lines == ['No broken requirements found.']) or
            (returncode == 1 and len(lines) == 1 and EXPECTED_CONFLICT.fullmatch(lines[0]) is not None))

def environment(gpu):
    env = os.environ.copy()
    for name in ('JPY_PARENT_PID', 'ISAAC_JUPYTER_KERNEL', 'ISAAC_JUPYTER_PYTHON_PACKAGE',
                 'RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        env.pop(name, None)
    env.update(OMNI_KIT_ACCEPT_EULA='YES', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES=str(gpu), WORLD_SIZE='1', LOCAL_RANK=str(gpu), MPLBACKEND='Agg',
               PYTHONPATH=str(ROOT/'src/holosoma') + ':' + str(ROOT/'src/holosoma_retargeting'))
    return env

def command(output, seed, formal):
    return [PY, '-B', str(ROOT/'scripts/train_core4d_smalltable.py'),
            '--experiment', 'smalltable5kg_A', '--reward-variant', VARIANT,
            '--num-envs', '2048', '--steps-per-env', '24', '--seed', str(seed),
            '--iterations', '12000' if formal else '2',
            '--save-interval', '2000' if formal else '2', '--output-dir', str(output)]

def validate_run(output, seed, iterations):
    config = read(output/'run_config.json')
    expected = dict(experiment='smalltable5kg_A', reward_variant=VARIANT, num_envs=2048,
                    iterations_requested=iterations, steps_per_env=24, seed=seed,
                    save_interval=2000 if iterations == 12000 else 2,
                    initialization='fresh', start_iteration=0, actor_input_dim=158, critic_input_dim=527)
    for key, value in expected.items():
        require(config.get(key) == value, f'Unexpected {key}: {config.get(key)}')
    contract = config['bucket_reward_contract']
    require(contract['block_weights_position_rotation_height_relation'] == [1,1,0,0], 'Reward block mismatch')
    require(contract['position_error_weights_xyz'] == [1,1,2], 'Position reward mismatch')
    rows = [json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    require([r['iteration'] for r in rows] == list(range(1, iterations+1)), 'Incomplete iteration sequence')
    require(finite(rows), 'Non-finite training metrics')
    require(all(r.get('Bucket/gate_mean') == 1 for r in rows), 'Unexpected contact gating')
    status = read(output/'status.json')
    require(status.get('passed') is True and status.get('final_iteration') == iterations, 'Incomplete run status')
    model = output/f'model_{iterations:05d}.pt'
    require(model.is_file() and model.stat().st_size > 0, 'Missing final checkpoint')
    return {'final_checkpoint': str(model), 'sha256': digest(model), 'final_iteration': iterations}

def stop_requested():
    require(not (OPS/'STOP_AUTORUN').exists(), 'STOP_AUTORUN requested: no further launches')

def wait_installer():
    deadline = time.monotonic() + 24*3600
    while not (OPS/'continue_install.exit.json').exists():
        stop_requested()
        require(time.monotonic() < deadline, 'Installer wait exceeded 24 hours')
        time.sleep(20)
    for label in ('install_torch', 'install_other', 'install_isaac', 'install_editables', 'pip_list'):
        require(read(OPS/(label+'.exit.json'))['returncode'] == 0, f'Installation failed: {label}')
    # The successful historical environment has this ONE declaration conflict.
    # Preserve its nonzero status; never exempt other missing/incompatible packages.
    result = read(OPS/'pip_check.exit.json')
    output = (OPS/'pip_check.log').read_text()
    require(acceptable_pip_check(result['returncode'], output), 'Unexpected pip dependency conflicts')
    save(OPS/'autorun_dependency_review.json', {
        'checked_utc': now(), 'pip_check_returncode': result['returncode'], 'raw_output': output,
        'known_historical_conflict_only': result['returncode'] == 1,
        'runtime_preflight_still_required': True})

def verify_source():
    record = read(OPS/'source_verified.json')
    require(record.get('verified') is True and record['archive_sha256'] == ARCHIVE_SHA, 'Source package mismatch')
    manifest = read(OPS/'verified_source/package_manifest.json')
    for name, entry in manifest['files'].items():
        require(digest(ROOT/name) == entry['sha256'], f'Source or asset changed: {name}')
    proof = read(OPS/'t07_source_comparison.json')
    require(proof['deployment_archive_sha256'] == ARCHIVE_SHA and not proof['differences'], 'Historical comparison mismatch')
    require(len(proof['matching_files']) == 411, 'Incomplete T07 source comparison')
    return manifest

def ensure_git_snapshot(manifest):
    # Trainer requires git rev-parse. This is explicitly a deployment snapshot,
    # not a fabricated original commit, and contains NO trained checkpoints.
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    if (ROOT/'.git').exists():
        identity = read(OPS/'deployment_git_identity.json')
        require(identity['archive_sha256'] == ARCHIVE_SHA and identity['snapshot_commit'] == git('rev-parse', 'HEAD'),
                'Existing Git snapshot identity differs')
        require(not git('diff', 'HEAD', '--'), 'Tracked source changed after deployment')
        return
    git('init', '-q')
    names = [name for name in manifest['files'] if name.endswith(('.py', '.toml', '.md', '.sh', '.in'))]
    git('add', '--', *names)
    git('-c', 'user.name=Deployment Snapshot', '-c', 'user.email=deployment@localhost',
        '-c', 'commit.gpgsign=false', 'commit', '-q', '-m',
        'Verified deployment snapshot; upstream base ' + manifest['base_git_head'])
    save(OPS/'deployment_git_identity.json', {'snapshot_commit': git('rev-parse', 'HEAD'),
         'upstream_base_commit': manifest['base_git_head'], 'original_git_history_included': False,
         'archive_sha256': ARCHIVE_SHA, 'no_push': True})

def launch(label, gpu, seed, formal):
    stop_requested()
    output = ROOT/'logs/PaperSupplement20260929'/label
    require(not output.exists(), 'Refusing output reuse: '+str(output))
    require(not (OPS/(label+'.launch.json')).exists(), 'Duplicate launch: '+label)
    argv = command(output, seed, formal)
    stream = (OPS/(label+'.stdout.log')).open('x')
    try:
        proc = subprocess.Popen(argv, cwd=ROOT, env=environment(gpu), stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        stream.close()
    item = {'label': label, 'gpu': gpu, 'seed': seed, 'formal': formal, 'argv': argv,
            'pid': proc.pid, 'started_utc': now(), 'output': str(output),
            'environment_overrides': {'CUDA_VISIBLE_DEVICES': str(gpu), 'WORLD_SIZE': '1', 'LOCAL_RANK': str(gpu)},
            'no_auto_retry': True, 'checkpoint_interval': 2000 if formal else 2}
    save(OPS/(label+'.launch.json'), item)
    return proc, item, time.monotonic()

def finish(job, timeout=False):
    proc, item, _ = job
    if timeout and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    rc = proc.wait()
    result = dict(item, ended_utc=now(), process_returncode=rc, timed_out=timeout, validated=False)
    if rc == 0 and not timeout:
        try:
            result.update(validate_run(Path(item['output']), item['seed'], 12000 if item['formal'] else 2))
            result['validated'] = True
        except Exception as exc:
            result['validation_error'] = repr(exc)
    save(OPS/(item['label']+'.exit.json'), result)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Explicitly arm the bounded two-run queue')
    parser.add_argument('--attempt-id', default='initial', help='Explicit manual retry identity; never automatic')
    parser.add_argument('--formal-suffix', default='', help='New output identity for an explicitly reviewed infrastructure retry')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9_-]{1,32}', args.attempt_id) is not None, 'Invalid attempt identity')
    require(not args.formal_suffix or re.fullmatch(r'[a-z0-9_-]{1,32}', args.formal_suffix) is not None, 'Invalid formal suffix')
    formal_suffix = '_'+args.formal_suffix if args.formal_suffix else ''
    if not args.execute:
        print(json.dumps({'runs': [{'gpu': 0, 'seed': 722}, {'gpu': 1, 'seed': 723}],
                          'iterations': 12000, 'envs': 2048, 'steps': 24, 'save_interval': 2000,
                          'experiment': 'smalltable5kg_A', 'reward_variant': VARIANT}, indent=2))
        return
    with (OPS/'autorun.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = OPS/('autorun.started.json' if args.attempt_id == 'initial' else f'autorun.{args.attempt_id}.started.json')
        require(not started.exists(), 'This autorun attempt already exists; manual review required')
        require(not any((OPS/f'T07_seed{s}_fresh12000{formal_suffix}.launch.json').exists() for s in (722,723)),
                'Formal run already attempted; refuse automatic duplicate')
        if (OPS/'autorun.status.json').exists():
            save(OPS/f'autorun.before_{args.attempt_id}.status.json', read(OPS/'autorun.status.json'))
        save(started, {'utc': now(), 'pid': os.getpid(), 'scope': 'T07 seeds722/723 only', 'attempt_id': args.attempt_id})
        try:
            save(OPS/'autorun.status.json', {'state': 'waiting_for_installation', 'utc': now()})
            wait_installer()
            manifest = verify_source()
            require(shutil.disk_usage(ROOT).free > 10*1024**3, 'Less than 10 GiB available')
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
            require(not active.strip(), 'GPUs already have compute jobs; do not interfere')
            ensure_git_snapshot(manifest)
            save(OPS/'autorun.status.json', {'state': 'preflight', 'utc': now()})
            for gpu, seed in ((0,722), (1,723)):
                suffix = '' if args.attempt_id == 'initial' else '_'+args.attempt_id
                job = launch(f'T07_seed{seed}_preflight2{suffix}', gpu, seed, False)
                try:
                    job[0].wait(timeout=45*60)
                    result = finish(job)
                except subprocess.TimeoutExpired:
                    result = finish(job, timeout=True)
                require(result['validated'], f'Preflight failed on GPU{gpu}; see exit record')
            verify_source()
            jobs = []
            try:
                for gpu, seed in ((0,722), (1,723)):
                    jobs.append(launch(f'T07_seed{seed}_fresh12000{formal_suffix}', gpu, seed, True))
            except Exception:
                # A launched job remains supervised even if the second launch fails.
                for job in jobs:
                    finish(job, timeout=True)
                raise
            save(OPS/'autorun.status.json', {'state': 'training', 'utc': now(), 'pids': [j[0].pid for j in jobs]})
            results = []
            while jobs:
                for job in jobs[:]:
                    timed_out = time.monotonic()-job[2] > 48*3600
                    if job[0].poll() is not None or timed_out:
                        results.append(finish(job, timeout=timed_out))
                        jobs.remove(job)
                if jobs:
                    time.sleep(20)
            require(all(r['validated'] for r in results), 'One or more training jobs failed; no automatic retry')
            save(OPS/'autorun.status.json', {'state': 'complete', 'utc': now(), 'runs': results})
        except Exception as exc:
            save(OPS/'autorun.status.json', {'state': 'blocked', 'utc': now(), 'error': repr(exc)})
            traceback.print_exc()
            raise

if __name__ == '__main__':
    main()
