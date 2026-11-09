#!/usr/bin/env python3
"""Pull only this bounded experiment's logs/models; never delete either side."""
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'logs/PaperSupplement20260929/autodl_received'
SSH = 'ssh -i /home/kevin/.ssh/id_ed25519_core4d_cloud -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -p 44248'
HOST = 'root@connect.nmb1.seetacloud.com'
SOURCES = {
    'operations': '/root/autodl-tmp/paper_supplement_20260929_ops/',
    'runs': '/root/autodl-tmp/paper_supplement_20260929/logs/PaperSupplement20260929/',
}

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST/'backup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 120*3600
        while time.monotonic() < deadline:
            record = {'utc':dt.datetime.now(dt.timezone.utc).isoformat(), 'sources':{}}
            for label, source in SOURCES.items():
                target = DEST/label
                target.mkdir(exist_ok=True)
                argv = ['rsync', '-az', '--timeout=120', '--partial-dir=.rsync-partial',
                        '--include=*/', '--include=*.json', '--include=*.jsonl',
                        '--include=*.log', '--include=*.pt', '--include=*.sha256', '--exclude=*',
                        '-e', SSH, HOST+':'+source, str(target)+'/']
                try:
                    result = subprocess.run(argv, capture_output=True, text=True, timeout=600)
                    record['sources'][label] = {'returncode':result.returncode, 'stderr':result.stderr[-3000:]}
                except subprocess.TimeoutExpired:
                    record['sources'][label] = {'returncode':None, 'error':'transfer timeout; retry next cycle'}
            with (DEST/'backup_history.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            # Include the approved staged Push/Pull batch before final shutdown.
            statuses = [DEST/'operations'/name for name in (
                'resume_t07_20260930.status.json', 'chair_pair_20260930.status.json',
                'bucket_rz0_pair_20260930.status.json', 'plan5_batch_20261001.status.json',
                'bucket_a_resume_20261003.status.json', 'table_a_pair_20261003.status.json')]
            if all(path.exists() for path in statuses):
                states = [json.loads(path.read_text()).get('state') for path in statuses]
                if (all(state in ('complete','blocked') for state in states)
                        and all(v['returncode']==0 for v in record['sources'].values())):
                    print('Final sync complete; remote states:',states,flush=True)
                    return
            if (DEST/'STOP_BACKUP').exists():
                return
            time.sleep(600)

if __name__ == '__main__':
    main()
