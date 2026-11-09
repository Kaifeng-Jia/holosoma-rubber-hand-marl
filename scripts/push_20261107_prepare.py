"""Build isolated research snapshot from verified R2 originals, no installation."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,importlib.metadata as md
ROOT=Path('/root/autodl-tmp/push_seeds_20261107');OLD=Path('/root/autodl-tmp/r2_20261024');OPS=Path('/root/autodl-tmp/push_seeds_20261107_ops')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not ROOT.exists(), 'Refuse overwrite'
manifest=json.loads((OLD/'r2_source_manifest.json').read_text());patches=json.loads(Path('/root/autodl-tmp/r2_20261024_retry1_ops/source_patch_manifest.json').read_text())['files']
checked=[]
for name,record in manifest['files'].items():
 p=OLD/name;expected=patches.get(name,{}).get('new_sha256',record['sha256'])
 assert sha(p)==expected,('Frozen source differs',name)
 checked.append(dict(path=name,sha256=expected))
ROOT.mkdir()
for rec in checked:
 p=ROOT/rec['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/rec['path'],p)
for name in ['r2_source_manifest.json','r2_tracked_diff.patch']:shutil.copy2(OLD/name,ROOT/name)
shutil.copytree(OLD/'.git',ROOT/'.git')
for p in (OPS/'incoming').iterdir():
 if p.suffix=='.py':shutil.copy2(p,ROOT/'scripts'/p.name)
shutil.copytree(OPS/'incoming/baselines',ROOT/'baseline_configs')
shutil.copy2(OPS/'incoming/request.md',ROOT/'request.md')
prior=ROOT/'logs/WholeBodyTracking/marl_compat_a1_v1/model_07999_actor158.pt'
assert sha(prior)=='11ef1fe7a4b340a47218f040e7675af4073067c5ce931e169818f903150e3224'
record=dict(ready=True,source_files=checked,base_commit=manifest['base_git_head'],authority_commit='7df11c92b07c204fc0ca86654669298ec24184ce',prior_sha256=sha(prior),python=sys.version,executable=sys.executable,packages={x:md.version(x) for x in ['torch','isaacsim','numpy','tyro']},code_note='Frozen R2 source copied byte-for-byte including approved reference-audit patch; entry adds config guard before first update only. No algorithm modifications.',original_source=str(OLD))
(OPS/'deployment.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='source_files'},indent=2))
