"""Preserve completed revision 1.14 evidence and build one verified local package."""
import csv, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
import numpy as np
R=Path('/home/kevin/holosoma-core4d-base')
O=R/'paper_handoff_exports/V3_20261102/V3_CLOSEOUT_SUPPLEMENT_20261105'
B=R/'logs/V3_20261102/cloud_backup_20261104/raw'
OLD=R/'paper_handoff_exports/V3_20261102/V3_GPU_CLOSEOUT_20261104.tar.gz'
EXPECTED_OLD='76271007fd5ec78484e2140b5d614e951e4b6dc0de235cda714ec1e08ce3ebba'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def ident(p):return dict(path=str(p),sha256=sha(p))
def patch_file(p,content):
 assert not p.exists(),p
 patch='*** Begin Patch\n*** Add File: '+str(p)+'\n'+''.join('+'+x+'\n' for x in content.splitlines())+'*** End Patch\n'
 subprocess.run(['apply_patch'],input=patch,text=True,check=True,capture_output=True)
def preserve():
 # Exact executed files remain immutable provenance; installed finalizer only
 # replaces the temporary import search directory with its own directory.
 for name in ['v314_compute.py','v314_render.py','v314_finalize.py','v314_package.py']:
  p=Path('/tmp')/name;shutil.copy2(p,O/'sources'/name)
  content=p.read_text()
  if name=='v314_finalize.py':content=content.replace("'/tmp',str(R/'scripts')","str(Path(__file__).resolve().parent),str(R/'scripts')")
  dest=R/'scripts'/name
  if not dest.exists():patch_file(dest,content)
  else:assert dest.read_text()==content
 for p in [B/'v3_palm_postprocess.py',R/'paper_handoff_exports/FIGURES_R2_20261027_approved/sources/figround_common.py',R/'paper_handoff_exports/FIGURES_R2_20261027_approved/sources/figround_r1.py',R/'scripts/reference_geometry_math.py']:
  shutil.copy2(p,O/'sources'/p.name)
 records=read(O/'sources/evidence_index.json');known={x['original_path'] for x in records}
 def evidence(p):
  p=Path(p)
  if str(p) not in known:
   h=sha(p);dest=O/'sources/evidence'/(h[:12]+'_'+p.name);shutil.copy2(p,dest);records.append(dict(original_path=str(p),sha256=h,package_path=str(dest.relative_to(O))));known.add(str(p))
 t=read(O/'reference/T07_wrist_audit.json')
 for d in [t['reference'],t['target'],*t['nominal_sources']]:evidence(d['path'])
 models=read(R/'logs/PaperSupplement20260929/unified_evaluation_20261003/models.json');cfgs=[]
 for row in models:
  if row['configuration']!='T-07':continue
  p=Path(row['config']);cfg=read(p);assert cfg['runtime_reference_sha256']==t['T07_5kg_runtime']['sha256']
  evidence(p);cfgs.append(dict(model_id=row['model_id'],config=ident(p),runtime_reference_sha256=cfg['runtime_reference_sha256'],recorded_runtime_reference=cfg['runtime_reference']))
 assert len(cfgs)==3
 t['actual_training_configs_verified']=cfgs;write(O/'reference/T07_wrist_audit.json',t)
 kick=Path('/home/kevin/holosoma-rubber-hand-largetable/logs/WholeBodyTracking/20260807_034545-rubberhand_kick_sub16_028_8000_seed42-eval/kick_7999_seed42_physics_eval_v1.npz')
 z=np.load(kick);phase=int(z['motion_time_step'][2418]);assert phase==1
 evidence(R/'src/holosoma_retargeting/holosoma_retargeting/dual_kick_reference.py')
 mapping=dict(task='kick',raw_record=ident(kick),selected_attempt=9,selected_record_start=2418,first_recorded_motion_time_step=phase,control_hz=50,human_fps=30,first_source_time_seconds=phase/50,first_source_frame=phase/50*30,definition='Source controller reference phase at first saved episode row. Physics robot state is post-step; no claim that source human pose equals physical robot pose.',generator=ident(R/'src/holosoma_retargeting/holosoma_retargeting/dual_kick_reference.py'))
 assert mapping['first_source_frame']==read(O/'fig3/F3_human_kick.json')['source_frame']
 write(O/'fig3/kick_source_time_audit.json',mapping)
 # Preserve exact code used for the common curve contract and target calibration.
 for p in [B/'v3_20261103_pull/src/holosoma/holosoma/utils/v3_episode_telemetry.py',R/'src/holosoma_retargeting/holosoma_retargeting/src/utils.py',R/'src/holosoma_retargeting/holosoma_retargeting/dual_pull_reference.py']:
  evidence(p)
 write(O/'sources/evidence_index.json',records)
 write(O/'sources/code_provenance.json',dict(authority_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),executed_sources=[dict(file=str(p.relative_to(O)),sha256=sha(p)) for p in sorted((O/'sources').glob('*.py'))],installed_sources=[ident(R/'scripts'/x) for x in ['v314_compute.py','v314_render.py','v314_finalize.py','v314_package.py']],note='Executed source snapshots preserved byte-for-byte. Installed finalizer replaces /tmp import path with its own directory. Absolute research input paths require original local assets; package is evidence, not a standalone environment.'))
def package():
 assert sha(OLD)==EXPECTED_OLD,'old frozen archive changed'
 rows=lambda p:list(csv.DictReader(Path(p).open()))
 groups=rows(O/'metrics/palm_group_summary.csv');seeds=rows(O/'metrics/palm_per_seed.csv');episodes=rows(O/'metrics/palm_episodes.csv')
 assert len(groups)==15 and len(seeds)==45
 assert all(int(x['n_episodes'])==50 for x in seeds)
 assert len(rows(O/'results/episodes.csv'))==6120
 assert len(rows(O/'curves/pull_training_scalars.csv'))==72450
 for x in groups:
  assert x['status']=='complete' or (x['variant']=='copied-single-prior' and x['palm_mean_deg']==x['palm_sample_std_deg']=='')
 for x in episodes:
  if x['palm_status']=='complete':assert x['completed_reference']=='True' and np.isfinite(float(x['palm_error_deg']))
  else:assert x['palm_error_deg']==''
 for x in read(O/'sources/evidence_index.json'):assert sha(O/x['package_path'])==x['sha256']
 for p in (O/'checks').glob('*.json'):
  d=read(p)
  if 'passed' in d:assert d['passed'],p
 assert read(O/'checks/render_validation.json')['count']==14
 sources={x['original_path']:x for x in read(O/'sources/evidence_index.json')}
 for row in rows(O/'reference/Table_III_reference.csv'):assert sources[row['source_file']]['sha256']==row['source_sha256']
 forbidden=[p for p in O.rglob('*') if p.is_file() and (p.suffix in ['.pt','.mp4','.webm','.pyc'] or '__pycache__' in p.parts)]
 assert not forbidden,forbidden
 checklist=[dict(item=i,status='complete',missing='copied-single-prior Palm: no completed episodes; intentionally missing' if i==1 else '') for i in range(1,7)]
 write(O/'checks/completion_manifest.json',dict(passed=True,revision='1.14',checklist=checklist,image_count=14,palm_group_rows=len(groups),palm_seed_rows=len(seeds),palm_completed_episode_count=sum(x['palm_status']=='complete' for x in episodes),tableIII_rows=len(rows(O/'reference/Table_III_reference.csv')),curve_rows=72450,old_package_sha256=EXPECTED_OLD,old_package_unchanged=True,new_training=False,new_rollouts=False,new_videos=False,cloud_accessed=False))
 manifest=O/'SHA256SUMS';files=sorted(p for p in O.rglob('*') if p.is_file() and p!=manifest)
 manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(O)}\n' for p in files))
 tar=O.with_suffix('.tar.gz');assert not tar.exists(),tar
 with tarfile.open(tar,'w:gz',compresslevel=6) as f:f.add(O,arcname=O.name)
 expected={str(p.relative_to(O)):sha(p) for p in files+[manifest]};actual={}
 with tarfile.open(tar,'r:gz') as f:
  for member in f:
   if not member.isfile():continue
   rel=str(Path(member.name).relative_to(O.name));h=hashlib.sha256()
   with f.extractfile(member) as stream:
    for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
   actual[rel]=h.hexdigest()
 assert expected==actual
 digest=sha(tar);tar.with_suffix(tar.suffix+'.sha256').write_text(f'{digest}  {tar.name}\n')
 write(tar.with_suffix(tar.suffix+'.verification.json'),dict(passed=True,archive=ident(tar),size_bytes=tar.stat().st_size,verified_files=len(actual),all_archive_members_hash_match=True,old_package_unchanged=True))
 print(json.dumps(dict(archive=str(tar),sha256=digest,size_bytes=tar.stat().st_size,verified_files=len(actual)),indent=2))
if __name__=='__main__':
 import sys
 if sys.argv[1:]==['preserve']:preserve()
 elif sys.argv[1:]==['package']:package()
 else:raise SystemExit('Use preserve, then author README, then package')
