#!/usr/bin/env python3
"""Package partial CPU evidence, separate old rollout archive, verify extraction."""
import argparse
import csv
import io
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from prepare_experiments_r2_cpu import ROOT,EVAL,GEOM,PKG,ROBOT,read,write,identity,sha
from paper_evaluation_utils import make_states,evaluation_result_dir


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    assert (out/'checks/wrist_recomputed.json').exists()
    archive=out.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    copied={};mapping=[]
    def copy(path, dest=None):
        path=Path(path).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        if str(path) in copied:return copied[str(path)]
        digest=sha(path)
        if dest is None:dest=Path('sources/inputs')/(digest[:12]+'_'+path.name)
        dest=out/dest;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        assert sha(dest)==digest
        copied[str(path)]=str(dest.relative_to(out))
        mapping.append(dict(original_path=str(path),package_path=str(dest.relative_to(out)),sha256=digest))
        return copied[str(path)]
    scripts=['prepare_experiments_r2_cpu.py','collect_experiments_r2_geometry.py','recompute_r2_wrist_cpu.py',
             'package_experiments_r2_cpu.py','paper_evaluation_utils.py','reference_geometry_math.py',
             'reference_geometry_round2_correspondence.py','evaluate_paper_supplement.py','train_plan5_push.py']
    new=['prepare_experiments_r2_cpu.py','collect_experiments_r2_geometry.py','recompute_r2_wrist_cpu.py','package_experiments_r2_cpu.py']
    for name in scripts:copy(ROOT/'scripts'/name,Path('sources/scripts')/name)
    copy(ROOT/'tests/test_experiments_r2_cpu.py',Path('sources/tests/test_experiments_r2_cpu.py'))
    diff=''
    for rel in ['scripts/'+n for n in new]+['tests/test_experiments_r2_cpu.py']:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',rel],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode==1;diff+=result.stdout
    (out/'sources/code_diff.patch').write_text(diff)
    # Source snapshots allow exact function inspection; do not merge branches.
    module_files=[
      'config_values/robot.py','config_values/simulator.py','config_values/marl/g1/command.py',
      'config_values/marl/g1/termination.py','config_values/marl/g1/reward.py','config_values/marl/g1/kick_command.py',
      'config_values/marl/g1/kick_experiment.py','config_values/marl/g1/core4d_smalltable_command.py',
      'config_values/marl/g1/core4d_smalltable_termination.py','config_values/marl/g1/core4d_smalltable_reward.py',
      'config_values/marl/g1/core4d_pair_experiments.py','config_values/marl/g1/core4d_bucket_contract.py',
      'managers/command/terms/wbt.py','managers/command/terms/marl.py','managers/command/terms/core4d_smalltable.py',
      'managers/termination/terms/marl.py','managers/termination/terms/core4d_smalltable.py','managers/reward/terms/marl.py',
      'envs/marl/paired_a1_reference.py','envs/marl/paired_motion_reference.py','utils/rotations.py']
    for rel in module_files:copy(PKG/'holosoma'/rel,Path('sources/holosoma')/rel)
    for rel in ['data_utils/core4d_adapter.py','data_utils/core4d_retarget.py','src/utils.py','src/interaction_mesh_retargeter.py',
                'config_types/data_type.py','config_types/robot.py','dual_a1_layout.py','dual_kick_reference.py']:
        copy(ROOT/'src/holosoma_retargeting/holosoma_retargeting'/rel,Path('sources/holosoma_retargeting')/rel)
    copy(ROBOT,Path('sources/robots/main_mesh_collision_rubberhand.urdf'))
    request=subprocess.check_output(['git','show','ab915c56:docs/experiments/ubuntu_handoff_experiments_r2_20261024_zh.md'],cwd=ROOT)
    (out/'sources/REQUEST.md').write_bytes(request)
    for case in ['push','pull']:
        for base,names in [(GEOM,['selection.json','correspondence_spec.json','input_manifest.json','new_pair_reference.npz','new_runtime.npz']),
                           (ROOT/'logs/ReferenceGeometry/reference_geometry_pilot_20261017',['preprocessed_source.npz','source_interactions.json','source_thresholds.json'])]:
            for name in names:
                path=base/case/name
                if not path.exists() and name=='source_thresholds.json':path=base/name
                if path.exists():copy(path,Path('sources/reference_geometry')/base.name/case/name)
        m=read(GEOM/case/'input_manifest.json')
        for key in ['source','source_object','target_object']:copy(m[key]['path'])
    for scene in read(out/'checks/shared_scene_sources.json'):
        for key in ['source_manifest','original_manifest']:copy(scene[key]['path'])
        e=scene['experiment']
        for k in ['runtime_reference_file','object_urdf_file','training_promotion_file']:copy(PKG/e[k])
    wrist=read(out/'checks/wrist_recomputed.json')
    for record in wrist['records']:
        for key in ['before','after','canonical','manifest','original_retarget_scene']:copy(record[key]['path'])
    for x in read(out/'checks/per_model_settings.json'):copy(x['config']['path'])
    preflight=read(out/'preflight_push_registered.json')
    for stage in preflight['stages']:copy(stage['evidence']['path'])
    kick=read(out/'checks/kick_training_inventory.json');copy(kick['source']['path'])
    kgeom=read(out/'checks/kick_registration_equivalence.json')
    for key in ['source','runtime','viser']:copy(kgeom[key]['path'])
    alignment=read(out/'checks/fk_alignment_check.json')
    for m in alignment['models']:copy(m['reference']['path'])
    copy(EVAL/'models.json',Path('sources/models.json'))
    for name in ['source_audit/review_20261004/review.json','source_audit/historical_reuse.json']:
        copy(EVAL/name,Path('sources')/name)
    # Fixed Kick state list is generated once, no GPU. New eval case still needs integration.
    write(out/'U6_kick/kick_states.json',make_states('kick'))
    write(out/'checks/cloud_training_plan.json',dict(status='waiting_for_user_cloud_gpu_notice',training_started=False,
        U5=dict(configuration='Push-registered',seeds=[721,722,723],stages=[[0,50],[50,8000],[8050,5000]],
            source_checkpoint=preflight['source_checkpoint'],spacing_m=preflight['lateral_spacing_m'],
            num_envs=2048,steps_per_env=24,mass_kg=20,actor_lr=1e-5,critic_lr=.001,
            remaining='Wire same spacing into all train/eval hooks with opt-in default preservation; cloud smoke after availability'),
        U6=dict(configuration='Kick-baseline',new_seeds=[722,723],existing_seed=721,stages=[[0,8000]],
            source_checkpoint=kick['initialization'],num_envs=2048,steps_per_env=24,actor_lr=.001,critic_lr=.001,
            actual_reference='legacy mirrored, checked geometry-pose equivalent under declared correspondence',
            states=identity(out/'U6_kick/kick_states.json'),remaining='Add Kick eval case and cloud smoke; no reselection of seed721'),
        priority='U5 then U6',deadline='2026-10-31',risk_review='2026-10-29',ETA='unknown until cloud availability and measured throughput'))
    # Independent tests and explicit process return code, not inferred from status flags.
    tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_experiments_r2_cpu.py','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'checks/tests.txt').write_text(tests.stdout+tests.stderr+'\nProcess exit code: '+str(tests.returncode)+'\n')
    assert tests.returncode==0
    deps={}
    for name in ['numpy','scipy','torch','yourdfpy','trimesh']:
        try:deps[name]=version(name)
        except PackageNotFoundError:deps[name]='not installed in packaging interpreter'
    write(out/'checks/package_environment.json',dict(python=sys.executable,version=sys.version,platform=platform.platform(),packages=deps))
    # Separate old episodes data pack, including nominal but labelled separately.
    trajectories=out.parent/'EXPERIMENTS_R2_20261024_OLD_TRAJECTORIES.tar.gz'
    if trajectories.exists():raise FileExistsError(trajectories)
    trajectory_map=[];members=[]
    with tarfile.open(trajectories,'w:gz',compresslevel=1) as tar:
        for model in read(EVAL/'models.json'):
            folder=evaluation_result_dir(EVAL,model);prefix=Path('old_trajectories')/model['model_id']
            for name in ['run_manifest.json','episodes.json']:
                path=folder/name;arc=str(prefix/name);tar.add(path,arcname=arc,recursive=False);members.append((arc,sha(path)))
            for e in read(folder/'episodes.json'):
                path=folder/Path(e['trajectory']).name;arc=str(prefix/path.name);tar.add(path,arcname=arc,recursive=False)
                members.append((arc,sha(path)))
                trajectory_map.append(dict(model_id=model['model_id'],kind=e['kind'],state_id=e['state_id'],
                    original_trajectory=e['trajectory'],archive_path=arc,sha256=sha(path)))
                initial=folder/(e['state_id']+'.initial.json')
                arc=str(prefix/initial.name);tar.add(initial,arcname=arc,recursive=False);members.append((arc,sha(initial)))
        sums=''.join(f'{digest}  {name}\n' for name,digest in members).encode()
        info=tarfile.TarInfo('SHA256SUMS');info.size=len(sums);tar.addfile(info,io.BytesIO(sums))
    digest=sha(trajectories);trajectories.with_name(trajectories.name+'.sha256').write_text(f'{digest}  {trajectories.name}\n')
    write(out/'checks/old_trajectory_archive.json',dict(name=trajectories.name,size_bytes=trajectories.stat().st_size,sha256=digest,
        perturbed=1050,nominal=21,mapping=trajectory_map))
    write(out/'sources/file_mapping.json',mapping)
    (out/'sources/git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write(out/'checks/final_cpu_status.json',dict(status='partial_delivery',training_started=False,tests_passed=6,
        U1='Push/Pull complete',U2='dimensions/source only; hand distance pending',U3='old1050 complete',U4='optional not run',
        U5='CPU reference equivalence passed; cloud launcher/smoke/training/eval pending',
        U6='geometry equivalence, original budget/model, fixed states ready; eval case and cloud training pending',
        U7='not rendered',U8='facts and Chair/Bucket SO3 recomputation; other target-frame/runtime-body-list audit pending'))
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    assert not any(p.is_symlink() for p in out.rglob('*'))
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(out,arcname=out.name)
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
    validation=[]
    # Verify by extracting only safe regular files/directories into fresh /tmp roots.
    for path,prefix in [(trajectories,''),(archive,out.name)]:
        temporary=Path(tempfile.mkdtemp(prefix='r2_package_verify_'))
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                assert member.isfile() or member.isdir()
            tar.extractall(temporary,filter='data')
        base=temporary/prefix;count=0
        for line in (base/'SHA256SUMS').read_text().splitlines():
            expected,rel=line.split('  ',1);assert sha(base/rel)==expected;count+=1
        validation.append(dict(archive=str(path),sha256=sha(path),size_bytes=path.stat().st_size,verified_files=count,verification_dir=str(temporary)))
    write(out.parent/'EXPERIMENTS_R2_20261024_CPU_package_validation.json',validation)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    import sys,json
    main()
