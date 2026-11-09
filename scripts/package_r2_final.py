#!/usr/bin/env python3
"""Collect R2 final evidence, retaining original file identities and explicit gaps."""
import argparse,csv,json,hashlib,shutil,subprocess,sys,platform,importlib.metadata as md
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import asdict
import numpy as np
from finalize_experiments_r2_handoff import ROOT,NEW,EVAL,read,write,sha,csv_read,csv_write

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--readme',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    mapping=[];seen={}
    def copy(src,dst=None):
        src=Path(src).resolve();assert src.is_file(),src
        h=sha(src)
        if (str(src),h) in seen:return seen[str(src),h]
        dst=Path(dst) if dst else out/'sources/inputs'/f'{h[:12]}_{src.name}'
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==h,(src,dst)
        else:shutil.copy2(src,dst)
        r=dict(original_path=str(src),package_path=str(dst.relative_to(out)),sha256=h)
        mapping.append(r);seen[str(src),h]=r;return r
    def portable_urdf(src):
        src=Path(src);raw=copy(src);tree=ET.parse(src)
        for node in tree.findall('.//mesh'):
            f=Path(node.attrib['filename']);f=f if f.is_absolute() else src.parent/f
            r=copy(f);node.set('filename','../'+r['package_path'].removeprefix('sources/'))
        dest=out/'sources/portable_urdf'/f'{sha(src)[:12]}_{src.name}';dest.parent.mkdir(parents=True,exist_ok=True)
        tree.write(dest,encoding='utf-8',xml_declaration=True)
        for node in ET.parse(dest).findall('.//mesh'):assert (dest.parent/node.attrib['filename']).exists()
        return dict(original=raw,portable_path=str(dest.relative_to(out)),portable_sha256=sha(dest))
    # Preserve previous batch mapping with its new prefix.
    for r in read(out/'CPU_first_batch/sources/file_mapping.json'):
        rr=dict(r,package_path='CPU_first_batch/'+r['package_path']);assert sha(out/rr['package_path'])==rr['sha256'];mapping.append(rr)
    scripts=['finalize_experiments_r2_handoff.py','render_r2_handoff.py','render_r2_shared_chair.py','package_r2_final.py',
             'export_recorded_demo_media.py','render_wrist_reference_pairs.py','prepare_experiments_r2_cpu.py',
             'collect_experiments_r2_geometry.py','recompute_r2_wrist_cpu.py','plan5_registered_placement.py',
             'run_experiments_r2_evaluation.py','collect_experiments_r2_evaluation.py']
    for n in scripts:copy(ROOT/'scripts'/n,out/'sources/current_scripts'/n)
    copy(ROOT/'paper_handoff_exports/WRIST_FIGURE3_BUCKET_AUDIT_20261005/render_tools/export_recorded_demo_media.py',out/'sources/old_render/export_recorded_demo_media.py')
    for n in ['retarget-rubberhand-A1.md','docs/experiments/retarget_method_comparison_20260925.json','docs/experiments/experiments_r2_cloud_20261024.md']:
        copy(ROOT/n)
    deployment=ROOT/'logs/ExperimentsR2/deployment_20261024';ops=deployment/'retry1/cloud_complete_20261026'
    copy(deployment/'r2_source_assets.tar.gz',out/'sources/training_source_assets.tar.gz')
    for f in ops.iterdir():
        if f.suffix=='.json' or f.name.endswith('.py'):copy(f,out/'sources/training_operations'/f.name)
    # Every real training phase config/status and sampled metric endpoints; not intermediate models.
    for d in sorted((ROOT/'logs/ExperimentsR2_retry1').iterdir()):
        if not d.is_dir() or 'smoke' in d.name:continue
        for n in ['run_config.json','status.json']:
            if (d/n).exists():copy(d/n,out/'sources/training_runs'/d.name/n)
        f=d/'metrics.jsonl'
        if f.exists():
            lines=f.read_text().splitlines();write(out/'sources/training_runs'/d.name/'metrics_endpoints.json',
                dict(source=str(f),source_sha256=sha(f),lines=len(lines),first=json.loads(lines[0]),last=json.loads(lines[-1])))
    hist=Path('/home/kevin/holosoma-rubber-hand-marl/logs/Plan5Kick/mirrored_kick_full8000_seed721_env2048/run_config.json')
    copy(hist,out/'sources/training_runs/historical_kick721/run_config.json')
    # Explicitly preserve config and actual timing/physics per model, not infer from filenames.
    settings=read(out/'CPU_first_batch/checks/per_model_settings.json')
    from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline,g1_29dof_plan5_pull_baseline
    from holosoma.config_values.marl.g1.kick_experiment import g1_29dof_plan5_kick_baseline
    configs=dict(push=g1_29dof_plan5_push_baseline,pull=g1_29dof_plan5_pull_baseline,kick=g1_29dof_plan5_kick_baseline)
    for case,cfg in configs.items():
        write(out/'checks'/f'{case}_resolved_terms_and_timing.json',dict(reward=asdict(cfg.reward),termination=asdict(cfg.termination),sim=asdict(cfg.simulator.config.sim)))
    for m in read(NEW/'models.json'):
        manifest=read(NEW/m['group']/'results'/m['model_id']/'run_manifest.json')
        d=ROOT/'logs/ExperimentsR2_retry1'/Path(m['checkpoint']).parent.name
        cfg=read(hist if m['training_seed']==721 and m['case']=='kick' else d/'run_config.json')
        sim=asdict(configs[m['case']].simulator.config.sim)
        settings.append(dict(model_id=m['model_id'],scalar_fields=cfg,actual_evaluation_physics=manifest['physics'],
            reference_frames=manifest['reference_frames'],reference_fps=50,
            first_last_sample_seconds=(manifest['reference_frames']-1)/50,full_horizon_seconds=manifest['reference_frames']/50,
            checkpoint_iteration=m['checkpoint_iteration'],physics=sim,w_z=1.0,independent_height_reward=False,
            partner_reference='registered paired A1 original loader' if m['case']=='push' else 'legacy mirrored runtime; checked geometric equivalence',
            critic_only_iterations=50 if m['case']=='push' else 0,actor_update_iterations=13000 if m['case']=='push' else 8000))
    write(out/'checks/per_model_settings.json',settings)
    # Terminal-term distributions include original results, separately labelled from new results.
    terminal=[]
    for label,rows in [('old',csv_read(EVAL/'episodes.csv')),('new',csv_read(out/'U5_push_registered/episodes.csv')+csv_read(out/'U6_kick/episodes.csv'))]:
        counts={}
        for r in rows:
            for term,active in json.loads(r['terminal_terms']).items():
                key=(r['configuration'],r['training_seed'],r['kind'],term);n,k=counts.get(key,(0,0));counts[key]=(n+1,k+bool(active))
        for (config,seed,kind,term),(n,k) in counts.items():terminal.append(dict(batch=label,configuration=config,seed=seed,kind=kind,term=term,episodes=n,active=k))
    csv_write(out/'checks/termination_distribution.csv',terminal)
    # Kick is retained in U1 with explicitly unavailable hand windows; never rename foot evidence.
    for name,byhand in [('ref_role_synthesis.csv',False),('ref_role_synthesis_by_hand.csv',True)]:
        rows=csv_read(out/name)
        if not any(r['task']=='kick' for r in rows):
            for role,parity in [(0,1),(1,-1)]:
                for hand in ['left','right'] if byhand else [None]:
                    r={k:'--' for k in rows[0]};r.update(task='kick',role=role,parity=parity,n_contact_frames=0)
                    r['placement(registered|prescribed)']='registered'
                    if byhand:r['hand']=hand
                    rows.append(r)
            csv_write(out/name,rows)
    write(out/'checks/u1_kick_missing.json',dict(reason='No frozen source-hand proximity windows; hand metric unavailable, not zero distance.',
          placement_label='registered denotes checked pose-equivalent geometry, NOT historical training provenance; actual runtime remains legacy mirrored',
          geometry_evidence='CPU_first_batch/checks/kick_registration_equivalence.json'))
    write(out/'checks/u2_definition_gap.json',dict(status='partial_size_complete_distance_unmeasured',
          missing=['consistent source palm/hand point corresponding to robot rubber_hand_link origin','frozen source proximity windows under that correspondence'],
          not_missing=['retarget nominal poses','human heights','actual object mesh and source trajectory','pre-wrist shared or reoptimized reference'],
          reason='Wrist origin is not rubber-hand origin; substitute wrist distances would measure a different quantity. No data-driven mapping/window was established for these U2 rows.',
          required_next='Agree source palm-frame origin correspondence and source-only proximity window before CPU distance calculation; no training necessary',
          source_evidence='CPU_first_batch/checks/shared_scene_sources.json',missing_value='--; not zero/no-contact'))
    # Classify known target sources without substituting metrics from another experiment.
    wrist=[dict(task=t,target_source=s,status=st,reason=r) for t,s,st,r in [
      ('Push A1','PT wrist quaternion + Wrist/Index1/Middle1/Pinky1 calibration','not recomputed this round','Original A1 has real hand landmarks; do not label fixed-palm; historical and September11 baselines differ'),
      ('Pull','original retarget/physical rollout; full palm target not archived in this audit','unavailable','Cannot reconstruct a target merely from robot orientation'),
      ('Kick','original retarget/physical rollout; no audited palm target','unavailable','Foot task is not a palm SO(3) ablation'),
      ('Chair','CORE4D SMPL-X palm landmarks + wrist calibration','recomputed','All235 original30Hzframes and4hand sequences'),
      ('Bucket','CORE4D SMPL-X palm landmarks + wrist calibration','recomputed','All299 original30Hzframes and4hand sequences'),
      ('Small Table T07','pre-wrist nominal shared-scale reference','not applicable','Selected training reference has no applied paired wrist-only before/after intervention'),
      ('Plan B','artificial fixed-palm target','out of scope','Not the A1 reference used in U5')]]
    write(out/'checks/wrist_target_applicability.json',wrist)
    # Resolve all visual assets into the package, retaining raw URDF and relative-path derivative.
    urdfs=set()
    for f in (out/'U7_media').rglob('manifest.json'):
        m=read(f)
        if 'robot_urdf_source' in m:
            urdfs.update([m['robot_urdf_source'],m['object_urdf_source']])
    urdfs.add(str(ROOT/'logs/Core4DPreviews/chair021_20231020_074_baseline_20260908/two_stage/assets/chair021.urdf'))
    write(out/'checks/portable_urdf_mapping.json',[portable_urdf(Path(p)) for p in sorted(urdfs)])
    # Complete final media index after visual review. No fabricated contact phase/force labels.
    media=read(out/'U7_media/render_manifest.json')
    for r in media['records']:
        if r.pop('stages_pending_visual_review',False):
            labels=['俯身准备','俯身随桌推进','随桌移动末段','起身结束'] if r['model_id'].startswith('Push') else ['屈膝准备','抬腿靠近桌边','随桌移动与收腿','恢复站姿']
            r['stages']=[dict(frame=c['frame'],time_s=c['time_s'],label=l,visual_basis='Saved pose image; no force inference') for c,l in zip(r['frame_cameras'],labels)]
        if r['render_manifest'].endswith('rollouts/Kick-baseline_seed721/manifest.json'):
            r['superseded_for_framing_by']='U7_media/rollouts/Kick-baseline_seed721_fullbody/manifest.json'
    full=read(out/'U7_media/rollouts/Kick-baseline_seed721_fullbody/manifest.json')
    media['preferred_kick_stills']=dict(manifest='U7_media/rollouts/Kick-baseline_seed721_fullbody/manifest.json',
                 reason='distance scale1.3 prevents foot clipping, no pose edits',camera=full['camera'],frame_cameras=full['camera_frames'],fovy=full['fovy_degrees'])
    media['shared_scene_manifest']='U7_media/shared_scene/render_manifest.json'
    media['failure_review']=dict(observed='one robot torso tips strongly over table; other remains nearer upright',
               not_established='shin/foot force causing failure',actual_frames=[67,87],aligned_reference_frame=87,
               termination='joint_bad_tracking aggregate OR; exact triggering subcondition not in recorded terminal_terms')
    write(out/'U7_media/render_manifest.json',media)
    cams=read(out/'U7_media/camera.json');cams.append(media['preferred_kick_stills']);cams.append(read(out/'U7_media/shared_scene/camera.json'));write(out/'U7_media/camera.json',cams)
    write(out/'U7_media/failure_case/callout_suggestions.json',dict(color_rgb=[220,30,30],images_unchanged=True,
       suggestions=[dict(image=f'actual/frame_{f:04d}.png',box_normalized_xyxy=[.34,.46,.52,.63],
                         label='Torso tipping / posture deviation; not measured leg force') for f in [67,87]],
       coordinate_basis='normalized full1920x1080 image; layout suggestion, inspect when composing'))
    # The original CSV absolute paths are untouched; this combined map makes evidence portable.
    write(out/'sources/file_mapping.json',mapping)
    status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=ROOT,text=True)
    (out/'sources/worktree_status.txt').write_text(status);(out/'sources/code_diff.patch').write_text(diff)
    write(out/'provenance.json',dict(instruction_commit='ab915c56e3bd6341a3214131fcfd7f539cb6f89e',
        local_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        local_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        local_worktree=str(ROOT),dirty_tree_preserved=True,script=str(Path(__file__)),script_sha256=sha(Path(__file__)),
        command=[sys.executable,*sys.argv],python=sys.version,platform=platform.platform(),
        packages={n:md.version(n) for n in ['numpy','scipy','torch','yourdfpy','trimesh']},
        train_source_archive='sources/training_source_assets.tar.gz',train_source_sha256=sha(out/'sources/training_source_assets.tar.gz'),
        cloud_patch_manifest='sources/training_operations/source_patch_manifest.json',
        actual_evaluation_sources='sources/evaluation_snapshot',training_complete=True,evaluation_complete=True,
        missing=['U2 source hand correspondence/window dependent distance','U4 optional proximity','full historical palm targets for some U8 tasks'],
        source_mapping='sources/file_mapping.json',new_trajectory_mapping='checks/trajectory_mapping.json',
        old_archive_mapping='CPU_first_batch/checks/old_trajectory_archive.json',
        cloud_shutdown_performed=False,git_push_performed=False))
    shutil.copy2(a.readme,out/'README.md')
    print('Evidence finalized',len(mapping),'source mappings')
if __name__=='__main__':main()
