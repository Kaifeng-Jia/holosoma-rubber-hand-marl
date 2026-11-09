#!/usr/bin/env python3
"""Package round2 results without environments, checkpoints or superseded runs."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np
from reference_geometry_pilot import ROOT,write,sha
from run_reference_geometry_round2 import BATCH,rotation_delta
from package_reference_geometry_pilot import FILES as FIRST_FILES,HELPERS

FILES=['scripts/reference_geometry_round2_correspondence.py','scripts/reference_geometry_round2_checks.py',
 'scripts/run_reference_geometry_round2.py','scripts/finalize_reference_geometry_round2.py',
 'scripts/render_reference_geometry_round2.py','scripts/package_reference_geometry_round2.py','tests/test_reference_geometry_round2.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    summary=json.loads((out/'summary.json').read_text());rows=[]
    for item in summary:
        case=item['case'];folder=out/case;before=np.load(BATCH/case/'legacy_runtime_recomputed.npz');after=np.load(folder/'new_runtime.npz');delta={}
        for key in before.files:
            if key not in after.files or not np.issubdtype(before[key].dtype,np.number) or before[key].ndim==0:continue
            delta[key+'_delta']=rotation_delta(after[key],before[key]) if 'quat' in key else after[key]-before[key]
        np.savez_compressed(folder/'runtime_delta_per_frame.npz',**delta)
        old=np.load(BATCH/case/'legacy_pair_reference.npz');new=np.load(folder/'new_pair_reference.npz')
        np.savez_compressed(folder/'pose_delta_per_frame.npz',root_position_m=new['robot_qpos'][...,:3]-old['robot_qpos'][...,:3],
            root_rotation_rad=rotation_delta(new['robot_qpos'][...,3:7],old['robot_qpos'][...,3:7]),joint_angle_rad=new['robot_qpos'][...,7:]-old['robot_qpos'][...,7:])
        shutil.copyfile(BATCH/case/'legacy_pair_reference.npz',folder/'old_pair_reference.npz')
        shutil.copyfile(BATCH/case/'legacy_runtime_recomputed.npz',folder/'old_runtime_recomputed.npz')
        shutil.copyfile(BATCH/case/'preprocessed_source.npz',folder/'preprocessed_source.npz')
        shutil.copyfile(BATCH/case/'hand_samples.npz',folder/'hand_samples.npz')
        interaction=json.loads((BATCH/case/'source_interactions.json').read_text());stats=[]
        for role in [0,1]:
            for hand in interaction['records']:
                name=hand['hand'];mask=np.zeros(len(new['robot_qpos']),bool)
                for start,end in hand['source_windows']:mask[start:end]=True
                for variant in ['old','new']:
                    d=np.load(folder/f'{variant}_role{role}_checks.npz')
                    for key in [k for k in d.files if k.startswith(name) and 'added_' not in k]:
                        for scope,eligible in [('all',np.ones(len(mask),bool)),('own_source_window',mask)]:
                            values=d[key][eligible & np.isfinite(d[key])]
                            stats.append(dict(role=role,hand=name,variant=variant,metric=key,scope=scope,frames=len(values),
                                mean=float(values.mean()) if len(values) else '',max=float(values.max()) if len(values) else '',rms=float(np.sqrt(np.mean(values*values))) if len(values) else ''))
        with (folder/'hand_window_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
        pose=item['old_new_pose_comparison'];rows.append(dict(case=case,status=item['status'],role0_candidates=item['role_candidate_counts'][0],role1_candidates=item['role_candidate_counts'][1],pair_count=item['pair_count'],passing_pairs=item['passing_pairs'],
            max_root_position_component_change_m=pose['root_translation_m']['max_abs'],root_position_component_rms_m=pose['root_translation_m']['rms'],
            max_root_angle_change_rad=pose['root_rotation_rad']['max_abs'],max_joint_angle_change_rad=pose['joints_rad']['max_abs']))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'metric_units.json',dict(position_distance_penetration='m',root_joint_rotation='rad unless key ends deg',
        linear_velocity='m/s',angular_velocity='rad/s',linear_acceleration='m/s^2',fps='Hz',
        component_max_rms='max_abs and RMS on raw components unless explicitly named displacement norm or geodesic quaternion angle',
        signed_proxy='Positive SAT separation lower bound capped at0.1m; negative OBB overlap, not exact concave mesh penetration',
        NaN='No active source window/support sample; CSV blank, summary count0/null. Not silently treated as zero.',
        mirror_hand_labels='Metrics retain SOURCE left/right identity; negative-parity role exchanges actual physical hands. See association_evaluation.json.'))
    for rel in list(dict.fromkeys(FILES+FIRST_FILES+HELPERS)):
        dest=out/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    patch=''
    for rel in FILES:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',rel],cwd=ROOT,text=True,capture_output=True)
        assert result.returncode in (0,1);patch+=result.stdout
    (out/'round2_new_files.patch').write_text(patch)
    (out/'REPRODUCE.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /home/kevin/holosoma-core4d-base
export OPENBLAS_NUM_THREADS=1
audit_python=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
audit_output=${1:?Supply a NEW output directory}
"$audit_python" -B scripts/run_reference_geometry_round2.py --output "$audit_output"
"$audit_python" -B scripts/finalize_reference_geometry_round2.py --output "$audit_output"
"$audit_python" -B scripts/render_reference_geometry_round2.py --output "$audit_output"
# Copy the reviewed README_CN.md into the new result before packaging.
"$audit_python" -B scripts/package_reference_geometry_round2.py --output "$audit_output"
''')
    (out/'git_status_at_delivery.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    files=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        for p in files+[out/'SHA256SUMS']:t.add(p,arcname=str(Path(out.name)/p.relative_to(out)))
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),member_checksums=len(files)),indent=2))

if __name__=='__main__':main()
