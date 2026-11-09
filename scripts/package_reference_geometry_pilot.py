#!/usr/bin/env python3
"""Package batch-one evidence; do not infer missing full-role candidates."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import numpy as np
from reference_geometry_pilot import ROOT, write, sha

FILES = ['scripts/reference_geometry_math.py', 'scripts/reference_geometry_pilot.py',
         'scripts/finalize_reference_geometry_pilot.py', 'scripts/render_reference_geometry_pilot.py',
         'scripts/package_reference_geometry_pilot.py', 'tests/test_reference_geometry_pilot.py']
HELPERS = ['scripts/render_wrist_reference_pairs.py', 'scripts/export_recorded_demo_media.py',
           'scripts/visualize_core4d_pair.py',
           'src/holosoma_retargeting/holosoma_retargeting/dual_a1_layout.py',
           'src/holosoma_retargeting/holosoma_retargeting/dual_pull_reference.py',
           'src/holosoma_retargeting/holosoma_retargeting/dual_pull_runtime_reference.py',
           'src/holosoma/holosoma/envs/marl/paired_a1_reference.py']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args=p.parse_args()
    out=args.output.resolve(); rows=[]; manifests={}; specs={}
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests',
                         '-p','test_reference_geometry_pilot.py','-v'],cwd=ROOT,text=True,capture_output=True)
    (out/'synthetic_tests.txt').write_text(test.stdout+test.stderr+f'\nExit code: {test.returncode}\n')
    if test.returncode: raise RuntimeError('Synthetic tests failed')
    for case in ['push','pull']:
        folder=out/case; m=json.loads((folder/'input_manifest.json').read_text()); manifests[case]=m
        specs[case]=json.loads((folder/'correspondence_spec.json').read_text())
        interactions=json.loads((folder/'source_interactions.json').read_text())
        fk=json.loads((folder/'robot_fk_mechanism_check.json').read_text())
        d=np.load(folder/'source_distances.npz'); n=m['frames']
        with (folder/'source_per_frame.csv').open('w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['frame','time_s','hand','min_unsigned_distance_m',
                'near_fraction','max_sampled_penetration_m'])
            for t in range(n):
                for r in interactions['records']:
                    h=r['hand']; writer.writerow([t,t/m['fps'],h,*[float(d[h+'_'+k][t]) for k in
                       ['min_distance_m','near_fraction','max_penetration_m']]])
        rows.append(dict(case=case,status='needs_role_regions',frames=n,fps=m['fps'],
          active_parts=';'.join(specs[case]['complete_source_part_tuple']),
          max_sampled_source_hand_penetration_m=max(r['max_penetration_m'] for r in interactions['records']),
          real_mesh_mirror_max_error_m=max(r['actual_mirrored_mesh_distance_max_m'] for r in fk['actual_mesh_mirror_errors']),
          full_role_candidates='undefined_pending_regions',selected_pair='none',new_reference='not_generated'))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(out/'input_manifest.json',manifests);write(out/'correspondence_spec.json',specs)
    # Geometry-only diagnostic equivalence checked AFTER complete rules freeze.
    leg=json.loads((out/'pull/leg_subtuple_diagnostic.json').read_text())
    simple=[c for c in leg['candidates'] if c['parity']==1 and np.allclose(c['Q'],np.eye(3),atol=1e-12)]
    write(out/'pull/leg_translation_diagnostic_after_freeze.json',dict(
      scope='Incomplete leg-subtuple fits only; never selected as full reference',candidates=simple,
      finding='Identity-rotation same-X-side leg correspondence recovers |delta Z|=0.4390236 m from actual geometry; this is not a full-clip feasibility or improvement claim.'))
    original=np.load(manifests['pull']['runtime_reference']['path']); rebuilt=np.load(out/'pull/legacy_runtime_recomputed.npz')
    comparison={}
    for key in original.files:
        if key in rebuilt.files and np.issubdtype(original[key].dtype,np.number):
            comparison[key]=dict(shape_equal=original[key].shape==rebuilt[key].shape,
                max_abs_difference=float(np.max(np.abs(original[key]-rebuilt[key]))) if original[key].shape==rebuilt[key].shape else None)
    write(out/'pull/legacy_runtime_channel_comparison.json',dict(scope='Raw numerical channel reproduction, not new candidate performance',fields=comparison))
    code=out/'code_snapshot'
    for rel in FILES+HELPERS:
        dst=code/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dst)
    patch=''
    for rel in FILES:
        diff=subprocess.run(['git','diff','--no-index','--','/dev/null',rel],cwd=ROOT,capture_output=True,text=True)
        if diff.returncode not in [0,1]:raise RuntimeError(diff.stderr)
        patch+=diff.stdout
    (out/'pilot_new_files.patch').write_text(patch)
    (out/'git_status_at_delivery.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write(out/'status.json',dict(batch=1,status='needs_role_regions',training_started=False,physics_steps=0,
      formal_assets_modified=False,committed=False,pushed=False,synthetic_tests=8,synthetic_test_exit_code=0,
      real_inputs_verified=True,source_surface_windows_extracted=True,real_robot_fk_mechanism_verified=True,
      full_candidate_generation=False,selected_pair=None,blocked_only=['complete-role mapping and dependent validation'],
      not_implemented_yet=['complete face/part adjacency graph','full candidate normal/ground/body/pair screening','new-candidate reference/runtime/media']))
    (out/'REPRODUCE.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /home/kevin/holosoma-core4d-base
audit_python=/home/kevin/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python
# Supply a NEW output directory; source analysis refuses to overwrite.
audit_output=${1:?Supply a new output directory}
"$audit_python" -B scripts/reference_geometry_pilot.py --output "$audit_output"
"$audit_python" -B scripts/finalize_reference_geometry_pilot.py --output "$audit_output"
"$audit_python" -B scripts/render_reference_geometry_pilot.py --output "$audit_output"
"$audit_python" -B scripts/package_reference_geometry_pilot.py --output "$audit_output"
''')
    excluded={'push_initial_nearest_only_diagnostic','pull_initial_nearest_only_diagnostic','push_incomplete_manifest_diagnostic'}
    files=sorted(p for p in out.rglob('*') if p.is_file() and not excluded.intersection(p.relative_to(out).parts) and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    archive=out.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for path in files+[out/'SHA256SUMS']:tar.add(path,arcname=str(Path(out.name)/path.relative_to(out)))
    checksum=archive.with_name(archive.name+'.sha256');checksum.write_text(f'{sha(archive)}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),checksum=str(checksum)),indent=2))


if __name__=='__main__':main()
