#!/usr/bin/env python3
"""Frozen current-source overlay for the isolated 5kg table, no environments/models."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from package_core4d_bucket_transfer import (
    PYTHON_PACKAGES, FORBIDDEN_COMPONENTS, TRAINING_ROBOT, REFERENCE_ROBOT, REFERENCE_XML,
    asset_dependencies, identity, safe_file, git,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=args.output_dir.resolve()
    if output.exists():raise FileExistsError(output)
    selected={f'scripts/{name}' for name in (
        'train_core4d_smalltable.py','smoke_core4d_smalltable.py','evaluate_core4d_smalltable.py',
        'prepare_core4d_smalltable_a.py','package_core4d_smalltable_a_transfer.py','package_core4d_bucket_transfer.py')}
    selected.add('SMALLTABLE5KG_A_TRAINING_CN.md')
    for package in PYTHON_PACKAGES:
        for parent,dirs,files in os.walk(root/package,followlinks=False):
            dirs[:]=[d for d in dirs if d not in FORBIDDEN_COMPONENTS and not d.startswith('converted_rank')]
            for name in files:
                if name.endswith('.py'):selected.add((Path(parent)/name).relative_to(root).as_posix())
    for package in ('src/holosoma','src/holosoma_retargeting'):
        selected.add(package+'/pyproject.toml')
    data='src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A'
    selected.update(data+'/'+name for name in ('core4d_pair_runtime_fps50.npz','desk001_5kg_training.urdf',
                                             'interaction_vectors_v1.npz','training_asset_manifest.json'))
    selected.update(asset_dependencies(root,(TRAINING_ROBOT,REFERENCE_ROBOT,REFERENCE_XML,data+'/desk001_5kg_training.urdf')))
    records={name:identity(root,name) for name in sorted(selected)}
    promotion=json.loads((root/data/'training_asset_manifest.json').read_text())
    assert promotion['training_ready'] and promotion['object_mass_kg']==5
    for key,name in (('runtime_reference_sha256','core4d_pair_runtime_fps50.npz'),
                     ('training_object_urdf_sha256','desk001_5kg_training.urdf'),
                     ('interaction_artifact_sha256','interaction_vectors_v1.npz')):
        assert promotion[key]==records[data+'/'+name]['sha256']
    manifest={'format':'core4d_smalltable5kg_A_overlay_v1','base_git_head':git(root,'rev-parse','HEAD'),
              'files':records,'payload_prefix':'payload','experiment':'smalltable5kg_A',
              'payload_file_count':len(records),'payload_bytes':sum(r['size_bytes'] for r in records.values()),
              'scope':'current dirty/untracked source and selected robot/table assets; no logs/checkpoints/environments',
              'formal_training_authorized':False}
    raw=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    hashes=[f"{r['sha256']}  payload/{name}" for name,r in records.items()]
    hashes.append(hashlib.sha256(raw).hexdigest()+'  package_manifest.json')
    output.mkdir(parents=True)
    archive=output/'smalltable5kg_A_transfer.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        def add(name,contents):
            info=tarfile.TarInfo(name);info.size=len(contents);info.mode=0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(contents))
        for name,r in records.items():
            contents=safe_file(root,name).read_bytes()
            assert hashlib.sha256(contents).hexdigest()==r['sha256']
            add('payload/'+name,contents)
        add('package_manifest.json',raw)
        add('SHA256SUMS',('\n'.join(hashes)+'\n').encode())
    with (output/'package_manifest.json').open('xb') as f:f.write(raw)
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    with (output/(archive.name+'.sha256')).open('x') as f:f.write(digest+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'sha256':digest,'files':len(records),'bytes':archive.stat().st_size}))


if __name__=='__main__':main()
