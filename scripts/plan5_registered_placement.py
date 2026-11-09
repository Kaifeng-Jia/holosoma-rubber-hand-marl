"""Opt-in R2 Push placement; preserve the historical loader and every other term."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np


def configure_push_placement(config, proof_path=None, *, skill='push', root=None):
    if proof_path is None:
        return config, None
    if skill != 'push':
        raise ValueError('Registered Push placement is only valid for Push')
    path = Path(proof_path)
    raw = path.read_bytes()
    proof = json.loads(raw)
    if proof['status'] != 'cpu_reference_preflight_passed_training_not_started':
        raise ValueError('Not the audited CPU placement proof')
    roles = proof['role_parameters']
    if len(roles) != 2:
        raise ValueError('Expected exactly two registered roles')
    c = np.array([r['c'] for r in roles], dtype=float)
    spacing = float(np.linalg.norm(c[1] - c[0]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Invalid separation')
    for role in roles:
        if role['parity'] != 1 or not np.allclose(role['Q'], np.eye(3), atol=1e-12, rtol=0):
            raise ValueError('This loader requires translation-only proper rotations')
    if not np.allclose(c, [[-spacing/2,0,0],[spacing/2,0,0]], atol=1e-12, rtol=0):
        raise ValueError('Placement cannot be represented by the original lateral-spacing loader')
    if not np.isclose(spacing, proof['lateral_spacing_m'], atol=1e-12, rtol=0):
        raise ValueError('Proof spacing differs from Q/c')
    if root is not None:
        for asset in proof['assets']:
            relative = 'src/' + asset['path'].split('/src/', 1)[1]
            if hashlib.sha256((Path(root)/relative).read_bytes()).hexdigest() != asset['sha256']:
                raise ValueError('Placement proof asset changed: ' + relative)
    hooks = {}
    for name in ('setup_terms', 'reset_terms', 'step_terms'):
        terms = dict(getattr(config.command, name))
        term = terms['paired_motion_command']
        if term.params.get('paired_reference_file') is not None:
            raise ValueError('Do not replace the original Push reference loader')
        terms['paired_motion_command'] = replace(term, params=dict(term.params, lateral_spacing_m=spacing))
        hooks[name] = terms
    result = replace(config, command=replace(config.command, **hooks))
    return result, dict(proof_sha256=hashlib.sha256(raw).hexdigest(),
                        lateral_spacing_m=spacing, original_diagnostic_status=proof['original_diagnostic_status'],
                        loader='original_PairedA1Reference', hooks=list(hooks), offline_runtime_used=False)


def comparison_body_indexes(runtime_names, expected_names, required_names):
    """Compare actual command bodies, not additional offline-only hand frames.

    The frozen CPU report also exports fixed rubber-hand frames. The historical
    simulator command requests its configured body list, which excludes them.
    Never substitute wrist origins for hand origins, or change simulation bodies.
    """
    expected_names = [str(n) for n in expected_names]
    missing = [n for n in expected_names if n not in runtime_names]
    offline_only = {'left_rubber_hand_link', 'right_rubber_hand_link'}
    if set(missing) - offline_only:
        raise ValueError('Required reference bodies absent at runtime: '+str(missing))
    if any(n not in runtime_names or n not in expected_names for n in required_names):
        raise ValueError('A training-tracked body is missing from the comparison')
    names = [n for n in expected_names if n in runtime_names]
    return [runtime_names.index(n) for n in names], [expected_names.index(n) for n in names], names, missing


def export_runtime_reference(env, output, provenance, proof_path):
    """Check actual GPU reference against the frozen CPU loader before updates."""
    if provenance is None:
        return None
    import torch
    command = env.command_manager.get_state('paired_motion_command')
    samples = command.reference.sample(torch.arange(command.reference.num_frames, device=env.device))
    arrays = {k:v.detach().cpu().numpy() for k,v in samples.items()}
    expected_path = Path(proof_path).with_name('push_registered_loader_channels.npz')
    if not expected_path.exists():
        expected_path = Path(proof_path).parent/'checks/push_registered_loader_channels.npz'
    errors = {}
    with np.load(expected_path, allow_pickle=False) as expected:
        indexes, expected_indexes, compared_names, offline_names = comparison_body_indexes(
            env.simulator._body_list, expected['body_names'], command.motion_cfg.body_names_to_track)
        for key,value in arrays.items():
            selected = value[:,:,indexes] if key.startswith('agent_body_') else value
            target = expected[key][:,:,expected_indexes] if key.startswith('agent_body_') else expected[key]
            if selected.shape != target.shape:
                raise ValueError('Runtime reference shape differs: '+key)
            delta = np.abs(selected.astype(float)-target.astype(float))
            maximum = float(delta.max())
            # CPU/GPU finite differences can amplify float32 rounding in velocity.
            tolerance = 1e-4 if 'vel' in key else 2e-6
            if not np.isfinite(maximum) or maximum > tolerance:
                raise ValueError(f'Runtime reference channel differs: {key}: {maximum}')
            errors[key] = dict(max_abs=maximum,tolerance=tolerance)
    path = Path(output)/'actual_reference_channels.npz'
    np.savez_compressed(path,**arrays,body_names=env.simulator._body_list,
                        joint_names=env.simulator.dof_names,fps=command.reference.fps)
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_channels_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
                comparison=errors,quaternion_order='xyzw', compared_body_names=compared_names,
                offline_only_body_names=offline_names,
                body_mapping_note='Offline fixed-hand frames are not runtime command bodies; no wrist-for-hand substitution')
