"""Height experiment CLI/evaluation checks without loading a simulator."""
import ast
import inspect
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from holosoma.config_values.marl.g1.core4d_smalltable_experiment import g1_29dof_core4d_smalltable_baseline
from holosoma.utils import eval_utils


ROOT = Path(__file__).resolve().parents[5]
GRAPH = ROOT / 'src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable/interaction_mesh_100_v1.npz'


class BeforeSimulator(Exception):
    pass


def prefix(monkeypatch, args):
    def stop(config):
        raise BeforeSimulator(dict(inspect.currentframe().f_back.f_globals))
    monkeypatch.setattr(eval_utils, 'init_sim_imports', stop)
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setattr(sys, 'argv', ['train_core4d_smalltable.py', *args])
    runpy.run_path(str(ROOT / 'scripts/train_core4d_smalltable.py'), run_name='__main__')


@pytest.mark.parametrize('flag,value', [
    ('--object-height-penalty-weight', '-1'), ('--object-height-penalty-weight', 'nan'),
    ('--object-height-penalty-weight', 'inf'), ('--object-height-penalty-scale', '0'),
    ('--object-height-penalty-scale', '-1'), ('--object-height-penalty-scale', 'nan'),
    ('--object-height-penalty-scale', 'inf'),
])
def test_bad_parameters_rejected_before_simulator(monkeypatch, flag, value):
    with pytest.raises(SystemExit) as caught:
        prefix(monkeypatch, [flag, value])
    assert caught.value.code == 2


def test_height_penalty_not_silently_enabled_for_chair(monkeypatch):
    with pytest.raises(SystemExit) as caught:
        prefix(monkeypatch, ['--experiment', 'chair021', '--object-height-penalty-weight', '1'])
    assert caught.value.code == 2


def test_requested_configuration_preserves_all_other_components(monkeypatch):
    with pytest.raises(BeforeSimulator) as caught:
        prefix(monkeypatch, ['--reward-variant', 'interaction_mesh', '--interaction-reference', str(GRAPH),
                            '--object-z-error-weight', '2', '--object-height-penalty-weight', '1',
                            '--object-height-penalty-scale', '.05'])
    state = caught.value.args[0]
    cfg, old = state['CONFIG'], g1_29dof_core4d_smalltable_baseline
    for field in ('robot', 'command', 'observation', 'action', 'algo', 'simulator', 'termination', 'randomization'):
        assert getattr(cfg, field) is getattr(old, field)
    assert cfg.reward.only_positive_rewards is False
    assert len(cfg.reward.terms) == len(old.reward.terms) + 2
    height = cfg.reward.terms['object_height_error_penalty']
    assert height.weight == -1.0 and height.params == {'scale_m': .05}
    assert cfg.reward.terms['interaction_mesh'].weight == 1
    position = cfg.reward.terms['object_global_ref_position_error_exp']
    assert position.params == {'sigma': .3, 'object_z_error_weight': 2.0}
    assert position.weight == 1
    for name, term in old.reward.terms.items():
        if name != 'object_global_ref_position_error_exp':
            assert cfg.reward.terms[name] == term
    assert 'height1_scale0.05' in state['ARGS'].output_dir.name
    assert state['ARGS'].resume is None


def height_metrics_function():
    path = ROOT / 'scripts/evaluate_core4d_smalltable.py'
    tree = ast.parse(path.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_episode_height_metrics')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_episode_height_metrics']


def test_each_episode_has_signed_bias_and_nonnegative_height_rmse():
    metrics = height_metrics_function()({'object_height_error_m': np.array([-.05, .05, 0.])})
    assert metrics['object_height_bias_m'] == pytest.approx(0)
    assert metrics['object_height_rmse_m'] == pytest.approx(np.sqrt(.005/3))
    assert metrics['object_height_abs_error_max_m'] == pytest.approx(.05)


@pytest.mark.parametrize('errors', [np.array([]), np.array([np.nan]), np.array([[0.1]])])
def test_invalid_height_samples_rejected(errors):
    with pytest.raises(ValueError):
        height_metrics_function()({'object_height_error_m': errors})
