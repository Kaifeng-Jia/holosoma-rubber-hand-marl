import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/holosoma')]
from plan5_registered_placement import configure_push_placement, comparison_body_indexes, export_runtime_reference
from holosoma.config_values.marl.g1.experiment import g1_29dof_plan5_push_baseline as BASE

PROOF=ROOT/'paper_handoff_exports/EXPERIMENTS_R2_20261024_CPU_v2/preflight_push_registered.json'

class PlacementTests(unittest.TestCase):
    def test_runtime_configured_body_list_export(self):
        from types import SimpleNamespace
        from prepare_experiments_r2_cpu import CaptureSimulator, PairedA1MotionCommand
        cfg, evidence=configure_push_placement(BASE,PROOF,root=ROOT)
        # Reproduce the real runtime list, rather than the expanded offline list.
        names=list(cfg.robot.body_names)
        self.assertNotIn('left_rubber_hand_link',names)
        env=SimpleNamespace(num_envs=1,device='cpu',simulator=CaptureSimulator(names),is_evaluating=True)
        command=PairedA1MotionCommand(cfg.command.setup_terms['paired_motion_command'],env)
        command.setup()
        env.command_manager=SimpleNamespace(get_state=lambda _:command)
        with tempfile.TemporaryDirectory() as d:
            result=export_runtime_reference(env,d,evidence,PROOF)
        self.assertEqual(len(result['compared_body_names']),14)
        self.assertEqual(set(result['offline_only_body_names']),{'left_rubber_hand_link','right_rubber_hand_link'})
        self.assertTrue(all(x['max_abs']==0 for x in result['comparison'].values()))

    def test_missing_required_body_is_still_rejected(self):
        with self.assertRaises(ValueError):comparison_body_indexes(['pelvis'],['pelvis','left_knee_link'],['pelvis','left_knee_link'])

    def test_default_identity(self):
        cfg, evidence=configure_push_placement(BASE)
        self.assertIs(cfg,BASE)
        self.assertIsNone(evidence)

    def test_only_three_spacing_params_change(self):
        before=asdict(BASE)
        cfg,evidence=configure_push_placement(BASE,PROOF,root=ROOT)
        expected=copy.deepcopy(before)
        for name in ('setup_terms','reset_terms','step_terms'):
            expected['command'][name]['paired_motion_command']['params']['lateral_spacing_m']=evidence['lateral_spacing_m']
        self.assertEqual(asdict(cfg),expected)
        self.assertEqual(asdict(BASE),before)
        self.assertAlmostEqual(evidence['lateral_spacing_m'],.8780472,places=14)

    def test_reject_non_push(self):
        with self.assertRaises(ValueError):configure_push_placement(BASE,PROOF,skill='kick')

    def test_reject_nonrepresentable_rotation(self):
        proof=json.loads(PROOF.read_text());proof['role_parameters'][0]['Q'][0][0]=-1
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'proof.json';path.write_text(json.dumps(proof))
            with self.assertRaises(ValueError):configure_push_placement(BASE,path)

if __name__=='__main__':unittest.main()
