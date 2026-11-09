"""CPU-only tests: importing this module cannot launch cloud jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('autorun', Path(__file__).resolve().parents[1]/'scripts/autodl_supplement_autorun.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

class AutoRunTests(unittest.TestCase):
    def test_gpu_cache_paths_are_isolated_without_distributed_training(self):
        for gpu in (0,1):
            env=runner.environment(gpu)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],str(gpu))
            self.assertEqual(env['LOCAL_RANK'],str(gpu))
            self.assertEqual(env['WORLD_SIZE'],'1')

    def test_dependency_exception_is_exact_and_exclusive(self):
        known = 'isaacsim-kernel 5.1.0.0 has requirement typing_extensions==4.12.2, but you have typing-extensions 4.16.0.'
        self.assertTrue(runner.acceptable_pip_check(1, known))
        self.assertTrue(runner.acceptable_pip_check(0, 'No broken requirements found.'))
        self.assertFalse(runner.acceptable_pip_check(0, known))
        self.assertFalse(runner.acceptable_pip_check(1, known+'\nMissing package'))
        self.assertFalse(runner.acceptable_pip_check(1, known.replace('4.16.0', '4.15.0')))

    def test_formal_command_is_fresh_and_bounded(self):
        command = runner.command(Path('/example/output'), 722, True)
        for key, value in {'--iterations':'12000', '--num-envs':'2048', '--steps-per-env':'24',
                           '--save-interval':'2000', '--seed':'722'}.items():
            self.assertEqual(command[command.index(key)+1], value)
        self.assertNotIn('--resume', command)

    def fixture(self, folder):
        cfg = dict(experiment='smalltable5kg_A', reward_variant=runner.VARIANT, num_envs=2048,
                   iterations_requested=2, steps_per_env=24, seed=722, save_interval=2,
                   initialization='fresh', start_iteration=0, actor_input_dim=158, critic_input_dim=527,
                   bucket_reward_contract=dict(block_weights_position_rotation_height_relation=[1,1,0,0],
                                               position_error_weights_xyz=[1,1,2]))
        (folder/'run_config.json').write_text(json.dumps(cfg))
        (folder/'status.json').write_text(json.dumps(dict(passed=True, final_iteration=2)))
        (folder/'metrics.jsonl').write_text('\n'.join(json.dumps({'iteration':i,'Bucket/gate_mean':1}) for i in (1,2)))
        (folder/'model_00002.pt').write_bytes(b'test-only-not-a-real-model')

    def test_validation_rejects_missing_or_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); self.fixture(folder)
            self.assertEqual(runner.validate_run(folder,722,2)['final_iteration'],2)
            with self.assertRaises(RuntimeError):
                runner.validate_run(folder,723,2)
            (folder/'metrics.jsonl').write_text('{"iteration":1,"Bucket/gate_mean":1}\n')
            with self.assertRaises(RuntimeError):
                runner.validate_run(folder,722,2)
            (folder/'metrics.jsonl').write_text('\n'.join(json.dumps({'iteration':i,'Bucket/gate_mean':1,'loss':float('nan')}) for i in (1,2)))
            with self.assertRaises(RuntimeError):
                runner.validate_run(folder,722,2)

    def test_preflight_success_is_not_early_policy_quality_gate(self):
        self.assertTrue(runner.finite({'reward':-100000.0, 'completion':0.0}))
        self.assertFalse(runner.finite({'nested':[float('inf')]}))

if __name__ == '__main__':
    unittest.main()
