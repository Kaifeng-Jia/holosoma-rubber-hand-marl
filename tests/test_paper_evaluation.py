"""CPU-only checks: real perturbation and absence of invented successful seeds."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_evaluation_utils import make_states, yaw_transform, metrics, evaluation_result_dir
from summarize_paper_evaluation import summarize
from organize_paper_evaluation import portable


class EvaluationProtocolTests(unittest.TestCase):
    def test_delivered_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'PAPER_SUPPLEMENT_EVALUATION.tar.gz'
            archive.write_bytes(b'previous delivery')
            with self.assertRaises(FileExistsError):
                portable(root)
            self.assertEqual(archive.read_bytes(), b'previous delivery')
            self.assertFalse((root / 'portable').exists())

    def test_package_name_must_stay_in_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../outside.tar.gz', '/tmp/outside.tar.gz', 'not_an_archive.zip'):
                with self.assertRaises(ValueError):
                    portable(Path(directory), name)

    def test_retry_selection_is_explicit_and_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            m=dict(model_id='T-07_seed721',evaluation_directory='results/T-07_seed721_retry01')
            self.assertEqual(evaluation_result_dir(root,m),root/'results/T-07_seed721_retry01')
            m['evaluation_directory']='../escape'
            with self.assertRaises(ValueError):evaluation_result_dir(root,m)

    def test_states_fixed_and_bounded(self):
        a = make_states('bucket')
        self.assertEqual(a, make_states('bucket'))
        self.assertEqual(len(a['states']), 51)
        self.assertEqual(len({s['state_id'] for s in a['states']}), 51)
        for s in a['states']:
            self.assertLessEqual(np.abs(s['robot_xy_m']).max(), .02)
            self.assertLessEqual(np.abs(s['robot_yaw_rad']).max(), np.deg2rad(2))

    def test_yaw_preserves_height_and_rotates_velocities(self):
        r = np.zeros((2, 13)); r[:, 2] = 1; r[:, 6] = 1; r[:, 7] = 1
        result = yaw_transform(r, [[.01, 0], [0, -.02]], [np.pi / 2, 0])
        np.testing.assert_allclose(result[:, 2], 1)
        np.testing.assert_allclose(result[0, 7:9], [0, 1], atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1)
        np.testing.assert_allclose(r[:, :2], 0)

    def test_rmse_not_mean_distance(self):
        m = metrics({'object_pos_w': np.array([[0., 0, 0], [0, 0, 2]]), 'reference_object_pos_w': np.zeros((2, 3))})
        self.assertAlmostEqual(m['object_rmse_3d_m'], np.sqrt(2))

    def test_pending_seed_excluded_and_zero_completion_not_zero_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); models = []
            for seed in (721, 722, 723):
                m = dict(model_id=f'B-RZ0_seed{seed}', configuration='B-RZ0', training_seed=seed,
                         reuse_status='source_audit_pending' if seed == 721 else 'new_training_verified', checkpoint_sha256='test')
                models.append(m); dest = root / 'results' / m['model_id']; dest.mkdir(parents=True)
                episodes = [dict(state_id='nominal' if i == 0 else f'perturbed_{i-1:03}', kind='nominal' if i == 0 else 'perturbed',
                                 terminal_terms={}, completed_reference=False, progress=.5, object_rmse_3d_m=.1) for i in range(51)]
                (dest / 'summary.json').write_text(json.dumps(dict(complete=True, episode_count=51, episodes=episodes)))
                (root / (m['model_id']+'.exit.json')).write_text(json.dumps(dict(validated=True, process_returncode=0)))
            (root / 'models.json').write_text(json.dumps(models))
            self.assertEqual(summarize(root), 3)
            import csv
            with (root / 'main_results.csv').open() as f:
                r = next(csv.DictReader(f))
            self.assertEqual(r['training_seeds_evaluated'], '2')
            self.assertEqual(r['completed_object_rmse_3d_m_mean'], '')
            self.assertEqual(r['complete_three_seed_result'], 'False')
            # Old attempt says success, but only explicitly selected retry is scored.
            selected=next(m for m in models if m['training_seed']==722)
            original=root/'results'/selected['model_id']
            retry=root/'results'/(selected['model_id']+'_retry01');retry.mkdir()
            (retry/'summary.json').write_text((original/'summary.json').read_text())
            old=json.loads((original/'summary.json').read_text())
            for episode in old['episodes']:episode['completed_reference']=True
            (original/'summary.json').write_text(json.dumps(old))
            selected['evaluation_directory']='results/'+retry.name
            (root/'models.json').write_text(json.dumps(models))
            self.assertEqual(summarize(root),3)
            with (root/'per_seed_summary.csv').open() as f:
                seed722=next(row for row in csv.DictReader(f) if row['training_seed']=='722')
            self.assertEqual(seed722['completed_episodes'],'0')
            # A complete summary does not override evidence of abnormal process exit.
            (root / 'B-RZ0_seed723.exit.json').write_text(json.dumps(dict(validated=False, process_returncode=-11)))
            self.assertEqual(summarize(root), 2)
            with (root / 'main_results.csv').open() as f:
                self.assertEqual(next(csv.DictReader(f))['training_seeds_evaluated'], '1')


if __name__ == '__main__': unittest.main()
