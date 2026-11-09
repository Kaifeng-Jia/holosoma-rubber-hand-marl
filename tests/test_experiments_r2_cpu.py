"""CPU-only checks; requires the existing local R2 evidence for integration tests."""
import csv
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_experiments_r2_cpu import BatchFK,quat_error,read
from collect_experiments_r2_geometry import finite_distance
OUT=ROOT/'paper_handoff_exports/EXPERIMENTS_R2_20261024_CPU_v2'

def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


class CPUChecks(unittest.TestCase):
    def test_fk_independent(self):
        self.assertTrue(BatchFK().validate()['passed'])

    def test_quaternion_sign(self):
        q=np.array([[.1,.2,.3,.9]])
        self.assertLess(quat_error(q,-q)['max_rad'],1e-12)

    def test_finite_surface_not_infinite_plane(self):
        f=[dict(patches=[dict(lo=[-1,-1,0],hi=[1,1,0])])]
        d=finite_distance(np.array([[0,0,1],[2,0,0],[1.1,0,0]]),f)
        np.testing.assert_allclose(d,[1,1,.1])

    def test_push_preflight_all_hooks(self):
        p=read(OUT/'preflight_push_registered.json')
        self.assertEqual(p['original_diagnostic_status'],'diagnostic_only')
        self.assertAlmostEqual(p['lateral_spacing_m'],.8780472,places=12)
        self.assertEqual(set(p['hooks']),{'setup_terms','reset_terms','step_terms'})
        self.assertTrue(all(p['invariant_channels_bitwise_equal'].values()))
        self.assertFalse(p['offline_runtime_used_for_training'])

    def test_tracking_summary_independent(self):
        rows=csv_rows(OUT/'body_tracking_episodes.csv')
        self.assertEqual(len(rows),1050)
        self.assertEqual(len({(r['model_id'],r['state_id']) for r in rows}),1050)
        for summary in csv_rows(OUT/'body_tracking_summary.csv'):
            selected=[r for r in rows if r['configuration']==summary['configuration'] and
                      (summary['scope']=='prefix' or r['completed']=='True')]
            self.assertEqual(len(selected),int(summary['episodes']))
            for field in ['E_body_cm','E_hand_cm']:
                means=[np.mean([float(r[field]) for r in selected if r['seed']==s]) for s in sorted({r['seed'] for r in selected})]
                self.assertAlmostEqual(float(summary[field+'_mean']),float(np.mean(means)),places=10)
                self.assertAlmostEqual(float(summary[field+'_sample_sd']),float(np.std(means,ddof=1)),places=10)

    def test_u1_aggregation(self):
        details=csv_rows(OUT/'ref_role_synthesis_by_hand.csv')
        for r in csv_rows(OUT/'ref_role_synthesis.csv'):
            selected=[x for x in details if all(x[k]==r[k] for k in ['task','role','placement(registered|prescribed)'])]
            self.assertEqual(len(selected),2)
            self.assertAlmostEqual(float(r['dd_mean_mm']),np.mean([float(x['dd_mean_mm']) for x in selected]),places=10)
            self.assertEqual(int(r['n_contact_frames']),sum(int(x['n_contact_frames']) for x in selected))


if __name__=='__main__':unittest.main()
