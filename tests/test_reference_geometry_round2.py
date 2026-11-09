import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_geometry_math import BoxUnion
from reference_geometry_round2_correspondence import clipped_faces,tie_safe_weights,construct,fit_candidates
from reference_geometry_pilot import DATA,ROOT
from reference_geometry_round2_checks import hull_box_sat,obb_sat
import trimesh
import json


class Round2Tests(unittest.TestCase):
    def test_signed_separation_and_penetration(self):
        hull=trimesh.creation.box([1,1,1]);edges=np.diff(hull.vertices[hull.edges_unique],axis=1)[:,0]
        for shift,expected in [(1.1,.1),(.9,-.1),(0.,-1.)]:
            self.assertAlmostEqual(hull_box_sat(hull.vertices,hull.face_normals,edges,np.array([shift,0,0]),np.ones(3)*.5,np.eye(3)),expected)
            self.assertAlmostEqual(obb_sat(np.zeros(3),np.eye(3),np.ones(3)*.5,np.array([shift,0,0]),np.eye(3),np.ones(3)*.5),expected)
    def test_sat_rigid_transform_invariance(self):
        from scipy.spatial.transform import Rotation
        r=Rotation.from_euler('xyz',[.3,.4,.5]).as_matrix();c=np.array([3.,4.,5.]);h=trimesh.creation.box([1,1,1]);e=np.diff(h.vertices[h.edges_unique],axis=1)[:,0]
        d=hull_box_sat(h.vertices@r.T+c,h.face_normals@r.T,e@r.T,r@np.array([1.1,0,0])+c,np.ones(3)*.5,r)
        self.assertAlmostEqual(d,.1)
    def test_clipping_does_not_create_wall(self):
        g=BoxUnion(DATA/'objects_widetable_plan5_training.urdf')
        fs=clipped_faces(g,0,[-.7,-.1780472]);table=[f for f in fs if 'tabletop' in f['id']]
        self.assertEqual(len(table),5)
        self.assertFalse(any(np.array_equal(f['normal'],[1,0,0]) for f in table))
        self.assertTrue(any(np.array_equal(f['normal'],[-1,0,0]) for f in table))
        for f in table:self.assertGreater(f['area_m2'],0)
    def test_tie_mass_is_not_duplicated(self):
        windows=[dict(start=0,end_exclusive=5,face_index=i,anchors=np.eye(3),face_id=str(i)) for i in range(2)]
        interaction=dict(records=[dict(hand='left',stable_face_windows=windows),dict(hand='right',stable_face_windows=windows[:1])])
        curves={}
        for h in ['left','right']:
            curves[h+'_competing_face_bits']=np.full((5,10),3,dtype=np.uint64)
            curves[h+'_signed_distance_m']=np.zeros((5,10))
        w=tie_safe_weights(interaction,curves)
        np.testing.assert_allclose([a['normalized_weight'] for a in w['windows']],[.25,.25,.5])
        self.assertEqual(w['hand_summaries'][0]['unique_support_samples'],50)
        self.assertEqual(w['hand_summaries'][0]['unique_active_frames'],5)
    def test_actual_full_window_correspondence(self):
        base=ROOT/'logs/ReferenceGeometry/reference_geometry_pilot_20261017'
        for case in ['push','pull']:
            m=json.loads((base/case/'input_manifest.json').read_text())
            interaction=json.loads((base/case/'source_interactions.json').read_text())
            curves=np.load(base/case/'source_distances.npz')
            w=tie_safe_weights(interaction,curves);s=construct(case,BoxUnion(m['source_object']['path']),BoxUnion(m['target_object']['path']),w)
            self.assertEqual(len(w['windows']),sum(len(r['stable_face_windows']) for r in interaction['records']))
            self.assertTrue(all(n>0 for n in s['accepted_per_role']))
            for c in fit_candidates(s):
                self.assertLess(c['fit']['weighted_rmse_m'],1e-10)
                self.assertAlmostEqual(np.linalg.det(c['fit']['Q']),c['parity'])
                for f in c['face_map'].values():np.testing.assert_allclose(c['fit']['Q']@f['source_normal'],f['target_normal'],atol=1e-10)

if __name__=='__main__':unittest.main()
