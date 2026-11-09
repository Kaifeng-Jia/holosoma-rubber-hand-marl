"""Synthetic solver/boundary checks, kept separate from real reference evidence."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reference_geometry_math import BoxUnion,procrustes,map_robot,continuous_windows


class GeometryPilotTests(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        self.w=np.array([1.,2.,3.,4.])
    def test_known_translation_planar(self):
        c=np.array([.4,-.1,.2]);r=procrustes(self.a,self.a+c,self.w,1)
        np.testing.assert_allclose(r['Q'],np.eye(3),atol=1e-12);np.testing.assert_allclose(r['c'],c,atol=1e-12)
    def test_weighted_rotation_translation(self):
        q=Rotation.from_rotvec([.3,-.5,.2]).as_matrix();c=np.array([.1,.2,.3]);r=procrustes(self.a,self.a@q.T+c,self.w,1)
        np.testing.assert_allclose(r['Q'],q,atol=1e-12);self.assertLess(r['weighted_rmse_m'],1e-12)
    def test_reflection_and_general_improper_not_involution(self):
        q=Rotation.from_euler('x',.7).as_matrix()@np.diag([-1.,1,1]);self.assertFalse(np.allclose(q@q,np.eye(3)))
        r=procrustes(self.a,self.a@q.T+[.1,.2,.3],self.w,-1)
        np.testing.assert_allclose(r['Q'],q,atol=1e-12);self.assertLess(r['inverse_error'],1e-12)
    def test_collinear_rejected(self):
        a=np.array([[0.,0,0],[1,0,0],[2,0,0]]);self.assertEqual(procrustes(a,a,np.ones(3),1)['status'],'degenerate')
    def test_scale_not_removed(self):
        self.assertGreater(procrustes(self.a,self.a*2,self.w,1)['weighted_rmse_m'],.4)
    def test_improper_map_proper_robot_orientation(self):
        qpos=np.zeros((3,36));qpos[:,3]=1;obj=np.zeros((3,7));obj[:,3]=1
        out=map_robot(qpos,obj,np.diag([1.,1,-1]),np.array([0,0,.4]),-1,np.arange(29),np.ones(29))
        np.testing.assert_allclose(np.linalg.det(Rotation.from_quat(out[:,[4,5,6,3]]).as_matrix()),1,atol=1e-12)
    def test_external_boundary_not_hidden_box_faces(self):
        xml='<robot name="u"><link name="o"><collision name="tabletop"><geometry><box size="2 2 2"/></geometry></collision><collision name="leg"><origin xyz="1 0 0"/><geometry><box size="2 2 2"/></geometry></collision></link></robot>'
        with tempfile.TemporaryDirectory(prefix='geometry_pilot_test_') as d:
            p=Path(d)/'obj.urdf';p.write_text(xml);g=BoxUnion(p)
            dist,signed,_,_,ties=g.distances(np.array([[1.,0,0],[3,2,0]]),with_ties=True)
            self.assertAlmostEqual(dist[0],1.);self.assertAlmostEqual(signed[0],-1.)
            self.assertGreater(int(ties[1]).bit_count(),1)
    def test_temporal_gap_and_duration(self):
        self.assertEqual(continuous_windows([0,1,1,0,1,1,0,0,1,0],5),[(1,6)])

if __name__=='__main__':unittest.main()
