"""CPU-only analytical checks for the frozen U2 measurement contract."""
import sys,unittest
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import measure_u2_shared_scene_handdist_cpu as m

class U2Tests(unittest.TestCase):
    def test_triangle_interior_not_vertex_distance(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[2,0,0],[0,2,0]],faces=[[0,1,2]],process=False)
        np.testing.assert_allclose(m.exact_distance(mesh,[[.5,.5,1]]),[1],atol=1e-12)
    def test_internal_distance_not_clamped(self):
        mesh=trimesh.creation.box(extents=[2,2,2]);p=np.array([[0,0,0],[2,0,0],[1,0,0],[.9,0,0]])
        d=m.exact_distance(mesh,p);np.testing.assert_allclose(d,[1,1,0,.1],atol=1e-12)
        flags,info=m.inside_flags(mesh,p,d);np.testing.assert_array_equal(flags,[1,0,0,1]);self.assertEqual(info['unknown'],0)
    def test_open_mesh_inside_unknown(self):
        mesh=trimesh.Trimesh(vertices=[[0,0,0],[2,0,0],[0,2,0]],faces=[[0,1,2]],process=False)
        flags,_=m.inside_flags(mesh,np.array([[.5,.5,1]]),np.array([1.]));self.assertEqual(flags[0],-1)
    def test_boundary_world_tolerance(self):
        mesh=trimesh.creation.box(extents=[2,2,2]);p=np.array([[1-5e-9,0,0]])
        flags,info=m.inside_flags(mesh,p,m.exact_distance(mesh,p));self.assertEqual(flags[0],0);self.assertEqual(info['boundary_count'],1)
    def test_scale_and_rotated_world_equivalence(self):
        mesh=trimesh.creation.box(extents=[2,2,2]);r=Rotation.from_euler('xyz',[.3,-.4,.9]);s=.74;p=np.array([4,-3,2])
        local=np.array([[.5,.5,2],[0,0,0]]);world=local@r.as_matrix().T*s+p
        q=np.r_[p,r.as_quat()[[3,0,1,2]]][None]
        restored=m.local_points(world[None],q,s)[0];np.testing.assert_allclose(restored,local,atol=1e-12)
        transformed=trimesh.Trimesh(vertices=mesh.vertices@r.as_matrix().T*s+p,faces=mesh.faces,process=False)
        np.testing.assert_allclose(s*m.exact_distance(mesh,restored),m.brute_distance(transformed,world),atol=1e-12)
    def test_empty_and_unknown_inside(self):
        row=m.stat_row('t','v',0,'left',np.array([.1]),np.array([1]),np.array([False]));self.assertEqual(row['n_window_frames'],0);self.assertEqual(row['dist_mean_mm'],'--')
        row=m.stat_row('t','v',0,'left',np.array([.1]),np.array([-1]),np.array([True]));self.assertEqual(row['dist_mean_mm'],100);self.assertEqual(row['frac_inside'],'--')
    def test_summary_linear_quantile(self):
        row=m.stat_row('t','v',0,'right',np.array([.001,.002,.009,.9]),np.array([1,0,0,1]),np.array([True,True,True,False]))
        self.assertEqual(row['n_window_frames'],3);self.assertAlmostEqual(row['dist_mean_mm'],4);self.assertAlmostEqual(row['dist_median_mm'],2);self.assertAlmostEqual(row['dist_p90_mm'],7.6);self.assertAlmostEqual(row['frac_inside'],1/3)
    def test_fk_unchanged_and_hand_points(self):
        fk,source=m.make_fk();self.assertTrue(fk.validate()['passed']);self.assertEqual(len(source['joint_names']),29)
        self.assertEqual(m.PALM,[[20,25,28,31],[21,40,43,46]])

if __name__=='__main__':unittest.main()
