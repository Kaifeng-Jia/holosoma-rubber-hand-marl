"""CPU-only invariant tests; no renderer, policy, solver or physics steps."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from render_wrist_reference_pairs import validate_pair, WRIST_COLUMNS


class FrozenPairTests(unittest.TestCase):
    def setUp(self):
        self.robot = np.zeros((3, 1, 36))
        self.obj = np.zeros((3, 7))
        self.raw = {'fps': np.array(30), 'human_joints': np.zeros((3, 52, 3))}
        self.a = (self.robot.copy(), self.obj.copy(), self.raw.copy())
        self.b = (self.robot.copy(), self.obj.copy(), self.raw.copy())
        self.b[0][:, :, WRIST_COLUMNS] += .1
        self.spec = {'name': 'push', 'robot_shape': [3, 1, 36]}

    def test_wrist_only_allowed(self):
        self.assertEqual(validate_pair(self.a, self.b, self.spec)['changed_qpos_columns'], WRIST_COLUMNS)

    def test_non_wrist_rejected(self):
        self.b[0][1, 0, 8] += .1
        with self.assertRaises(AssertionError): validate_pair(self.a, self.b, self.spec)

    def test_object_change_rejected(self):
        self.b[1][1, 0] += .1
        with self.assertRaises(AssertionError): validate_pair(self.a, self.b, self.spec)

    def test_runtime_fps_rejected(self):
        self.b[2]['fps'] = np.array(50)
        with self.assertRaises(AssertionError): validate_pair(self.a, self.b, self.spec)

    def test_chair_scale_rejected(self):
        self.spec['name'] = 'chair'
        for key in ('human_heights','human_to_robot_scales','scale_anchor','shared_object_scale'):
            self.a[2][key] = self.b[2][key] = np.array(.744)
        self.b[2]['shared_object_scale'] = np.array(.8)
        with self.assertRaises(AssertionError): validate_pair(self.a, self.b, self.spec)


if __name__ == '__main__': unittest.main()
