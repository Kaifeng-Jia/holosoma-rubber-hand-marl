"""CPU-only checks of the offline audit CLI and URDF geometry support."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_core4d_interaction_vectors.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("offline_vector_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def _check_mesh_scale_and_origin_are_applied(tmp_path):
    mesh_path = tmp_path / "triangle.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    urdf = tmp_path / "object.urdf"
    urdf.write_text('<robot><link name="object"><collision><origin xyz="1 2 3" '
                    'rpy="0 0 1.5707963267948966"/><geometry><mesh filename="triangle.obj" '
                    'scale="2 3 4"/></geometry></collision></link></robot>')
    mesh, metadata = audit.load_single_mesh(urdf)
    expected = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]])
    expected = expected @ Rotation.from_euler("z", np.pi / 2).as_matrix().T + [1, 2, 3]
    np.testing.assert_allclose(mesh.vertices, expected, atol=1e-15)
    assert metadata["geometry_source"] == "collision"
    assert metadata["mesh_sha256"] == audit.digest(mesh_path)
    assert np.isclose(mesh.area, 3.0)


def _check_unsupported_geometry_is_rejected(tmp_path, xml):
    urdf = tmp_path / "object.urdf"
    urdf.write_text(xml)
    try:
        audit.load_single_mesh(urdf)
    except ValueError:
        return
    raise AssertionError("Unsupported geometry accepted")


def _check_existing_output_is_untouched(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")
    try:
        audit.main(["--input", "does-not-exist.npz", "--output-dir", str(tmp_path)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("Existing output was not rejected")
    assert marker.read_text() == "unchanged"


def _check_fixed_sampler_deterministic():
    mesh = audit.trimesh.creation.icosphere(subdivisions=1)
    metadata = {"components": [{"name": "test"}], "face_component_ids": [0] * len(mesh.faces),
                "face_axis_labels": ["+x"] * len(mesh.faces), "vertices_faces_sha256": "test"}
    first, first_meta = audit.sample_object_points(mesh, metadata, 100, 42)
    second, second_meta = audit.sample_object_points(mesh, metadata, 100, 42)
    np.testing.assert_array_equal(first, second)
    assert first_meta["points_sha256"] == second_meta["points_sha256"]


class Checks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vector-audit-check-")
        self.path = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_transform(self):
        _check_mesh_scale_and_origin_are_applied(self.path)

    def test_box_rejected(self):
        _check_unsupported_geometry_is_rejected(self.path, '<robot><link><collision><geometry><box size="1 1 1"/></geometry></collision></link></robot>')

    def test_multilink_rejected(self):
        _check_unsupported_geometry_is_rejected(self.path, '<robot><link/><link/></robot>')

    def test_compound_rejected(self):
        _check_unsupported_geometry_is_rejected(self.path, '<robot><link><collision/><collision/></link></robot>')

    def test_no_overwrite(self):
        _check_existing_output_is_untouched(self.path)

    def test_sampling(self):
        _check_fixed_sampler_deterministic()


if __name__ == "__main__":
    unittest.main()
