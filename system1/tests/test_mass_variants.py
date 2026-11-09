"""Smoke tests for system1.mass_variants (stdlib only; does not write into the package)."""
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from system1 import mass_variants as mv  # noqa: E402

PULL = "holosoma/data/motions/g1_29dof/whole_body_tracking/objects_widetable_plan5_pull_training.urdf"
TABLE = "holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_smalltable5kg_A/desk001_5kg_training.urdf"


def _check(rel, m0, mass):
    text = (mv.PACKAGE_ROOT / rel).read_text()
    out, rec = mv.scaled_urdf_text(text, mass)
    assert rec["source_mass_kg"] == m0 and rec["mass_kg"] == mass
    for k, v in rec["source_inertia"].items():
        assert abs(rec["inertia"][k] - v * mass / m0) <= 1e-12 * max(1.0, abs(v)), k
    changed = [l for l in difflib.ndiff(text.splitlines(), out.splitlines()) if l[:1] in "+-"]
    for line in changed:
        assert "mass" in line or any(f'{k}="' in line for k in mv._INERTIA_KEYS), line
    same, _ = mv.scaled_urdf_text(text, m0)
    assert mv.read_mass_inertia(same) == mv.read_mass_inertia(text)


def test_pull_scaling():
    for m in (5.0, 10.0, 40.0):
        _check(PULL, 20.0, m)


def test_table_scaling():
    for m in (2.5, 10.0, 20.0):
        _check(TABLE, 5.0, m)


def test_paths():
    assert mv.variant_relpath(PULL, 2.5).endswith("objects_widetable_plan5_pull_training__system1_mass2p5kg.urdf")
    assert mv.mass_tag(40.0) == "40"


if __name__ == "__main__":
    test_pull_scaling(); test_table_scaling(); test_paths(); print("ok")
