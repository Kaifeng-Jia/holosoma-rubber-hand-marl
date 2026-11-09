"""Object-mass variants for the T04 mass sweep.

Only the object's mass changes. The inertia tensor is scaled by the same ratio
(geometry, COM, collision shapes, friction and references are untouched), so a
variant URDF differs from its source only in the ``<mass value>`` attribute and
the six ``<inertia>`` attributes.

Generated URDFs live next to the source asset with the suffix
``__system1_mass<m>kg.urdf`` so relative asset paths keep working.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "holosoma"
_INERTIA_KEYS = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
_MASS_RE = re.compile(r'(<mass\s+value=")([^"]+)(")')
_INERTIA_RE = re.compile(r"<inertia\b[^>]*/>", re.S)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mass_tag(mass_kg: float) -> str:
    return f"{mass_kg:g}".replace(".", "p")


def _fmt(x: float) -> str:
    return repr(float(x))


def read_mass_inertia(text: str) -> tuple[float, dict[str, float]]:
    masses = _MASS_RE.findall(text)
    inertias = _INERTIA_RE.findall(text)
    if len(masses) != 1 or len(inertias) != 1:
        raise ValueError(f"Expected exactly one <mass> and one <inertia>, got {len(masses)} / {len(inertias)}")
    values = {k: float(re.search(rf'{k}="([^"]+)"', inertias[0]).group(1)) for k in _INERTIA_KEYS}
    return float(masses[0][1]), values


def scaled_urdf_text(text: str, mass_kg: float) -> tuple[str, dict]:
    m0, inertia0 = read_mass_inertia(text)
    ratio = mass_kg / m0
    out = _MASS_RE.sub(lambda m: m.group(1) + _fmt(mass_kg) + m.group(3), text)

    def scale_inertia(match: re.Match) -> str:
        block = match.group(0)
        for k in _INERTIA_KEYS:
            block = re.sub(rf'({k}=")([^"]+)(")', lambda mm: mm.group(1) + _fmt(float(mm.group(2)) * ratio) + mm.group(3), block)
        return block

    out = _INERTIA_RE.sub(scale_inertia, out)
    m1, inertia1 = read_mass_inertia(out)
    record = dict(
        source_mass_kg=m0,
        mass_kg=m1,
        ratio=ratio,
        source_inertia=inertia0,
        inertia=inertia1,
    )
    return out, record


def variant_relpath(source_rel: str, mass_kg: float) -> str:
    p = Path(source_rel)
    return str(p.with_name(f"{p.stem}__system1_mass{mass_tag(mass_kg)}kg{p.suffix}"))


def ensure_mass_variant(source_rel: str, mass_kg: float) -> tuple[str, dict]:
    """Create (or verify) the variant URDF; return its package-relative path and a record."""
    src = PACKAGE_ROOT / source_rel
    text, record = scaled_urdf_text(src.read_text(), mass_kg)
    rel = variant_relpath(source_rel, mass_kg)
    dst = PACKAGE_ROOT / rel
    if dst.exists():
        if dst.read_text() != text:
            raise RuntimeError(f"Existing mass variant differs from the deterministic rebuild: {dst}")
    else:
        dst.write_text(text)
    record.update(source_urdf=source_rel, source_urdf_sha256=sha256(src), urdf=rel, urdf_sha256=sha256(dst))
    return rel, record


def scaled_inertia_row_major(inertia_row_major, ratio: float):
    return tuple(float(v) * ratio for v in inertia_row_major)


def core4d_experiment_with_mass(experiment, mass_kg: float):
    """Same CORE4D pair experiment with only the object URDF/mass replaced.

    ``experiment_id`` is kept so every id-keyed branch (reward contract, 5 kg table
    checks) behaves exactly as for the source experiment. The source URDF hash is
    kept in ``source_object_urdf_sha256`` for checks that compare against assets
    frozen with the original geometry (e.g. the bucket interaction artifact).
    """
    from dataclasses import dataclass, fields

    from holosoma.config_values.marl.g1.core4d_pair_experiments import Core4DPairExperiment

    @dataclass(frozen=True)
    class Core4DPairExperimentMassVariant(Core4DPairExperiment):
        source_object_urdf_sha256: str = ""
        system1_mass_variant: bool = True

    rel, record = ensure_mass_variant(experiment.object_urdf_file, mass_kg)
    if abs(record["source_mass_kg"] - experiment.object_mass_kg) > 1e-9:
        raise ValueError("Experiment mass does not match its URDF")
    if record["source_urdf_sha256"] != experiment.object_urdf_sha256:
        raise ValueError("Experiment URDF hash does not match the source asset")
    values = {f.name: getattr(experiment, f.name) for f in fields(Core4DPairExperiment)}
    values.update(
        object_urdf_file=rel,
        object_urdf_sha256=record["urdf_sha256"],
        object_mass_kg=float(mass_kg),
        source_object_urdf_sha256=experiment.object_urdf_sha256,
    )
    return Core4DPairExperimentMassVariant(**values), record
