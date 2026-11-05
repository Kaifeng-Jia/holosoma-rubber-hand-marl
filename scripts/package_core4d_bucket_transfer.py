#!/usr/bin/env python3
"""Package the CURRENT working-tree bucket A/B code/assets for a same-HEAD checkout.

Offline only: no upload, commit, installation, simulation, training, or repository
writes. Extract into a separate directory, verify SHA256SUMS, then overlay payload/
onto the clean checkout identified by package_manifest.json's base_git_head.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import subprocess
import tarfile
import xml.etree.ElementTree as ET


MAX_FILE_BYTES = 50 * 1024 * 1024
BUCKET_DIR = Path("src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/core4d_bucket003_20231020_071_a1")
PYTHON_PACKAGES = ("src/holosoma/holosoma", "src/holosoma_retargeting/holosoma_retargeting")
SCRIPTS = (
    "scripts/train_core4d_smalltable.py", "scripts/evaluate_core4d_smalltable.py",
    "scripts/smoke_core4d_smalltable.py", "scripts/prepare_core4d_bucket_training.py",
    "scripts/synthesize_core4d_smalltable_runtime_reference.py",
    "scripts/audit_core4d_interaction_vectors.py", "scripts/visualize_core4d_pair.py",
)
BUCKET_FILES = (
    "core4d_pair_runtime_fps50.npz", "core4d_pair_runtime_fps50.manifest.json",
    "interaction_vectors_v1.npz", "bucket003_m.obj", "bucket003_training.urdf",
    "training_asset_manifest.json", "core4d_pair_compact_fps30.npz",
    "source_manifest.json", "source_canonical.npz",
    "source_contact_diagnostics.json", "source_contact_diagnostics.npz",
)
TRAINING_ROBOT = "src/holosoma/holosoma/data/robots/g1/main_mesh_collision_rubberhand.urdf"
REFERENCE_ROBOT = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
REFERENCE_XML = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
FORBIDDEN_COMPONENTS = {".git", "logs", "checkpoints", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".venv", "node_modules"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def safe_relative(value: str | Path) -> str:
    name = str(value)
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or "\n" in name or "\r" in name
            or "\x00" in name or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe relative path: {name!r}")
    if set(path.parts) & FORBIDDEN_COMPONENTS or any(part.startswith("converted_rank") for part in path.parts):
        raise ValueError(f"Excluded environment/cache/log/checkpoint path: {name}")
    if path.suffix.lower() in (".usd", ".usda", ".usdc", ".pyc", ".pt", ".pth", ".ckpt"):
        raise ValueError(f"Excluded generated/cache/checkpoint file: {name}")
    return path.as_posix()


def safe_file(repo: Path, relative: str | Path) -> Path:
    relative = safe_relative(relative)
    path = repo / relative
    # Reject all symlinks (including internal ones), avoiding external references
    # and archive extraction surprises. No symlink is silently dereferenced.
    current = repo
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Symlink is not portable and cannot be packaged: {relative}")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise FileNotFoundError(f"Required regular file missing: {path}")
    if not path.resolve().is_relative_to(repo):
        raise ValueError(f"File escapes repository: {relative}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Individual file exceeds 50 MiB limit: {relative}")
    return path


def read_file(repo: Path, relative: str) -> bytes:
    path = safe_file(repo, relative)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError(f"File grew beyond 50 MiB limit: {relative}")
    return raw


def identity(repo: Path, relative: str) -> dict:
    raw = read_file(repo, relative)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def local_asset(repo: Path, owner: Path, filename: str, directory: str = "") -> str:
    if (not filename or "://" in filename or Path(filename).is_absolute()
            or "://" in directory or Path(directory).is_absolute()):
        raise ValueError(f"Nonportable asset reference in {owner}: {filename!r}")
    path = owner.parent / directory / filename
    # Lexically normalize '..' only after verifying its resolved target stays in
    # repo; safe_file still examines the final normalized path for symlinks.
    current = owner.parent
    for part in (Path(directory) / filename).parts:
        current = current.parent if part == ".." else current / part
        if current.is_symlink() or not current.is_relative_to(repo):
            raise ValueError(f"Asset dependency traverses a symlink or leaves repository: {filename!r}")
    if path.is_symlink() or not path.resolve().is_relative_to(repo):
        raise ValueError(f"Asset dependency escapes repository: {filename!r}")
    relative = os.path.relpath(os.path.abspath(path), repo)
    safe_file(repo, relative)
    return relative


def asset_dependencies(repo: Path, roots: tuple[str, ...]) -> set[str]:
    pending = list(roots)
    found: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = safe_file(repo, relative)
        found.add(relative)
        dependencies = []
        if path.suffix.lower() in (".urdf", ".xml"):
            root = ET.fromstring(read_file(repo, relative))
            if root.tag == "robot":
                dependencies.extend(local_asset(repo, path, item.get("filename", ""))
                                    for item in root.findall(".//mesh") + root.findall(".//texture")
                                    if item.get("filename") is not None)
            elif root.tag == "mujoco":
                compiler = root.find("compiler")
                values = {} if compiler is None else compiler.attrib
                for item in root.findall(".//mesh") + root.findall(".//texture"):
                    if item.get("file"):
                        directory = values.get("meshdir" if item.tag == "mesh" else "texturedir", values.get("assetdir", ""))
                        dependencies.append(local_asset(repo, path, item.get("file"), directory))
                dependencies.extend(local_asset(repo, path, item.get("file", "")) for item in root.findall(".//include"))
        elif path.suffix.lower() == ".obj":
            for line in read_file(repo, relative).decode("utf-8").splitlines():
                parts = shlex.split(line, comments=True)
                if parts and parts[0] == "mtllib":
                    dependencies.extend(local_asset(repo, path, name) for name in parts[1:])
        elif path.suffix.lower() == ".mtl":
            for line in read_file(repo, relative).decode("utf-8").splitlines():
                parts = shlex.split(line, comments=True)
                if parts and (parts[0].lower().startswith("map_") or parts[0].lower() in ("bump", "disp", "decal")):
                    if len(parts) != 2:
                        raise ValueError(f"Review texture options before packaging {relative}: {line}")
                    dependencies.append(local_asset(repo, path, parts[1]))
        pending.extend(dependencies)
    return found


def package_plan(repo: Path, handoff_doc: str, include_reference_robot: bool) -> tuple[dict, dict[str, dict]]:
    if git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise ValueError("--repo-root must be the Git working-tree root")
    base_head = git(repo, "rev-parse", "HEAD")
    selected: set[str] = set(SCRIPTS)
    for package in PYTHON_PACKAGES:
        package_root = repo / package
        if not package_root.is_dir():
            raise FileNotFoundError(package_root)
        for parent, directories, files in os.walk(package_root, followlinks=False):
            directories[:] = [name for name in directories if name not in FORBIDDEN_COMPONENTS and not name.startswith("converted_rank")]
            for name in directories:
                if (Path(parent) / name).is_symlink():
                    raise ValueError(f"Refusing symlink directory: {Path(parent) / name}")
            for name in files:
                if name.endswith(".py"):
                    selected.add((Path(parent) / name).relative_to(repo).as_posix())
    for package in ("src/holosoma", "src/holosoma_retargeting"):
        selected.add(f"{package}/pyproject.toml")
        for name in ("setup.py", "setup.cfg", "MANIFEST.in", "README.md"):
            if (repo / package / name).is_file():
                selected.add(f"{package}/{name}")
    tool_path = "scripts/package_core4d_bucket_transfer.py"
    if (repo / tool_path).exists():
        selected.add(tool_path)
    selected.add(safe_relative(handoff_doc))
    selected.update((BUCKET_DIR / name).as_posix() for name in BUCKET_FILES)
    training_dependencies = asset_dependencies(repo, (TRAINING_ROBOT,))
    reference_dependencies = asset_dependencies(repo, (REFERENCE_ROBOT, REFERENCE_XML))
    selected.update(training_dependencies)
    selected.update(asset_dependencies(repo, ((BUCKET_DIR / "bucket003_training.urdf").as_posix(),)))
    if include_reference_robot:
        selected.update(reference_dependencies)
    records = {name: identity(repo, name) for name in sorted(selected)}
    required_base = {name: identity(repo, name) for name in sorted(reference_dependencies - selected)}
    promotion = json.loads(read_file(repo, (BUCKET_DIR / "training_asset_manifest.json").as_posix()))
    asset_checks = {
        "runtime_reference_sha256": (BUCKET_DIR / "core4d_pair_runtime_fps50.npz").as_posix(),
        "interaction_artifact_sha256": (BUCKET_DIR / "interaction_vectors_v1.npz").as_posix(),
        "training_object_urdf_sha256": (BUCKET_DIR / "bucket003_training.urdf").as_posix(),
        "object_mesh_sha256": (BUCKET_DIR / "bucket003_m.obj").as_posix(),
        "source_pair_sha256": (BUCKET_DIR / "core4d_pair_compact_fps30.npz").as_posix(),
        "source_canonical_sha256": (BUCKET_DIR / "source_canonical.npz").as_posix(),
        "source_manifest_sha256": (BUCKET_DIR / "source_manifest.json").as_posix(),
        "training_robot_urdf_sha256": TRAINING_ROBOT,
    }
    for key, relative in asset_checks.items():
        if promotion.get(key) != records[relative]["sha256"]:
            raise ValueError(f"Training asset manifest hash mismatch: {key} / {relative}")
    runtime_manifest = json.loads(read_file(repo, (BUCKET_DIR / "core4d_pair_runtime_fps50.manifest.json").as_posix()))
    if runtime_manifest.get("frames") != 497 or runtime_manifest.get("fps") != 50:
        raise ValueError("Expected the prepared 497-frame, 50Hz runtime")
    if promotion.get("object_mass_kg") != 1.0 or promotion.get("object_collider_type") != "convex_decomposition":
        raise ValueError("Bucket package requires the recorded 1kg convex_decomposition asset")
    # Missing tracked source files must be removed on the receiving checkout;
    # tar overlays cannot encode deletions, so expose the requirement explicitly.
    tracked = git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", *PYTHON_PACKAGES).splitlines()
    required_absent = [safe_relative(name) for name in tracked if name.endswith(".py") and not (repo / name).exists()]
    manifest = {
        "format": "core4d_bucket_worktree_overlay_v1", "base_git_head": base_head,
        "source_capture": "current_working_tree_including_dirty_and_untracked_python_not_just_git_commit",
        "apply_contract": "Require clean checkout at base_git_head; verify required_base_files; apply required_absent_paths if any; overlay payload/ only.",
        "payload_prefix": "payload", "files": records,
        "source_files_sha256": {name: record["sha256"] for name, record in records.items() if name.endswith(".py") or Path(name).name == "pyproject.toml"},
        "payload_file_count": len(records), "payload_uncompressed_bytes": sum(record["size_bytes"] for record in records.values()),
        "max_individual_file_bytes": MAX_FILE_BYTES,
        "required_base_files": required_base, "required_absent_paths": required_absent,
        "training_robot_files": {name: records[name] for name in sorted(training_dependencies)},
        "reference_robot_files": {name: (records if name in records else required_base)[name] for name in sorted(reference_dependencies)},
        "include_reference_robot": include_reference_robot, "handoff_document": handoff_doc,
        "bucket_asset_hashes": {key: promotion[key] for key in asset_checks},
        "training_ready_as_packaged": promotion.get("training_ready"),
        "formal_training_authorized_as_packaged": promotion.get("formal_training_authorized"),
        "scope": "No logs, checkpoints, .git, cached converted_rank*/USD, Python environments, or full dataset. Only the selected canonical clip is included.",
        "external_requirements": "Existing compatible hssim/hsretargeting Python environments, Isaac Sim/IsaacLab and CUDA remain required; they are not collected or installed.",
        "asset_regeneration_scope": "Cloud train/eval/smoke consume the packaged prepared assets. The preparation script and copied source manifests retain original preview/dataset provenance paths; rerunning that original-source preparation is not promised by this dataset-free package.",
        "authorization": "Packaging does not authorize upload or training and runs neither.",
        "checksum_scope": "SHA256SUMS verifies every payload file plus package_manifest.json; archive.sha256 verifies the compressed archive.",
    }
    if git(repo, "rev-parse", "HEAD") != base_head:
        raise RuntimeError("Git HEAD changed during package planning")
    return manifest, records


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    safe_relative(name)
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = mode
    entry.mtime = 0
    entry.uid = entry.gid = 0
    archive.addfile(entry, io.BytesIO(data))


def write_package(repo: Path, output: Path, manifest: dict, records: dict[str, dict]) -> dict:
    output = output.expanduser().resolve()
    if output.is_relative_to(repo):
        raise ValueError("Output directory must be outside the repository")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    checksums = [f"{record['sha256']}  payload/{name}\n" for name, record in records.items()]
    checksums.append(f"{hashlib.sha256(manifest_bytes).hexdigest()}  package_manifest.json\n")
    sums_bytes = "".join(checksums).encode()
    output.mkdir(parents=True)
    archive_path = output / "core4d_bucket_transfer.tar.gz"
    partial_path = output / "core4d_bucket_transfer.tar.gz.partial"
    with partial_path.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for relative, record in records.items():
                    raw = read_file(repo, relative)
                    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                        raise RuntimeError(f"Working-tree file changed during packaging: {relative}")
                    mode = stat.S_IMODE(safe_file(repo, relative).stat().st_mode) & 0o777
                    add_bytes(archive, f"payload/{relative}", raw, mode)
                add_bytes(archive, "package_manifest.json", manifest_bytes)
                add_bytes(archive, "SHA256SUMS", sums_bytes)
    for relative, expected in {**manifest["required_base_files"], **records}.items():
        if identity(repo, relative) != expected:
            raise RuntimeError(f"Source changed before package completion: {relative}")
    if git(repo, "rev-parse", "HEAD") != manifest["base_git_head"]:
        raise RuntimeError("Git HEAD changed before package completion")
    partial_path.rename(archive_path)
    for name, data in (("package_manifest.json", manifest_bytes), ("SHA256SUMS", sums_bytes)):
        with (output / name).open("xb") as stream:
            stream.write(data)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    with (output / "archive.sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {archive_path.name}\n")
    return {"archive": str(archive_path), "archive_sha256": digest,
            "archive_bytes": archive_path.stat().st_size, "payload_files": len(records),
            "payload_uncompressed_bytes": manifest["payload_uncompressed_bytes"],
            "base_git_head": manifest["base_git_head"], "required_base_file_count": len(manifest["required_base_files"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="New directory outside the repository; never overwritten")
    parser.add_argument("--handoff-doc", default="CORE4D_BUCKET_AB_TRAINING_CN.md", help="Required existing repo-relative manual")
    parser.add_argument("--include-reference-robot", action="store_true", help="Bundle reference G1 URDF/XML and transitive mesh/texture dependencies too")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report the plan without creating any output")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        parser.error("Packaging requires --output-dir")
    repo = args.repo_root.expanduser().resolve()
    manifest, records = package_plan(repo, args.handoff_doc, args.include_reference_robot)
    result = ({"dry_run": True, **manifest} if args.dry_run else write_package(repo, args.output_dir, manifest, records))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
