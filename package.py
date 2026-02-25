#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataset_release_manifest.json must be a JSON object")
    return raw


def _manifest_files_map(manifest: dict) -> dict:
    schema_version = manifest.get("schema_version")
    if schema_version == 2:
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict):
            raise ValueError("schema v2 manifest missing checksums object")
        files = checksums.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("schema v2 manifest checksums.files must be a non-empty object")
        return files

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("manifest files must be a non-empty object")
    return files


def validate_release_dir(release_dir: Path) -> dict:
    manifest_path = release_dir / "dataset_release_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    manifest = read_manifest(manifest_path)
    profile = manifest.get("profile")
    if profile not in {"cadis_dataset_release", "cadis.dataset.release"}:
        raise ValueError(
            "unsupported manifest profile; expected cadis_dataset_release or cadis.dataset.release"
        )
    schema_version = manifest.get("schema_version")
    manifest_version = manifest.get("manifest_version")
    if schema_version is not None and schema_version != 2:
        raise ValueError("unsupported schema_version; expected 2")
    if schema_version is None and manifest_version not in {"1.0", "1.1"}:
        raise ValueError("unsupported manifest_version; expected 1.0 or 1.1")

    files = _manifest_files_map(manifest)

    for rel, meta in sorted(files.items()):
        if not isinstance(meta, dict):
            raise ValueError(f"files[{rel!r}] must be an object")
        p = release_dir / rel
        if not p.exists():
            raise FileNotFoundError(f"missing release file from manifest: {p}")
        expected_sha = str(meta.get("sha256", "")).strip()
        expected_size = meta.get("size")
        if not expected_sha:
            raise ValueError(f"files[{rel!r}].sha256 is required")
        if not isinstance(expected_size, int):
            raise ValueError(f"files[{rel!r}].size must be integer")
        actual_sha = sha256_file(p)
        actual_size = p.stat().st_size
        if actual_sha != expected_sha:
            raise ValueError(
                f"sha256 mismatch for {rel}: expected={expected_sha} actual={actual_sha}"
            )
        if actual_size != expected_size:
            raise ValueError(
                f"size mismatch for {rel}: expected={expected_size} actual={actual_size}"
            )

    if schema_version is None:
        runtime_policy_sha = str(manifest.get("runtime_policy_checksum", "")).strip()
        runtime_policy_meta = files.get("runtime_policy.json") or {}
        runtime_policy_meta_sha = str(runtime_policy_meta.get("sha256", "")).strip()
        if runtime_policy_sha and runtime_policy_meta_sha and runtime_policy_sha != runtime_policy_meta_sha:
            raise ValueError(
                "runtime_policy_checksum must equal files['runtime_policy.json'].sha256"
            )

    return manifest


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    # Ensure deterministic package bytes across rebuilds.
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def build_package_tar(release_dir: Path, out_tar: Path) -> None:
    excluded = {
        "dataset_release_manifest.sha256",
        "dataset_package.tar.gz",
        "dataset_package.tar.gz.sha256",
    }
    with tarfile.open(out_tar, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(release_dir.iterdir(), key=lambda x: x.name):
            if p.name in excluded:
                continue
            tf.add(p, arcname=p.name, recursive=True, filter=_tar_filter)


def write_sidecar(artifact: Path, sidecar: Path) -> str:
    digest = sha256_file(artifact)
    sidecar.write_text(f"{digest}  {artifact.name}\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Cadis CDN package artifacts from a release version directory."
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        required=True,
        help="Path to cadis-dataset releases/<ISO2>/<dataset_id>/<version>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write packaged CDN artifacts.",
    )
    args = parser.parse_args()

    release_dir = args.release_dir.resolve()
    out_dir = args.output_dir.resolve()
    if not release_dir.exists() or not release_dir.is_dir():
        raise SystemExit(f"release-dir missing: {release_dir}")

    manifest = validate_release_dir(release_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_src = release_dir / "dataset_release_manifest.json"
    manifest_dst = out_dir / "dataset_release_manifest.json"
    shutil.copy2(manifest_src, manifest_dst)

    tar_path = out_dir / "dataset_package.tar.gz"
    build_package_tar(release_dir, tar_path)

    manifest_sha = out_dir / "dataset_release_manifest.sha256"
    package_sha = out_dir / "dataset_package.tar.gz.sha256"
    manifest_digest = write_sidecar(manifest_dst, manifest_sha)
    package_digest = write_sidecar(tar_path, package_sha)

    result = {
        "status": "ok",
        "release_dir": str(release_dir),
        "output_dir": str(out_dir),
        "dataset_id": manifest.get("dataset_id"),
        "version": manifest.get("dataset_version") or manifest.get("version"),
        "manifest_sha256": manifest_digest,
        "package_sha256": package_digest,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
