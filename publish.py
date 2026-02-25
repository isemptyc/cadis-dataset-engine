#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED_PACKAGE_FILES = (
    "dataset_release_manifest.json",
    "dataset_release_manifest.sha256",
    "dataset_package.tar.gz",
    "dataset_package.tar.gz.sha256",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha256_sidecar(path: Path, expected_filename: str) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    parts = raw.split()
    if len(parts) < 2:
        raise ValueError(f"invalid sha256 sidecar format: {path}")
    digest = parts[0].strip()
    filename = parts[-1].lstrip("*").strip()
    if filename != expected_filename:
        raise ValueError(
            f"sidecar filename mismatch in {path}: expected={expected_filename} actual={filename}"
        )
    if len(digest) != 64:
        raise ValueError(f"invalid sha256 digest length in {path}")
    return digest


def load_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("dataset_release_manifest.json must be a JSON object")
    if raw.get("profile") != "cadis_dataset_release":
        raise ValueError("unsupported manifest profile; expected cadis_dataset_release")
    return raw


def ensure_package_dir(package_dir: Path) -> dict:
    for name in REQUIRED_PACKAGE_FILES:
        p = package_dir / name
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"missing package artifact: {p}")

    manifest = load_manifest(package_dir / "dataset_release_manifest.json")
    manifest_expected = parse_sha256_sidecar(
        package_dir / "dataset_release_manifest.sha256",
        "dataset_release_manifest.json",
    )
    package_expected = parse_sha256_sidecar(
        package_dir / "dataset_package.tar.gz.sha256",
        "dataset_package.tar.gz",
    )

    manifest_actual = sha256_file(package_dir / "dataset_release_manifest.json")
    package_actual = sha256_file(package_dir / "dataset_package.tar.gz")
    if manifest_actual != manifest_expected:
        raise ValueError(
            "dataset_release_manifest.sha256 mismatch: "
            f"expected={manifest_expected} actual={manifest_actual}"
        )
    if package_actual != package_expected:
        raise ValueError(
            "dataset_package.tar.gz.sha256 mismatch: "
            f"expected={package_expected} actual={package_actual}"
        )

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish packaged Cadis release artifacts into cadis-dataset repo (immutable path)."
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Directory created by package.py",
    )
    parser.add_argument(
        "--dataset-repo-root",
        type=Path,
        required=True,
        help="Path to cadis-dataset repository root",
    )
    args = parser.parse_args()

    package_dir = args.package_dir.resolve()
    dataset_root = args.dataset_repo_root.resolve()
    if not package_dir.exists() or not package_dir.is_dir():
        raise SystemExit(f"package-dir missing: {package_dir}")
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"dataset-repo-root missing: {dataset_root}")

    manifest = ensure_package_dir(package_dir)
    country_iso = str(manifest.get("country_iso", "")).strip().upper()
    dataset_id = str(manifest.get("dataset_id", "")).strip()
    version = str(manifest.get("version", "")).strip()
    if not country_iso or not dataset_id or not version:
        raise ValueError("manifest must include country_iso, dataset_id, version")

    target_dir = dataset_root / "releases" / country_iso / dataset_id / version
    if target_dir.exists():
        raise FileExistsError(f"immutable release path already exists: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_PACKAGE_FILES:
        shutil.copy2(package_dir / name, target_dir / name)

    result = {
        "status": "ok",
        "target_dir": str(target_dir),
        "country_iso": country_iso,
        "dataset_id": dataset_id,
        "version": version,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

