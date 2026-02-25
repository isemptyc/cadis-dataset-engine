from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _normalize_country(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        raise ValueError("country must be non-empty")
    return value


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify built dataset artifacts against cadis-dataset release manifest."
    )
    parser.add_argument("--country", required=True, help="Country ISO2 code, e.g. tw")
    parser.add_argument("--version", required=True, help="Dataset version, e.g. 1.0.0")
    parser.add_argument("--output", required=True, type=Path, help="Build output root used by build.py")
    parser.add_argument(
        "--release-manifest",
        required=True,
        type=Path,
        help="Path to cadis-dataset dataset_release_manifest.json",
    )
    args = parser.parse_args()

    country = _normalize_country(args.country)
    version = args.version.strip()
    if not version:
        raise ValueError("version must be non-empty")

    built_dir = args.output.resolve() / country.upper() / version
    if not built_dir.exists() or not built_dir.is_dir():
        raise FileNotFoundError(f"built dataset directory not found: {built_dir}")

    manifest_path = args.release_manifest.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"release manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("release manifest files must be a non-empty object")

    errors: list[str] = []
    checked = 0

    for rel, spec in sorted(files.items()):
        p = built_dir / rel
        if not p.exists():
            errors.append(f"missing file: {rel}")
            continue
        if not p.is_file():
            errors.append(f"not a regular file: {rel}")
            continue

        expected_sha = spec if isinstance(spec, str) else spec.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            errors.append(f"invalid expected sha256 in manifest for: {rel}")
            continue
        actual_sha = _sha256_file(p)
        if actual_sha != expected_sha:
            errors.append(f"sha256 mismatch for {rel}: expected={expected_sha} actual={actual_sha}")

        if isinstance(spec, dict) and "size" in spec:
            expected_size = int(spec["size"])
            actual_size = p.stat().st_size
            if actual_size != expected_size:
                errors.append(f"size mismatch for {rel}: expected={expected_size} actual={actual_size}")

        checked += 1

    if errors:
        print("VERIFY_FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    print("VERIFY_PASS")
    print(f"checked_files={checked}")
    print(f"built_dir={built_dir}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
