from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from engines.tw.engine_tw import TaiwanAdminEngine
import osmium


TW_1_0_0_OSM_SHA256 = "6b899702570a6554c5e2bcdd30bd569c5685943acea10c054eed34843e3c215a"
TW_1_0_0_OSM_REPLICATION_TIMESTAMP_UTC = "2025-10-19T20:21:00Z"


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


def _read_osm_replication_timestamp_utc(pbf_path: Path) -> str | None:
    reader = osmium.io.Reader(str(pbf_path))
    try:
        header = reader.header()
        val = header.get("osmosis_replication_timestamp")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic dataset build dispatcher for cadis-dataset-engine."
    )
    parser.add_argument("--country", required=True, help="Country ISO2 code, e.g. tw")
    parser.add_argument("--version", required=True, help="Dataset version, e.g. 1.0.0")
    parser.add_argument("--osm", required=True, type=Path, help="Path to OSM PBF input")
    parser.add_argument("--output", required=True, type=Path, help="Output directory root")
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Enable strict OSM identity verification (checksum + replication timestamp) for pinned targets.",
    )
    args = parser.parse_args()

    country = _normalize_country(args.country)
    version = args.version.strip()
    if not version:
        raise ValueError("version must be non-empty")

    out_root = args.output.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    work_dir = out_root / country.upper() / version
    work_dir.mkdir(parents=True, exist_ok=True)

    if country == "tw" and version == "1.0.0":
        if args.verified:
            actual_sha = _sha256_file(args.osm.resolve())
            if actual_sha != TW_1_0_0_OSM_SHA256:
                raise ValueError(
                    "OSM checksum mismatch for tw/1.0.0: "
                    f"expected={TW_1_0_0_OSM_SHA256} actual={actual_sha}"
                )
            actual_ts = _read_osm_replication_timestamp_utc(args.osm.resolve())
            if actual_ts != TW_1_0_0_OSM_REPLICATION_TIMESTAMP_UTC:
                raise ValueError(
                    "OSM replication timestamp mismatch for tw/1.0.0: "
                    f"expected={TW_1_0_0_OSM_REPLICATION_TIMESTAMP_UTC} actual={actual_ts!r}"
                )
        TaiwanAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        print(work_dir)
        return 0

    raise ValueError(f"Unsupported country/version build target: {country}/{version}")


if __name__ == "__main__":
    raise SystemExit(main())
