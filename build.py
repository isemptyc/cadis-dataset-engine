from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from engines.au.engine_au import AustraliaAdminEngine
from engines.be.engine_be import BelgiumAdminEngine
from engines.ca.engine_ca import CA_REGIONS, CanadaAdminEngine
from engines.de.engine_de import GermanyAdminEngine
from engines.dk.engine_dk import DenmarkAdminEngine
from engines.es.engine_es import SpainAdminEngine
from engines.fr.engine_fr import FranceAdminEngine
from engines.fi.engine_fi import FinlandAdminEngine
from engines.gb.engine_gb import GreatBritainAdminEngine
from engines.it.engine_it import ItalyAdminEngine
from engines.jp.engine_jp import JapanAdminEngine
from engines.kr.engine_kr import SouthKoreaAdminEngine
from engines.nl.engine_nl import NetherlandsAdminEngine
from engines.no.engine_no import NorwayAdminEngine
from engines.pt.engine_pt import PortugalAdminEngine
from engines.se.engine_se import SwedenAdminEngine
from engines.tw.engine_tw import TaiwanAdminEngine
from engines.us.engine_us import US_REGIONS, UnitedStatesAdminEngine
import osmium


TW_1_0_0_OSM_SHA256 = "6b899702570a6554c5e2bcdd30bd569c5685943acea10c054eed34843e3c215a"
TW_1_0_0_OSM_REPLICATION_TIMESTAMP_UTC = "2025-10-19T20:21:00Z"


def _normalize_country(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        raise ValueError("country must be non-empty")
    if value == "uk":
        return "gb"
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


def _write_source_osm_identity(
    *,
    work_dir: Path,
    osm_pbf_path: Path,
    include_file_names: set[str] | None = None,
) -> None:
    manifest_path = work_dir / "dataset_build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset build manifest missing: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset build manifest must be a JSON object: {manifest_path}")

    osm_path = osm_pbf_path.resolve()
    if osm_path.is_dir():
        pbf_paths = sorted(osm_path.glob("*-latest.osm.pbf"))
        if include_file_names is not None:
            pbf_paths = [p for p in pbf_paths if p.name in include_file_names]
        if not pbf_paths:
            raise ValueError(f"OSM directory contains no *-latest.osm.pbf files: {osm_path}")
        source_files = []
        timestamps = []
        for pbf in pbf_paths:
            timestamp = _read_osm_replication_timestamp_utc(pbf)
            if timestamp:
                timestamps.append(timestamp)
            source_files.append(
                {
                    "file_name": pbf.name,
                    "file_sha256": _sha256_file(pbf),
                    "replication_timestamp_utc": timestamp,
                }
            )
        source_manifest = {
            "directory_name": osm_path.name,
            "files": source_files,
        }
        manifest_blob = json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["source_osm"] = {
            "file_name": osm_path.name,
            "file_sha256": hashlib.sha256(manifest_blob).hexdigest(),
            "replication_timestamp_utc": max(timestamps) if timestamps else None,
            "source_manifest": source_manifest,
        }
    else:
        payload["source_osm"] = {
            "file_name": osm_path.name,
            "file_sha256": _sha256_file(osm_path),
            "replication_timestamp_utc": _read_osm_replication_timestamp_utc(osm_path),
        }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generic dataset build dispatcher for cadis-dataset-engine."
    )
    parser.add_argument("--country", required=True, help="Country ISO2 code, e.g. tw")
    parser.add_argument("--version", required=True, help="Dataset version, e.g. 1.0.0")
    parser.add_argument("--osm", required=True, type=Path, help="Path to OSM PBF input")
    parser.add_argument("--output", required=True, type=Path, help="Output directory root")
    parser.add_argument(
        "--country-geometry",
        type=Path,
        help="Optional country boundary JSON used for scoped export metadata.",
    )
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

    if country == "tw":
        if args.verified and version == "1.0.0":
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
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "au":
        AustraliaAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "jp":
        JapanAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "kr":
        SouthKoreaAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "gb":
        GreatBritainAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "it":
        ItalyAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "se":
        SwedenAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "no":
        NorwayAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "nl":
        NetherlandsAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "dk":
        DenmarkAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "de":
        GermanyAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "es":
        SpainAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "fr":
        FranceAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "fi":
        FinlandAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "pt":
        PortugalAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "be":
        BelgiumAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
        )
        print(work_dir)
        return 0

    if country == "ca":
        CanadaAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
            include_file_names={f"{region}-latest.osm.pbf" for region in CA_REGIONS},
        )
        print(work_dir)
        return 0

    if country == "us":
        UnitedStatesAdminEngine.prepare_datasets(
            osm_pbf_path=args.osm,
            work_dir=work_dir,
            country_geometry_path=args.country_geometry,
        )
        _write_source_osm_identity(
            work_dir=work_dir,
            osm_pbf_path=args.osm,
            include_file_names={f"{region}-latest.osm.pbf" for region in US_REGIONS},
        )
        print(work_dir)
        return 0

    raise ValueError(f"Unsupported country/version build target: {country}/{version}")


if __name__ == "__main__":
    raise SystemExit(main())
