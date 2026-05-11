from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from engines.at.engine_at import AustriaAdminEngine
from engines.al.engine_al import AlbaniaAdminEngine
from engines.ar.engine_ar import ArgentinaAdminEngine
from engines.au.engine_au import AustraliaAdminEngine
from engines.ba.engine_ba import BosniaHerzegovinaAdminEngine
from engines.be.engine_be import BelgiumAdminEngine
from engines.bg.engine_bg import BulgariaAdminEngine
from engines.bo.engine_bo import BoliviaAdminEngine
from engines.br.engine_br import BrazilAdminEngine
from engines.bs.engine_bs import BahamasAdminEngine
from engines.bz.engine_bz import BelizeAdminEngine
from engines.ca.engine_ca import CA_REGIONS, CanadaAdminEngine
from engines.co.engine_co import ColombiaAdminEngine
from engines.cr.engine_cr import CostaRicaAdminEngine
from engines.cu.engine_cu import CubaAdminEngine
from engines.cy.engine_cy import CyprusAdminEngine
from engines.cz.engine_cz import CzechRepublicAdminEngine
from engines.de.engine_de import GermanyAdminEngine
from engines.dk.engine_dk import DenmarkAdminEngine
from engines.ee.engine_ee import EstoniaAdminEngine
from engines.ec.engine_ec import EcuadorAdminEngine
from engines.es.engine_es import SpainAdminEngine
from engines.fr.engine_fr import FranceAdminEngine
from engines.fi.engine_fi import FinlandAdminEngine
from engines.ge.engine_ge import GeorgiaAdminEngine
from engines.gr.engine_gr import GreeceAdminEngine
from engines.gb.engine_gb import GreatBritainAdminEngine
from engines.gy.engine_gy import GuyanaAdminEngine
from engines.hr.engine_hr import CroatiaAdminEngine
from engines.hu.engine_hu import HungaryAdminEngine
from engines.id.engine_id import IndonesiaAdminEngine
from engines.it.engine_it import ItalyAdminEngine
from engines.jp.engine_jp import JapanAdminEngine
from engines.kr.engine_kr import SouthKoreaAdminEngine
from engines.lu.engine_lu import LuxembourgAdminEngine
from engines.lt.engine_lt import LithuaniaAdminEngine
from engines.lv.engine_lv import LatviaAdminEngine
from engines.md.engine_md import MoldovaAdminEngine
from engines.mc.engine_mc import MonacoAdminEngine
from engines.me.engine_me import MontenegroAdminEngine
from engines.mk.engine_mk import MacedoniaAdminEngine
from engines.my.engine_my import MalaysiaAdminEngine
from engines.mx.engine_mx import MexicoAdminEngine
from engines.nl.engine_nl import NetherlandsAdminEngine
from engines.no.engine_no import NorwayAdminEngine
from engines.nz.engine_nz import NewZealandAdminEngine
from engines.pl.engine_pl import PolandAdminEngine
from engines.ph.engine_ph import PhilippinesAdminEngine
from engines.pe.engine_pe import PeruAdminEngine
from engines.pt.engine_pt import PortugalAdminEngine
from engines.py.engine_py import ParaguayAdminEngine
from engines.ro.engine_ro import RomaniaAdminEngine
from engines.rs.engine_rs import SerbiaAdminEngine
from engines.se.engine_se import SwedenAdminEngine
from engines.sg.engine_sg import SingaporeAdminEngine
from engines.sk.engine_sk import SlovakiaAdminEngine
from engines.si.engine_si import SloveniaAdminEngine
from engines.sr.engine_sr import SurinameAdminEngine
from engines.sv.engine_sv import ElSalvadorAdminEngine
from engines.ch.engine_ch import SwitzerlandAdminEngine
from engines.cl.engine_cl import ChileAdminEngine
from engines.th.engine_th import ThailandAdminEngine
from engines.tr.engine_tr import TurkeyAdminEngine
from engines.tw.engine_tw import TaiwanAdminEngine
from engines.ua.engine_ua import UkraineAdminEngine
from engines.uy.engine_uy import UruguayAdminEngine
from engines.us.engine_us import US_REGIONS, UnitedStatesAdminEngine
from engines.ve.engine_ve import VenezuelaAdminEngine
from engines.vn.engine_vn import VietnamAdminEngine
from engines.xk.engine_xk import KosovoAdminEngine
import osmium

IcelandAdminEngine = importlib.import_module("engines.is.engine_is").IcelandAdminEngine
IndiaAdminEngine = importlib.import_module("engines.in.engine_in").IndiaAdminEngine


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

    if country == "al":
        AlbaniaAdminEngine.prepare_datasets(
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

    if country == "ba":
        BosniaHerzegovinaAdminEngine.prepare_datasets(
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

    if country == "bg":
        BulgariaAdminEngine.prepare_datasets(
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

    if country == "ec":
        EcuadorAdminEngine.prepare_datasets(
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

    if country == "gy":
        GuyanaAdminEngine.prepare_datasets(
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

    if country == "sr":
        SurinameAdminEngine.prepare_datasets(
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

    if country == "uy":
        UruguayAdminEngine.prepare_datasets(
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

    if country == "ve":
        VenezuelaAdminEngine.prepare_datasets(
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

    if country == "bo":
        BoliviaAdminEngine.prepare_datasets(
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

    if country == "bs":
        BahamasAdminEngine.prepare_datasets(
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

    if country == "bz":
        BelizeAdminEngine.prepare_datasets(
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

    if country == "cr":
        CostaRicaAdminEngine.prepare_datasets(
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

    if country == "cu":
        CubaAdminEngine.prepare_datasets(
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

    if country == "sv":
        ElSalvadorAdminEngine.prepare_datasets(
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

    if country == "hr":
        CroatiaAdminEngine.prepare_datasets(
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

    if country == "ee":
        EstoniaAdminEngine.prepare_datasets(
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

    if country == "hu":
        HungaryAdminEngine.prepare_datasets(
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

    if country == "lv":
        LatviaAdminEngine.prepare_datasets(
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

    if country == "lt":
        LithuaniaAdminEngine.prepare_datasets(
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

    if country == "mc":
        MonacoAdminEngine.prepare_datasets(
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

    if country == "me":
        MontenegroAdminEngine.prepare_datasets(
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

    if country == "ro":
        RomaniaAdminEngine.prepare_datasets(
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

    if country == "rs":
        SerbiaAdminEngine.prepare_datasets(
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

    if country == "sk":
        SlovakiaAdminEngine.prepare_datasets(
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

    if country == "si":
        SloveniaAdminEngine.prepare_datasets(
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

    if country == "cy":
        CyprusAdminEngine.prepare_datasets(
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

    if country == "ge":
        GeorgiaAdminEngine.prepare_datasets(
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

    if country == "xk":
        KosovoAdminEngine.prepare_datasets(
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

    if country == "mk":
        MacedoniaAdminEngine.prepare_datasets(
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

    if country == "md":
        MoldovaAdminEngine.prepare_datasets(
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

    if country == "ua":
        UkraineAdminEngine.prepare_datasets(
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

    if country == "ar":
        ArgentinaAdminEngine.prepare_datasets(
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

    if country == "at":
        AustriaAdminEngine.prepare_datasets(
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

    if country == "sg":
        SingaporeAdminEngine.prepare_datasets(
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

    if country == "th":
        ThailandAdminEngine.prepare_datasets(
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

    if country == "tr":
        TurkeyAdminEngine.prepare_datasets(
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

    if country == "id":
        IndonesiaAdminEngine.prepare_datasets(
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

    if country == "in":
        IndiaAdminEngine.prepare_datasets(
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

    if country == "nz":
        NewZealandAdminEngine.prepare_datasets(
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

    if country == "gr":
        GreeceAdminEngine.prepare_datasets(
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

    if country == "is":
        IcelandAdminEngine.prepare_datasets(
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

    if country == "ph":
        PhilippinesAdminEngine.prepare_datasets(
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

    if country == "pe":
        PeruAdminEngine.prepare_datasets(
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

    if country == "py":
        ParaguayAdminEngine.prepare_datasets(
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

    if country == "vn":
        VietnamAdminEngine.prepare_datasets(
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

    if country == "pl":
        PolandAdminEngine.prepare_datasets(
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

    if country == "lu":
        LuxembourgAdminEngine.prepare_datasets(
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

    if country == "my":
        MalaysiaAdminEngine.prepare_datasets(
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

    if country == "mx":
        MexicoAdminEngine.prepare_datasets(
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

    if country == "br":
        BrazilAdminEngine.prepare_datasets(
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

    if country == "ch":
        SwitzerlandAdminEngine.prepare_datasets(
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

    if country == "cl":
        ChileAdminEngine.prepare_datasets(
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

    if country == "co":
        ColombiaAdminEngine.prepare_datasets(
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

    if country == "cz":
        CzechRepublicAdminEngine.prepare_datasets(
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
