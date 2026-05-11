from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from engines.at.engine_at import AustriaAdminEngine
from engines.af.engine_af import AfghanistanAdminEngine
from engines.al.engine_al import AlbaniaAdminEngine
from engines.am.engine_am import ArmeniaAdminEngine
from engines.ar.engine_ar import ArgentinaAdminEngine
from engines.au.engine_au import AustraliaAdminEngine
from engines.az.engine_az import AzerbaijanAdminEngine
from engines.td.engine_td import ChadAdminEngine
from engines.cf.engine_cf import CentralAfricanRepublicAdminEngine
from engines.cv.engine_cv import CapeVerdeAdminEngine
from engines.cm.engine_cm import CameroonAdminEngine
from engines.bi.engine_bi import BurundiAdminEngine
from engines.bf.engine_bf import BurkinaFasoAdminEngine
from engines.bw.engine_bw import BotswanaAdminEngine
from engines.bj.engine_bj import BeninAdminEngine
from engines.ao.engine_ao import AngolaAdminEngine
from engines.dz.engine_dz import AlgeriaAdminEngine
from engines.ye.engine_ye import YemenAdminEngine
from engines.uz.engine_uz import UzbekistanAdminEngine
from engines.tm.engine_tm import TurkmenistanAdminEngine
from engines.tj.engine_tj import TajikistanAdminEngine
from engines.sy.engine_sy import SyriaAdminEngine
from engines.lk.engine_lk import SriLankaAdminEngine
from engines.pk.engine_pk import PakistanAdminEngine
from engines.kp.engine_kp import NorthKoreaAdminEngine
from engines.np.engine_np import NepalAdminEngine
from engines.mm.engine_mm import MyanmarAdminEngine
from engines.mn.engine_mn import MongoliaAdminEngine
from engines.mv.engine_mv import MaldivesAdminEngine
from engines.bn.engine_bn import BruneiAdminEngine
from engines.lb.engine_lb import LebanonAdminEngine
from engines.la.engine_la import LaosAdminEngine
from engines.kg.engine_kg import KyrgyzstanAdminEngine
from engines.kz.engine_kz import KazakhstanAdminEngine
from engines.jo.engine_jo import JordanAdminEngine
from engines.ps.engine_ps import PalestineAdminEngine
from engines.il.engine_il import IsraelAdminEngine
from engines.iq.engine_iq import IraqAdminEngine
from engines.ir.engine_ir import IranAdminEngine
from engines.ae.engine_ae import UnitedArabEmiratesAdminEngine
from engines.sa.engine_sa import SaudiArabiaAdminEngine
from engines.qa.engine_qa import QatarAdminEngine
from engines.om.engine_om import OmanAdminEngine
from engines.kw.engine_kw import KuwaitAdminEngine
from engines.bh.engine_bh import BahrainAdminEngine
from engines.tl.engine_tl import TimorLesteAdminEngine
from engines.kh.engine_kh import CambodiaAdminEngine
from engines.bt.engine_bt import BhutanAdminEngine
from engines.bd.engine_bd import BangladeshAdminEngine
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
from engines.do.engine_do import DominicanRepublicAdminEngine
from engines.ee.engine_ee import EstoniaAdminEngine
from engines.ec.engine_ec import EcuadorAdminEngine
from engines.es.engine_es import SpainAdminEngine
from engines.fr.engine_fr import FranceAdminEngine
from engines.fi.engine_fi import FinlandAdminEngine
from engines.ge.engine_ge import GeorgiaAdminEngine
from engines.gr.engine_gr import GreeceAdminEngine
from engines.gb.engine_gb import GreatBritainAdminEngine
from engines.gy.engine_gy import GuyanaAdminEngine
from engines.gt.engine_gt import GuatemalaAdminEngine
from engines.ht.engine_ht import HaitiAdminEngine
from engines.hr.engine_hr import CroatiaAdminEngine
from engines.hn.engine_hn import HondurasAdminEngine
from engines.hu.engine_hu import HungaryAdminEngine
from engines.id.engine_id import IndonesiaAdminEngine
from engines.it.engine_it import ItalyAdminEngine
from engines.jm.engine_jm import JamaicaAdminEngine
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
from engines.ni.engine_ni import NicaraguaAdminEngine
from engines.nl.engine_nl import NetherlandsAdminEngine
from engines.no.engine_no import NorwayAdminEngine
from engines.nz.engine_nz import NewZealandAdminEngine
from engines.pa.engine_pa import PanamaAdminEngine
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

    if country == "gt":
        GuatemalaAdminEngine.prepare_datasets(
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

    if country == "ht":
        HaitiAdminEngine.prepare_datasets(
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

    if country == "do":
        DominicanRepublicAdminEngine.prepare_datasets(
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

    if country == "hn":
        HondurasAdminEngine.prepare_datasets(
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

    if country == "jm":
        JamaicaAdminEngine.prepare_datasets(
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

    if country == "ni":
        NicaraguaAdminEngine.prepare_datasets(
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

    if country == "pa":
        PanamaAdminEngine.prepare_datasets(
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

    if country == "af":
        AfghanistanAdminEngine.prepare_datasets(
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

    if country == "am":
        ArmeniaAdminEngine.prepare_datasets(
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

    if country == "az":
        AzerbaijanAdminEngine.prepare_datasets(
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

    if country == "bd":
        BangladeshAdminEngine.prepare_datasets(
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

    if country == "bt":
        BhutanAdminEngine.prepare_datasets(
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

    if country == "kh":
        CambodiaAdminEngine.prepare_datasets(
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

    if country == "tl":
        TimorLesteAdminEngine.prepare_datasets(
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

    if country == "bh":
        BahrainAdminEngine.prepare_datasets(
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

    if country == "kw":
        KuwaitAdminEngine.prepare_datasets(
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

    if country == "om":
        OmanAdminEngine.prepare_datasets(
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

    if country == "qa":
        QatarAdminEngine.prepare_datasets(
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

    if country == "sa":
        SaudiArabiaAdminEngine.prepare_datasets(
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

    if country == "ae":
        UnitedArabEmiratesAdminEngine.prepare_datasets(
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

    if country == "ir":
        IranAdminEngine.prepare_datasets(
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

    if country == "iq":
        IraqAdminEngine.prepare_datasets(
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

    if country == "il":
        IsraelAdminEngine.prepare_datasets(
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

    if country == "ps":
        PalestineAdminEngine.prepare_datasets(
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

    if country == "jo":
        JordanAdminEngine.prepare_datasets(
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

    if country == "kz":
        KazakhstanAdminEngine.prepare_datasets(
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

    if country == "kg":
        KyrgyzstanAdminEngine.prepare_datasets(
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

    if country == "la":
        LaosAdminEngine.prepare_datasets(
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

    if country == "lb":
        LebanonAdminEngine.prepare_datasets(
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

    if country == "bn":
        BruneiAdminEngine.prepare_datasets(
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

    if country == "mv":
        MaldivesAdminEngine.prepare_datasets(
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

    if country == "mn":
        MongoliaAdminEngine.prepare_datasets(
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

    if country == "mm":
        MyanmarAdminEngine.prepare_datasets(
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

    if country == "np":
        NepalAdminEngine.prepare_datasets(
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

    if country == "kp":
        NorthKoreaAdminEngine.prepare_datasets(
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

    if country == "pk":
        PakistanAdminEngine.prepare_datasets(
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

    if country == "lk":
        SriLankaAdminEngine.prepare_datasets(
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

    if country == "sy":
        SyriaAdminEngine.prepare_datasets(
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

    if country == "tj":
        TajikistanAdminEngine.prepare_datasets(
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

    if country == "tm":
        TurkmenistanAdminEngine.prepare_datasets(
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

    if country == "uz":
        UzbekistanAdminEngine.prepare_datasets(
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

    if country == "ye":
        YemenAdminEngine.prepare_datasets(
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

    if country == "dz":
        AlgeriaAdminEngine.prepare_datasets(
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

    if country == "ao":
        AngolaAdminEngine.prepare_datasets(
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

    if country == "bj":
        BeninAdminEngine.prepare_datasets(
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

    if country == "bw":
        BotswanaAdminEngine.prepare_datasets(
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

    if country == "bf":
        BurkinaFasoAdminEngine.prepare_datasets(
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

    if country == "bi":
        BurundiAdminEngine.prepare_datasets(
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

    if country == "cm":
        CameroonAdminEngine.prepare_datasets(
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

    if country == "cv":
        CapeVerdeAdminEngine.prepare_datasets(
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

    if country == "cf":
        CentralAfricanRepublicAdminEngine.prepare_datasets(
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

    if country == "td":
        ChadAdminEngine.prepare_datasets(
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
