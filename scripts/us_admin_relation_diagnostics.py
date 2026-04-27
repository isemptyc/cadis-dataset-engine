from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import osmium
from shapely import wkb as shapely_wkb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.us.engine_us import US_REGION_NAMES, US_REGIONS
from dataset import _pick_name


NAME_KEYS = ("name:en", "name", "official_name")


def _parse_levels(raw: str) -> tuple[int, ...]:
    levels = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not levels:
        raise ValueError("--levels must include at least one admin level")
    return levels


def _relation_member_stats(relation) -> dict[str, int]:
    stats = Counter()
    for member in relation.members:
        stats[f"type_{member.type}"] += 1
        role = str(member.role or "").strip() or "(empty)"
        stats[f"role_{role}"] += 1
    return dict(sorted(stats.items()))


class AdminRelationScanner(osmium.SimpleHandler):
    def __init__(self, *, levels: Iterable[int]):
        super().__init__()
        self.levels = set(levels)
        self.relations: dict[str, dict] = {}
        self.level_counts = Counter()

    def relation(self, relation):
        tags = dict(relation.tags)
        if tags.get("boundary") != "administrative":
            return
        level_tag = tags.get("admin_level")
        if not isinstance(level_tag, str) or not level_tag.isdigit():
            return
        level = int(level_tag)
        if level not in self.levels:
            return

        rel_id = f"r{relation.id}"
        row = {
            "id": rel_id,
            "osm_id": relation.id,
            "level": level,
            "name": _pick_name(tags, NAME_KEYS),
            "tags": {
                key: tags.get(key)
                for key in ("name", "name:en", "official_name", "boundary", "admin_level", "type")
                if tags.get(key) is not None
            },
            "member_count": len(relation.members),
            "member_stats": _relation_member_stats(relation),
        }
        self.relations[rel_id] = row
        self.level_counts[level] += 1


class AdminAreaScanner(osmium.SimpleHandler):
    def __init__(self, *, levels: Iterable[int]):
        super().__init__()
        self.levels = set(levels)
        self.wkbfab = osmium.geom.WKBFactory()
        self.area_emitted: dict[str, dict] = {}
        self.wkb_success: dict[str, dict] = {}
        self.wkb_errors: dict[str, dict] = {}
        self.invalid_geometries: dict[str, dict] = {}
        self.empty_geometries: dict[str, dict] = {}
        self.level_counts = Counter()
        self.success_level_counts = Counter()

    def area(self, area):
        if area.tags.get("boundary") != "administrative":
            return
        if area.from_way():
            return
        level_tag = area.tags.get("admin_level")
        if not level_tag or not level_tag.isdigit():
            return
        level = int(level_tag)
        if level not in self.levels:
            return

        rel_id = f"r{area.orig_id()}"
        tags = dict(area.tags)
        base = {
            "id": rel_id,
            "osm_id": area.orig_id(),
            "level": level,
            "name": _pick_name(tags, NAME_KEYS),
        }
        self.area_emitted[rel_id] = base
        self.level_counts[level] += 1

        try:
            geom = shapely_wkb.loads(self.wkbfab.create_multipolygon(area), hex=True)
        except Exception as exc:
            self.wkb_errors[rel_id] = {
                **base,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            return

        row = {
            **base,
            "is_valid": bool(geom.is_valid),
            "is_empty": bool(geom.is_empty),
            "area_degrees2": float(geom.area),
            "bounds": list(geom.bounds) if not geom.is_empty else None,
            "geom_type": geom.geom_type,
        }
        self.wkb_success[rel_id] = row
        self.success_level_counts[level] += 1
        if geom.is_empty:
            self.empty_geometries[rel_id] = row
        elif not geom.is_valid:
            self.invalid_geometries[rel_id] = row


def _level_counter(rows: Iterable[dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        level = row.get("level")
        if isinstance(level, int):
            counts[str(level)] += 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _name_matches(row: dict, needles: tuple[str, ...]) -> bool:
    name = row.get("name")
    if not isinstance(name, str):
        return False
    normalized = name.casefold()
    return any(needle.casefold() in normalized for needle in needles)


def analyze_pbf(path: Path, *, levels: tuple[int, ...], trace_names: tuple[str, ...]) -> dict:
    relation_scanner = AdminRelationScanner(levels=levels)
    relation_scanner.apply_file(str(path), locations=False)

    area_scanner = AdminAreaScanner(levels=levels)
    area_scanner.apply_file(str(path), locations=True)

    relation_ids = set(relation_scanner.relations)
    emitted_ids = set(area_scanner.area_emitted)
    success_ids = set(area_scanner.wkb_success)

    not_emitted = relation_ids - emitted_ids
    emitted_but_wkb_failed = emitted_ids - success_ids
    missing_success = relation_ids - success_ids

    level4_success = [
        row for row in area_scanner.wkb_success.values()
        if row.get("level") == 4 and not row.get("is_empty")
    ]
    level4_success.sort(key=lambda row: float(row.get("area_degrees2") or 0.0), reverse=True)

    trace_rows = []
    if trace_names:
        for source, rows in (
            ("relation", relation_scanner.relations.values()),
            ("area_emitted", area_scanner.area_emitted.values()),
            ("wkb_success", area_scanner.wkb_success.values()),
            ("wkb_error", area_scanner.wkb_errors.values()),
        ):
            for row in rows:
                if _name_matches(row, trace_names):
                    relation_id = row.get("id")
                    trace_rows.append(
                        {
                            "source": source,
                            "status": (
                                "wkb_success"
                                if relation_id in success_ids
                                else "area_not_emitted"
                                if relation_id not in emitted_ids
                                else "wkb_failed"
                            ),
                            **row,
                        }
                    )

    return {
        "region": path.name.removesuffix("-latest.osm.pbf"),
        "pbf": str(path),
        "pbf_size": path.stat().st_size,
        "relation_count_by_level": _level_counter(relation_scanner.relations.values()),
        "area_emitted_by_level": _level_counter(area_scanner.area_emitted.values()),
        "wkb_success_by_level": _level_counter(area_scanner.wkb_success.values()),
        "not_emitted_by_level": _level_counter(
            relation_scanner.relations[rel_id] for rel_id in not_emitted
        ),
        "wkb_failed_by_level": _level_counter(
            area_scanner.area_emitted[rel_id] for rel_id in emitted_but_wkb_failed
        ),
        "invalid_success_by_level": _level_counter(area_scanner.invalid_geometries.values()),
        "empty_success_by_level": _level_counter(area_scanner.empty_geometries.values()),
        "totals": {
            "relations": len(relation_ids),
            "area_emitted": len(emitted_ids),
            "wkb_success": len(success_ids),
            "area_not_emitted": len(not_emitted),
            "wkb_failed": len(emitted_but_wkb_failed),
            "missing_success": len(missing_success),
            "invalid_success": len(area_scanner.invalid_geometries),
            "empty_success": len(area_scanner.empty_geometries),
        },
        "level4_largest": level4_success[0] if level4_success else None,
        "level4_expected_name": US_REGION_NAMES.get(path.name.removesuffix("-latest.osm.pbf")),
        "level4_expected_found": any(
            row.get("name") == US_REGION_NAMES.get(path.name.removesuffix("-latest.osm.pbf"))
            for row in level4_success
        ),
        "trace": trace_rows,
        "samples": {
            "area_not_emitted": [
                relation_scanner.relations[rel_id]
                for rel_id in sorted(not_emitted)[:20]
            ],
            "wkb_failed": [
                area_scanner.wkb_errors[rel_id]
                for rel_id in sorted(emitted_but_wkb_failed)
                if rel_id in area_scanner.wkb_errors
            ][:20],
            "invalid_success": [
                area_scanner.invalid_geometries[rel_id]
                for rel_id in sorted(area_scanner.invalid_geometries)[:20]
            ],
        },
    }


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "region",
        "relations",
        "area_emitted",
        "wkb_success",
        "area_not_emitted",
        "wkb_failed",
        "missing_success",
        "invalid_success",
        "empty_success",
        "level4_expected_name",
        "level4_expected_found",
        "level4_largest_name",
        "level4_largest_area_degrees2",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            largest = row.get("level4_largest") or {}
            writer.writerow(
                {
                    "region": row["region"],
                    **row["totals"],
                    "level4_expected_name": row.get("level4_expected_name"),
                    "level4_expected_found": row.get("level4_expected_found"),
                    "level4_largest_name": largest.get("name"),
                    "level4_largest_area_degrees2": largest.get("area_degrees2"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--levels", default="4,6,8")
    parser.add_argument(
        "--regions",
        default=",".join(US_REGIONS),
        help="Comma-separated Geofabrik region names, default: all US regions.",
    )
    parser.add_argument(
        "--trace-name",
        action="append",
        default=[],
        help="Case-insensitive name substring to include in trace output.",
    )
    args = parser.parse_args()

    osm_dir = Path(args.osm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = _parse_levels(args.levels)
    regions = tuple(part.strip() for part in args.regions.split(",") if part.strip())
    trace_names = tuple(args.trace_name)

    reports = []
    for region in regions:
        pbf_path = osm_dir / f"{region}-latest.osm.pbf"
        if not pbf_path.exists():
            raise FileNotFoundError(pbf_path)
        print(f"[{datetime.now()}] analyzing {region}...")
        reports.append(analyze_pbf(pbf_path, levels=levels, trace_names=trace_names))

    aggregate = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "osm_dir": str(osm_dir),
        "levels": list(levels),
        "regions": list(regions),
        "region_count": len(reports),
        "totals": dict(sum((Counter(row["totals"]) for row in reports), Counter())),
        "by_region": reports,
    }

    (output_dir / "us_admin_relation_diagnostics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(reports, output_dir / "us_admin_relation_diagnostics.csv")
    print(output_dir / "us_admin_relation_diagnostics.json")
    print(output_dir / "us_admin_relation_diagnostics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
