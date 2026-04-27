from __future__ import annotations

import json
import hashlib
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import osmium
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, split, transform, unary_union

from dataset import (
    AdminLevelPolicy,
    AdminProfile,
    TopologyEngine,
    _extract_multilingual_names,
    _pick_name,
    _resolve_level_policy,
    serialize_output,
)


class _AdminRelationCollector(osmium.SimpleHandler):
    def __init__(self, *, levels: Iterable[int], profile: AdminProfile):
        super().__init__()
        self.levels = set(levels)
        self.profile = profile
        self.relations: dict[int, dict] = {}
        self.relation_regions: dict[int, set[str]] = defaultdict(set)
        self.level_counts = Counter()
        self.duplicate_relation_count = 0

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

        way_members = [
            {
                "ref": int(member.ref),
                "role": str(member.role or ""),
            }
            for member in relation.members
            if member.type == "w"
        ]
        if not way_members:
            return

        relation_id = int(relation.id)
        row = {
            "id": relation_id,
            "osm_id": f"r{relation_id}",
            "level": level,
            "name": _pick_name(tags, self.profile.name_keys),
            "names": _extract_multilingual_names(tags, self.profile),
            "tags": tags,
            "way_members": way_members,
        }

        existing = self.relations.get(relation_id)
        if existing is None or len(way_members) > len(existing["way_members"]):
            self.relations[relation_id] = row
        elif existing is not None:
            self.duplicate_relation_count += 1
        self.level_counts[level] += 1


class _RequiredWayCollector(osmium.SimpleHandler):
    def __init__(self, *, required_way_ids: set[int]):
        super().__init__()
        self.required_way_ids = required_way_ids
        self.ways: dict[int, list[int]] = {}
        self.variant_way_ids: set[int] = set()

    def way(self, way):
        way_id = int(way.id)
        if way_id not in self.required_way_ids:
            return
        node_refs = [int(node.ref) for node in way.nodes]
        existing = self.ways.get(way_id)
        if existing is None:
            self.ways[way_id] = node_refs
        elif existing != node_refs:
            self.variant_way_ids.add(way_id)


class _RequiredNodeCollector(osmium.SimpleHandler):
    def __init__(self, *, required_node_ids: set[int]):
        super().__init__()
        self.required_node_ids = required_node_ids
        self.nodes: dict[int, tuple[float, float]] = {}
        self.variant_node_ids: set[int] = set()

    def node(self, node):
        node_id = int(node.id)
        if node_id not in self.required_node_ids:
            return
        coord = (float(node.location.lon), float(node.location.lat))
        existing = self.nodes.get(node_id)
        if existing is None:
            self.nodes[node_id] = coord
        elif existing != coord:
            self.variant_node_ids.add(node_id)


def _scan_relations(pbf_paths: list[Path], *, levels: list[int], profile: AdminProfile) -> _AdminRelationCollector:
    collector = _AdminRelationCollector(levels=levels, profile=profile)
    for pbf_path in pbf_paths:
        print(f"[{datetime.now()}] stitch scan relations: {pbf_path.name}", flush=True)
        collector.apply_file(str(pbf_path), locations=False)
        region = pbf_path.name.removesuffix("-latest.osm.pbf")
        for relation_id in collector.relations:
            # Filled conservatively after all scans; exact per-region is diagnostic-only.
            collector.relation_regions[relation_id].add(region)
    return collector


def _source_fingerprint(*, pbf_paths: list[Path], levels: list[int]) -> str:
    payload = {
        "cache_schema": "us-stitch-v2-path-independent",
        "levels": levels,
        "pbf": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in pbf_paths
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _cache_path(cache_dir: Path, fingerprint: str, name: str) -> Path:
    return cache_dir / fingerprint / name


def _load_pickle(path: Path):
    if not path.exists():
        return None
    print(f"[{datetime.now()}] stitch cache hit: {path}", flush=True)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)
    print(f"[{datetime.now()}] stitch cache written: {path}", flush=True)


def _scan_ways(pbf_paths: list[Path], required_way_ids: set[int]) -> _RequiredWayCollector:
    collector = _RequiredWayCollector(required_way_ids=required_way_ids)
    remaining = set(required_way_ids)
    for pbf_path in pbf_paths:
        if not remaining:
            break
        print(
            f"[{datetime.now()}] stitch scan ways: {pbf_path.name} "
            f"remaining={len(remaining)}",
            flush=True,
        )
        before = set(collector.ways)
        collector.apply_file(str(pbf_path), locations=False)
        remaining -= set(collector.ways) - before
    return collector


def _scan_nodes(pbf_paths: list[Path], required_node_ids: set[int]) -> _RequiredNodeCollector:
    collector = _RequiredNodeCollector(required_node_ids=required_node_ids)
    remaining = set(required_node_ids)
    for pbf_path in pbf_paths:
        if not remaining:
            break
        print(
            f"[{datetime.now()}] stitch scan nodes: {pbf_path.name} "
            f"remaining={len(remaining)}",
            flush=True,
        )
        before = set(collector.nodes)
        collector.apply_file(str(pbf_path), locations=False)
        remaining -= set(collector.nodes) - before
    return collector


def _member_lines(
    relation: dict,
    *,
    ways: dict[int, list[int]],
    nodes: dict[int, tuple[float, float]],
    role: str,
    unwrap_antimeridian: bool = False,
) -> tuple[list[LineString], list[int], list[int]]:
    lines: list[LineString] = []
    missing_ways: list[int] = []
    missing_nodes: list[int] = []

    for member in relation["way_members"]:
        member_role = str(member.get("role") or "")
        if role == "outer":
            if member_role not in {"", "outer"}:
                continue
        elif member_role != role:
            continue

        way_id = int(member["ref"])
        node_refs = ways.get(way_id)
        if not node_refs:
            missing_ways.append(way_id)
            continue
        coords = []
        way_missing_nodes = []
        for node_id in node_refs:
            coord = nodes.get(node_id)
            if coord is None:
                way_missing_nodes.append(node_id)
                missing_nodes.append(node_id)
            else:
                lon, lat = coord
                if unwrap_antimeridian and lon < 0:
                    lon += 360.0
                coords.append((lon, lat))
        if len(coords) >= 2 and not way_missing_nodes:
            lines.append(LineString(coords))

    return lines, missing_ways, missing_nodes


def _polygonize_lines(lines: list[LineString]):
    if not lines:
        return None
    merged = unary_union(lines)
    polygons = [poly for poly in polygonize(merged) if not poly.is_empty]
    if not polygons:
        return None
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _append_dangling_endpoint_repair(lines: list[LineString]) -> tuple[list[LineString], dict]:
    endpoint_counts = Counter()
    endpoint_coords: dict[tuple[float, float], tuple[float, float]] = {}
    for line in lines:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        for coord in (coords[0], coords[-1]):
            key = (round(float(coord[0]), 9), round(float(coord[1]), 9))
            endpoint_counts[key] += 1
            endpoint_coords[key] = (float(coord[0]), float(coord[1]))

    odd = [key for key, count in endpoint_counts.items() if count % 2]
    diagnostic = {
        "dangling_endpoint_count": len(odd),
        "dangling_endpoint_repaired": False,
    }
    if len(odd) != 2:
        return lines, diagnostic

    a = endpoint_coords[odd[0]]
    b = endpoint_coords[odd[1]]
    repair_line = LineString([a, b])
    # This repair is meant for tiny extract clipping gaps, not inferred borders.
    if repair_line.length > 1.0:
        diagnostic["dangling_endpoint_repair_skipped_reason"] = "gap_too_large"
        diagnostic["dangling_endpoint_gap_degrees"] = float(repair_line.length)
        return lines, diagnostic

    diagnostic["dangling_endpoint_repaired"] = True
    diagnostic["dangling_endpoint_gap_degrees"] = float(repair_line.length)
    return [*lines, repair_line], diagnostic


def _relation_needs_antimeridian_unwrap(
    relation: dict,
    *,
    ways: dict[int, list[int]],
    nodes: dict[int, tuple[float, float]],
) -> bool:
    lons = []
    for member in relation["way_members"]:
        node_refs = ways.get(int(member["ref"]))
        if not node_refs:
            continue
        for node_id in node_refs:
            coord = nodes.get(node_id)
            if coord is not None:
                lons.append(coord[0])
    if not lons:
        return False
    return min(lons) < -120.0 and max(lons) > 120.0


def _normalize_antimeridian_geometry(geom):
    if geom.is_empty:
        return geom

    splitter = LineString([(180.0, -90.0), (180.0, 90.0)])
    try:
        pieces = split(geom, splitter)
    except Exception:
        pieces = [geom]

    normalized = []
    for piece in getattr(pieces, "geoms", pieces):
        if piece.is_empty:
            continue
        minx, _, maxx, _ = piece.bounds
        right_side = minx >= 180.0 or piece.representative_point().x > 180.0

        def shift_lon(x, y, z=None):
            threshold = 180.0 if right_side else 180.0 + 1e-12
            shifted = x - 360.0 if x >= threshold else x
            if z is None:
                return shifted, y
            return shifted, y, z

        normalized_piece = transform(shift_lon, piece)
        if not normalized_piece.is_empty:
            normalized.append(normalized_piece)

    if not normalized:
        return geom
    return unary_union(normalized)


def _assemble_relation_geometry(
    relation: dict,
    *,
    ways: dict[int, list[int]],
    nodes: dict[int, tuple[float, float]],
) -> tuple[object | None, dict]:
    unwrap_antimeridian = _relation_needs_antimeridian_unwrap(
        relation,
        ways=ways,
        nodes=nodes,
    )
    outer_lines, missing_outer_ways, missing_outer_nodes = _member_lines(
        relation,
        ways=ways,
        nodes=nodes,
        role="outer",
        unwrap_antimeridian=unwrap_antimeridian,
    )
    inner_lines, missing_inner_ways, missing_inner_nodes = _member_lines(
        relation,
        ways=ways,
        nodes=nodes,
        role="inner",
        unwrap_antimeridian=unwrap_antimeridian,
    )
    outer_lines, endpoint_repair = _append_dangling_endpoint_repair(outer_lines)
    outer_geom = _polygonize_lines(outer_lines)
    inner_geom = _polygonize_lines(inner_lines)

    diagnostic = {
        "missing_outer_way_count": len(set(missing_outer_ways)),
        "missing_inner_way_count": len(set(missing_inner_ways)),
        "missing_outer_node_count": len(set(missing_outer_nodes)),
        "missing_inner_node_count": len(set(missing_inner_nodes)),
        "outer_line_count": len(outer_lines),
        "inner_line_count": len(inner_lines),
        "antimeridian_unwrapped": unwrap_antimeridian,
        **endpoint_repair,
    }

    if outer_geom is None:
        diagnostic["failure_reason"] = "outer_polygonize_failed"
        return None, diagnostic
    geom = outer_geom
    if inner_geom is not None:
        geom = geom.difference(inner_geom)
    if unwrap_antimeridian:
        geom = _normalize_antimeridian_geometry(geom)
    if geom.is_empty:
        diagnostic["failure_reason"] = "empty_geometry"
        return None, diagnostic
    if not isinstance(geom, (Polygon, MultiPolygon)):
        geom = unary_union([part for part in getattr(geom, "geoms", []) if isinstance(part, (Polygon, MultiPolygon))])
    if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
        diagnostic["failure_reason"] = "non_polygon_geometry"
        return None, diagnostic
    return geom, diagnostic


def _apply_geometry_policy(gdf: gpd.GeoDataFrame, *, levels: list[int], profile: AdminProfile) -> tuple[gpd.GeoDataFrame, dict]:
    stats = {"simplified_by_level": {}, "invalid_fixed_by_level": {}}
    for lvl in levels:
        policy, _ = _resolve_level_policy(lvl, profile, None)
        if not isinstance(policy, AdminLevelPolicy):
            continue
        mask = gdf.level == lvl
        if policy.simplify:
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].simplify(
                policy.simplify_tolerance,
                preserve_topology=True,
            )
            stats["simplified_by_level"][str(lvl)] = int(mask.sum())
        if policy.fix_invalid:
            invalid_mask = mask & (~gdf.geometry.is_valid)
            stats["invalid_fixed_by_level"][str(lvl)] = int(invalid_mask.sum())
            if invalid_mask.any():
                gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].buffer(0)
    return gdf, stats


def build_stitched_admin_dataset(
    *,
    pbf_paths: list[Path],
    output_path: Path,
    report_path: Path,
    levels: Iterable[int],
    profile: AdminProfile,
    country_code: str,
    country_name: str,
    level_labels: dict[int, str],
    id_prefix: str,
    cache_dir: Path | None = None,
    allowed_level4_names: set[str] | None = None,
) -> Path:
    start_time = datetime.now()
    levels = sorted({int(level) for level in levels})
    pbf_paths = [Path(path) for path in pbf_paths]

    fingerprint = _source_fingerprint(pbf_paths=pbf_paths, levels=levels)
    cache_root = Path(cache_dir) if cache_dir is not None else output_path.parent / "_stitch_cache"

    relations_cache_path = _cache_path(cache_root, fingerprint, "relations.pkl")
    relations_payload = _load_pickle(relations_cache_path)
    if relations_payload is None:
        relation_collector = _scan_relations(pbf_paths, levels=levels, profile=profile)
        relations_payload = {
            "relations": relation_collector.relations,
            "level_counts": dict(relation_collector.level_counts),
            "duplicate_relation_count": relation_collector.duplicate_relation_count,
        }
        _write_pickle(relations_cache_path, relations_payload)
    relations = relations_payload["relations"]
    required_way_ids = {
        int(member["ref"])
        for relation in relations.values()
        for member in relation["way_members"]
    }

    ways_cache_path = _cache_path(cache_root, fingerprint, "ways.pkl")
    ways_payload = _load_pickle(ways_cache_path)
    if ways_payload is None:
        way_collector = _scan_ways(pbf_paths, required_way_ids)
        ways_payload = {
            "ways": way_collector.ways,
            "variant_way_ids": sorted(way_collector.variant_way_ids),
        }
        _write_pickle(ways_cache_path, ways_payload)
    ways = ways_payload["ways"]
    required_node_ids = {
        int(node_id)
        for node_refs in ways.values()
        for node_id in node_refs
    }
    nodes_cache_path = _cache_path(cache_root, fingerprint, "nodes.pkl")
    nodes_payload = _load_pickle(nodes_cache_path)
    if nodes_payload is None:
        node_collector = _scan_nodes(pbf_paths, required_node_ids)
        nodes_payload = {
            "nodes": node_collector.nodes,
            "variant_node_ids": sorted(node_collector.variant_node_ids),
        }
        _write_pickle(nodes_cache_path, nodes_payload)
    nodes = nodes_payload["nodes"]

    rows = []
    failure_samples = []
    failure_counts = Counter()
    success_counts = Counter()
    for relation_id, relation in sorted(relations.items()):
        geom, diagnostic = _assemble_relation_geometry(
            relation,
            ways=ways,
            nodes=nodes,
        )
        if geom is None:
            reason = diagnostic.get("failure_reason", "unknown")
            failure_counts[reason] += 1
            if len(failure_samples) < 50:
                failure_samples.append(
                    {
                        "relation_id": relation_id,
                        "name": relation.get("name"),
                        "level": relation.get("level"),
                        **diagnostic,
                    }
                )
            continue
        level = int(relation["level"])
        success_counts[level] += 1
        rows.append(
            {
                "id": f"{id_prefix}_r{relation_id}",
                "osm_id": f"r{relation_id}",
                "level": level,
                "name": relation.get("name"),
                "names": relation.get("names"),
                "geometry": geom,
            }
        )

    if not rows:
        raise RuntimeError("US stitched admin build produced no assembled polygons.")

    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs="EPSG:4326")
    gdf, policy_stats = _apply_geometry_policy(gdf, levels=levels, profile=profile)
    scope_filter_stats = {"enabled": False}
    if allowed_level4_names:
        allowed_names = {name for name in allowed_level4_names if name}
        allowed_level4 = gdf[(gdf.level == 4) & (gdf.name.isin(allowed_names))].copy()
        if not allowed_level4.empty:
            scope_geom = unary_union(list(allowed_level4.geometry))
            before_count = len(gdf)
            rep_points = gdf.geometry.representative_point()
            keep_mask = (gdf.level == 4) & (gdf.name.isin(allowed_names))
            keep_mask = keep_mask | ((gdf.level != 4) & rep_points.apply(scope_geom.covers))
            gdf = gdf[keep_mask].copy()
            scope_filter_stats = {
                "enabled": True,
                "allowed_level4_count": int(len(allowed_level4)),
                "before_count": int(before_count),
                "after_count": int(len(gdf)),
                "removed_count": int(before_count - len(gdf)),
            }
    gdf = TopologyEngine.infer_hierarchy(gdf, levels)
    gdf = gdf.replace({pd.NA: None})

    meta_info = {
        "country": country_code,
        "country_name": country_name,
        "country_geometry_filter_applied": False,
        "levels": levels,
        "source": "OpenStreetMap Geofabrik US state extracts stitched below area assembly",
        "source_extract_count": len(pbf_paths),
        "generated_at": datetime.utcnow().isoformat(),
        "processing_time_sec": (datetime.now() - start_time).total_seconds(),
    }
    final_json = serialize_output(gdf, levels, profile, meta_info)
    for lvl, label in level_labels.items():
        if str(lvl) in final_json["admin_by_level"]:
            final_json[label] = final_json["admin_by_level"][str(lvl)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(final_json, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "pbf_paths": [str(path) for path in pbf_paths],
        "levels": levels,
        "relation_count": len(relations),
        "required_way_count": len(required_way_ids),
        "present_way_count": len(ways),
        "missing_way_count": len(required_way_ids - set(ways)),
        "required_node_count": len(required_node_ids),
        "present_node_count": len(nodes),
        "missing_node_count": len(required_node_ids - set(nodes)),
        "assembled_count": len(rows),
        "assembled_by_level": {str(k): v for k, v in sorted(success_counts.items())},
        "failure_count": sum(failure_counts.values()),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failure_samples": failure_samples,
        "variant_way_count": len(ways_payload.get("variant_way_ids", [])),
        "variant_node_count": len(nodes_payload.get("variant_node_ids", [])),
        "cache": {
            "fingerprint": fingerprint,
            "cache_dir": str(_cache_path(cache_root, fingerprint, "")),
            "relations_cache": str(relations_cache_path),
            "ways_cache": str(ways_cache_path),
            "nodes_cache": str(nodes_cache_path),
        },
        "geometry_policy": policy_stats,
        "scope_filter": scope_filter_stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{datetime.now()}] US stitched admin dataset written: {output_path}", flush=True)
    print(f"[{datetime.now()}] US stitched admin report written: {report_path}", flush=True)
    return output_path
