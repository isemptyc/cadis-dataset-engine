"""Derive a consumer profile from a published Cadis dataset release.

A profile keeps a subset of a release's administrative levels for a consumer
that does not need every level (for example PhotoLens, which displays only its
configured geographic ranks). It is derived deterministically from the
published runtime artifacts, not rebuilt from OpenStreetMap, so every retained
feature keeps its exact feature_id, names and quantized geometry bytes.

Retained levels are the requested levels plus the release's country-scope
level (the coarsest level carrying country_scope_flag, or the coarsest level
when none is flagged), because the runtime derives its country scope, nearby
and offshore behavior from that level.

Features of other levels are dropped unless they fill a gap: a feature whose
area is not covered by the retained polygons its lookups depend on (the
hierarchy-repair parent level when retained, otherwise every retained level)
is kept. Without it, a point there would lose the hierarchy repair or fall
back to nearest-feature resolution, and resolve differently from the full
release. Coverage uses a small tolerance so quantization slivers along shared
boundaries do not count as gaps.

The derivation rewrites, consistently:

- geometry.ffsf: only retained and gap-filling features; part, geometry and ring tables are
  re-indexed and geometry bytes are copied verbatim (no re-quantization);
- geometry_meta.json: retained features in FFSF order, each parent_id pointing
  at its nearest retained ancestor;
- hierarchy.json: retained nodes with parents re-linked the same way, and
  branch identity recomputed when the release carries it. A retained feature
  with no ancestor at a coarser retained level (many releases leave lower
  units parentless) gets the retained coarser feature containing its
  representative point as its parent, so consumers can tell which parent a
  unit belongs to (a Leningrad Oblast district is not under Saint
  Petersburg even where boundary polygons overlap);
- runtime_policy.json: allowed levels and shapes projected onto the retained
  levels (a projected shape is "ok" when any source shape projecting to it is
  "ok"), and hierarchy repair kept only while its parent level is retained;
- dataset_release_manifest.json: the profile version, recomputed checksums and
  a `derived_profile` block naming the source release.

The source release is never modified. The full dataset remains the baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DERIVED_PROFILE_SCHEMA = "cadis.derived-profile/1"
# Degrees; about 30 m. Shared boundaries are quantized per part, so levels
# disagree by a few meters along every border.
DEFAULT_GAP_TOLERANCE = 3e-4
# Marks a parent inferred by representative-point containment, never
# recorded in the source release (geometry_meta.json and hierarchy.json).
PARENT_SOURCE_INFERRED = "inferred_containment"
RUNTIME_FILES = ("geometry.ffsf", "geometry_meta.json", "hierarchy.json", "runtime_policy.json")


def _load_runtime_hierarchy():
    # Loaded by path: the ffsf package __init__ imports the shapely-based
    # exporter, which derivation does not need.
    spec = importlib.util.spec_from_file_location(
        "cadis_runtime_hierarchy", HERE / "ffsf" / "runtime_hierarchy.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


# MARK: - FFSF


def read_ffsf(data: bytes) -> dict[str, Any]:
    if data[:4] != b"FFSF":
        raise ValueError("not an FFSF file")
    version, feature_count, part_count = struct.unpack_from("<III", data, 4)
    if version not in (2, 3):
        raise ValueError(f"unsupported FFSF version {version}")
    offset = 16
    features = [struct.unpack_from("<4I", data, offset + 16 * i) for i in range(feature_count)]
    offset += 16 * feature_count
    bboxes = [data[offset + 16 * i: offset + 16 * (i + 1)] for i in range(part_count)]
    offset += 16 * part_count
    geoms = [struct.unpack_from("<4I", data, offset + 16 * i) for i in range(part_count)]
    offset += 16 * part_count
    ring_total = sum(g[3] for g in geoms)
    rings = list(struct.unpack_from(f"<{ring_total}I", data, offset)) if ring_total else []
    offset += 4 * ring_total
    return {
        "version": version,
        "features": features,
        "bboxes": bboxes,
        "geoms": geoms,
        "rings": rings,
        "geometry": data[offset:],
    }


def subset_ffsf(ffsf: dict[str, Any], keep: list[int]) -> bytes:
    """Re-encode `ffsf` with only the feature indices in `keep`, in order."""
    features_out: list[tuple[int, int, int, int]] = []
    bboxes_out: list[bytes] = []
    geoms_out: list[tuple[int, int, int, int]] = []
    rings_out: list[int] = []
    geometry_out = bytearray()
    geometry = ffsf["geometry"]
    for index in keep:
        string_offset, string_len, part_start, part_count = ffsf["features"][index]
        new_part_start = len(geoms_out)
        for part in range(part_start, part_start + part_count):
            byte_offset, byte_len, ring_start, ring_count = ffsf["geoms"][part]
            new_offset = len(geometry_out)
            geometry_out.extend(geometry[byte_offset: byte_offset + byte_len])
            new_ring_start = len(rings_out)
            rings_out.extend(ffsf["rings"][ring_start: ring_start + ring_count])
            geoms_out.append((new_offset, byte_len, new_ring_start, ring_count))
            bboxes_out.append(ffsf["bboxes"][part])
        features_out.append((string_offset, string_len, new_part_start, part_count))

    out = bytearray(b"FFSF")
    out.extend(struct.pack("<III", ffsf["version"], len(features_out), len(geoms_out)))
    for entry in features_out:
        out.extend(struct.pack("<4I", *entry))
    for bbox in bboxes_out:
        out.extend(bbox)
    for entry in geoms_out:
        out.extend(struct.pack("<4I", *entry))
    if rings_out:
        out.extend(struct.pack(f"<{len(rings_out)}I", *rings_out))
    out.extend(geometry_out)
    return bytes(out)


def feature_geometry(ffsf: dict[str, Any], index: int):
    """Dequantized shapely geometry of one FFSF feature, with the runtime's
    containment semantics: a point is inside a part when it is inside the
    outer ring and inside none of the hole rings, each ring tested by ray
    casting (even-odd, so a self-intersecting ring excludes its overlaps)."""
    import numpy as np
    import shapely
    from shapely.geometry import Polygon

    def ring_region(points):
        valid = shapely.make_valid(Polygon(points), method="linework")
        areas = [part for part in shapely.get_parts(valid) if part.geom_type in ("Polygon", "MultiPolygon")]
        return shapely.union_all(areas) if areas else Polygon()

    _, _, part_start, part_count = ffsf["features"][index]
    polygons = []
    for part in range(part_start, part_start + part_count):
        byte_offset, byte_len, ring_start, ring_count = ffsf["geoms"][part]
        minx, miny, maxx, maxy = struct.unpack("<4f", ffsf["bboxes"][part])
        values = np.frombuffer(ffsf["geometry"], dtype="<u2", count=byte_len // 2, offset=byte_offset)
        points = values.reshape(-1, 2).astype(np.float64) / 65535.0
        points[:, 0] = minx + points[:, 0] * (maxx - minx)
        points[:, 1] = miny + points[:, 1] * (maxy - miny)
        rings, cursor = [], 0
        for count in ffsf["rings"][ring_start: ring_start + ring_count]:
            rings.append(points[cursor: cursor + count])
            cursor += count
        rings = [ring for ring in rings if len(ring) >= 3]
        if rings:
            region = ring_region(rings[0])
            for hole in rings[1:]:
                region = region.difference(ring_region(hole))
            polygons.append(region)
    return shapely.union_all(polygons) if polygons else Polygon()


def gap_fillers(
    ffsf: dict[str, Any],
    meta: list[dict[str, Any]],
    retained: set[int],
    policy: dict[str, Any],
    tolerance: float,
) -> list[int]:
    """Indices of dropped features not covered by the retained polygons their
    lookups depend on."""
    import shapely
    from shapely.ops import unary_union

    parent = (policy.get("hierarchy_repair_rules") or {}).get("parent_level")
    target = {parent} if parent in retained else retained
    coverage = unary_union([
        feature_geometry(ffsf, i) for i, m in enumerate(meta) if m["level"] in target
    ]).buffer(tolerance)
    shapely.prepare(coverage)
    candidates = [i for i, m in enumerate(meta) if m["level"] not in retained]
    if not candidates:
        return []
    geometries = [feature_geometry(ffsf, i) for i in candidates]
    covered = shapely.contains(coverage, geometries)
    return [i for i, is_covered in zip(candidates, covered) if not is_covered]


def spatial_parents(
    ffsf: dict[str, Any],
    meta: list[dict[str, Any]],
    keep: list[int],
    hierarchy_nodes: list[dict[str, Any]],
    retained: set[int],
    prefix: str,
) -> dict[str, str]:
    """Inferred parents for retained features with no recorded ancestor at any
    coarser retained level.

    Rule: take the finest coarser retained level that has any polygon
    containing the feature's representative point. Exactly one containing
    unit becomes the parent. Two or more (overlapping candidates) is
    ambiguous: nothing is assigned and coarser levels are not tried, so an
    overlap never resolves to an arbitrary unit. No containing unit at any
    level leaves the ancestry unknown.

    Representative-point containment is a practical inference, not proof of
    administrative membership; the written parents are marked
    `parent_source: "inferred_containment"` so they stay distinguishable from
    source-recorded parents. Returns (parents by feature id, counts)."""
    import shapely
    from shapely.geometry import Point

    parents = {n["id"]: n.get("parent_id") for n in hierarchy_nodes}
    levels = {n["id"]: n["level"] for n in hierarchy_nodes}
    parents.update({m["feature_id"]: m.get("parent_id") for m in meta})
    levels.update({m["feature_id"]: m["level"] for m in meta})
    ancestry = _Ancestry(parents, levels, prefix)
    kept = set(keep)
    coarsest = min(retained)
    containers: dict[int, tuple[list[int], Any]] = {}
    for level in sorted(retained):
        indices = [i for i in kept if meta[i]["level"] == level]
        if indices:
            geometries = [feature_geometry(ffsf, i) for i in indices]
            containers[level] = (indices, shapely.STRtree(geometries), geometries)

    out: dict[str, str] = {}
    counts = {"assigned": 0, "ambiguous": 0, "no_container": 0}
    for i in sorted(kept):
        entry = meta[i]
        level = entry["level"]
        if level not in retained or level == coarsest:
            continue
        coarser = {l for l in retained if l < level}
        # Any recorded ancestor at a coarser retained level settles it.
        seen: set[str] = set()
        current = ancestry.canonical(entry.get("parent_id"))
        has_ancestor = False
        while current is not None and current not in seen:
            seen.add(current)
            if ancestry.levels.get(current) in coarser:
                has_ancestor = True
                break
            current = ancestry.canonical(ancestry.parents.get(current))
        if has_ancestor:
            continue
        lon, lat = entry["representative_point_exact"]
        point = Point(lon, lat)
        outcome = "no_container"
        for parent_level in sorted(coarser, reverse=True):
            if parent_level not in containers:
                continue
            indices, tree, geometries = containers[parent_level]
            hits = [indices[j] for j in tree.query(point) if geometries[j].contains(point)]
            if len(hits) == 1:
                out[entry["feature_id"]] = meta[hits[0]]["feature_id"]
                outcome = "assigned"
                break
            if len(hits) > 1:
                outcome = "ambiguous"
                break
        counts[outcome] += 1
    return out, counts


# MARK: - Levels and parents


def scope_level(meta: list[dict[str, Any]]) -> int:
    """The level the runtime derives country scope from (FFSFIndex)."""
    flagged = [m["level"] for m in meta if m.get("country_scope_flag") is True]
    return min(flagged) if flagged else min(m["level"] for m in meta)


class _Ancestry:
    """Nearest retained ancestor over a parent graph whose ids may appear with
    or without the dataset's `<iso2>_` prefix."""

    def __init__(self, parents: dict[str, str | None], levels: dict[str, int], prefix: str):
        self.parents = parents
        self.levels = levels
        self.prefix = prefix

    def canonical(self, node_id: str | None) -> str | None:
        if node_id is None:
            return None
        if node_id in self.parents:
            return node_id
        if self.prefix + node_id in self.parents:
            return self.prefix + node_id
        if node_id.startswith(self.prefix) and node_id[len(self.prefix):] in self.parents:
            return node_id[len(self.prefix):]
        return None

    def nearest_kept(self, parent_id: str | None, kept_ids: set[str]) -> str | None:
        """The nearest ancestor (starting at `parent_id`) whose id, with or
        without the prefix, is in `kept_ids`; returned in this graph's form."""
        seen: set[str] = set()
        current = self.canonical(parent_id)
        while current is not None and current not in seen:
            seen.add(current)
            if self.is_kept(current, kept_ids):
                return current
            current = self.canonical(self.parents.get(current))
        return None

    def is_kept(self, node_id: str, kept_ids: set[str]) -> bool:
        return (
            node_id in kept_ids
            or self.prefix + node_id in kept_ids
            or (node_id.startswith(self.prefix) and node_id[len(self.prefix):] in kept_ids)
        )


def derive_meta(
    meta: list[dict[str, Any]], keep: list[int], hierarchy_nodes: list[dict[str, Any]],
    retained: set[int], prefix: str, overrides: dict[str, str] | None = None,
) -> list[dict]:
    # Parents may be hierarchy-only nodes (no polygon), so walk the union.
    parents = {n["id"]: n.get("parent_id") for n in hierarchy_nodes}
    levels = {n["id"]: n["level"] for n in hierarchy_nodes}
    parents.update({m["feature_id"]: m.get("parent_id") for m in meta})
    levels.update({m["feature_id"]: m["level"] for m in meta})
    ancestry = _Ancestry(parents, levels, prefix)
    kept_ids = {meta[i]["feature_id"] for i in keep} | {
        n["id"] for n in hierarchy_nodes if n["level"] in retained
    }
    out = []
    for i in keep:
        entry = dict(meta[i])
        if entry.get("parent_id") is not None:
            entry["parent_id"] = ancestry.nearest_kept(entry["parent_id"], kept_ids)
        if overrides and entry["feature_id"] in overrides:
            entry["parent_id"] = overrides[entry["feature_id"]]
            entry["parent_source"] = PARENT_SOURCE_INFERRED
        out.append(entry)
    return out


def derive_hierarchy(
    hierarchy: dict[str, Any], kept_ids: set[str], retained: set[int], prefix: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    nodes = hierarchy["nodes"]
    ancestry = _Ancestry(
        {n["id"]: n.get("parent_id") for n in nodes},
        {n["id"]: n["level"] for n in nodes},
        prefix,
    )
    core = ("id", "level", "name", "names", "parent_id")
    # A retained-level node is kept even without a polygon: some releases
    # resolve a level only through hierarchy repair (TW 高雄市 has no level-4
    # polygon and is repaired from its districts).
    kept_ids = kept_ids | {n["id"] for n in nodes if n["level"] in retained}
    derived_nodes = []
    for node in nodes:
        if not ancestry.is_kept(node["id"], kept_ids):
            continue
        entry = {key: node[key] for key in core if key in node}
        entry["parent_id"] = ancestry.nearest_kept(node.get("parent_id"), kept_ids)
        if overrides:
            node_id = node["id"]
            override = overrides.get(node_id) or overrides.get(prefix + node_id)
            if override is not None:
                # Written in this graph's id form.
                entry["parent_id"] = (
                    override[len(prefix):]
                    if override.startswith(prefix) and not node_id.startswith(prefix)
                    else override
                )
                entry["parent_source"] = PARENT_SOURCE_INFERRED
        derived_nodes.append(entry)
    if "branch_identity_version" in hierarchy:
        return _load_runtime_hierarchy().build_runtime_hierarchy_payload(derived_nodes)
    out = {key: value for key, value in hierarchy.items() if key != "nodes"}
    out["nodes"] = derived_nodes
    return out


def derive_policy(policy: dict[str, Any], retained: set[int]) -> dict[str, Any]:
    out = dict(policy)
    out["allowed_levels"] = [level for level in policy["allowed_levels"] if level in retained]

    status_by_shape: dict[tuple[int, ...], str] = {}
    for entry in policy.get("shape_status", []):
        status_by_shape[tuple(sorted(entry["levels"]))] = entry["status"]

    projected: dict[tuple[int, ...], str] = {}
    for shape in policy["allowed_shapes"]:
        source = tuple(sorted(shape))
        target = tuple(level for level in source if level in retained)
        if not target:
            continue
        status = status_by_shape.get(source, "partial")
        if projected.get(target) != "ok":
            projected[target] = "ok" if status == "ok" else "partial"
    shapes = sorted(projected)
    out["allowed_shapes"] = [list(shape) for shape in shapes]
    if "shape_status" in policy:
        out["shape_status"] = [{"levels": list(shape), "status": projected[shape]} for shape in shapes]

    for key in ("hierarchy_repair_rules", "repair_rules"):
        rules = policy.get(key)
        if rules is None:
            continue
        if rules.get("parent_level") in retained:
            kept = dict(rules)
            kept["child_levels"] = [level for level in rules.get("child_levels", []) if level in retained]
            out[key] = kept
        else:
            out.pop(key)
    if "hierarchy_repair_rules" not in out and isinstance(out.get("layers"), dict):
        # Without rules the runtime would repair every missing level from
        # every hit; a profile that loses its repair parent repairs nothing.
        layers = dict(out["layers"])
        layers["hierarchy_required"] = False
        out["layers"] = layers
    return out


# MARK: - Profile


def derive_profile(
    source_dir: Path,
    output_dir: Path,
    *,
    levels: set[int],
    profile: str,
    revision: int,
    tolerance: float = DEFAULT_GAP_TOLERANCE,
) -> dict[str, Any]:
    manifest_bytes = (source_dir / "dataset_release_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    iso2 = manifest["country_iso"].upper()
    prefix = iso2.lower() + "_"

    meta = json.loads((source_dir / "geometry_meta.json").read_text(encoding="utf-8"))
    policy = json.loads((source_dir / "runtime_policy.json").read_text(encoding="utf-8"))
    ffsf = read_ffsf((source_dir / "geometry.ffsf").read_bytes())
    if len(ffsf["features"]) != len(meta):
        raise ValueError("geometry_meta.json does not match geometry.ffsf")
    retained = set(levels) | {scope_level(meta)}
    fillers = set(gap_fillers(ffsf, meta, retained, policy, tolerance))
    keep = [i for i, m in enumerate(meta) if m["level"] in retained or i in fillers]
    kept_ids = {meta[i]["feature_id"] for i in keep}
    runtime_levels = retained | {meta[i]["level"] for i in fillers}

    hierarchy = json.loads((source_dir / "hierarchy.json").read_text(encoding="utf-8"))
    overrides, parent_counts = spatial_parents(ffsf, meta, keep, hierarchy["nodes"], retained, prefix)
    files = {
        "geometry.ffsf": subset_ffsf(ffsf, keep),
        "geometry_meta.json": _dump_json(
            derive_meta(meta, keep, hierarchy["nodes"], retained, prefix, overrides)
        ),
        "hierarchy.json": _dump_json(
            derive_hierarchy(hierarchy, kept_ids, retained, prefix, overrides)
        ),
        "runtime_policy.json": _dump_json(derive_policy(policy, runtime_levels)),
    }

    derived_manifest = dict(manifest)
    derived_manifest["dataset_version"] = f"{manifest['dataset_version']}-{profile}.{revision}"
    derived_manifest["checksums"] = {
        "files": {
            name: {"sha256": _sha256_bytes(data), "size": len(data)}
            for name, data in sorted(files.items())
        }
    }
    derived_manifest["derived_profile"] = {
        "schema": DERIVED_PROFILE_SCHEMA,
        "profile": profile,
        "revision": revision,
        "requested_levels": sorted(levels),
        "retained_levels": sorted(retained),
        "gap_filler_features": len(fillers),
        "gap_tolerance_degrees": tolerance,
        "inferred_parents": {
            "source": PARENT_SOURCE_INFERRED,
            "rule": "representative point in exactly one unit at the finest coarser retained level",
            **parent_counts,
        },
        "source": {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    manifest_out = json.dumps(derived_manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    (output_dir / "dataset_release_manifest.json").write_bytes(manifest_out)
    return {
        "country_iso": iso2,
        "dataset_id": manifest["dataset_id"],
        "dataset_version": derived_manifest["dataset_version"],
        "manifest_sha256": _sha256_bytes(manifest_out),
        "retained_levels": sorted(retained),
        "gap_filler_features": len(fillers),
        "spatial_parents": len(overrides),
        "inferred_parents": parent_counts,
        "feature_count": len(keep),
        "source_feature_count": len(meta),
        "bytes": sum(len(data) for data in files.values()) + len(manifest_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive a level-subset profile from a Cadis dataset release.")
    parser.add_argument("--source", required=True, type=Path, help="Extracted release directory")
    parser.add_argument("--output", required=True, type=Path, help="Profile output directory")
    parser.add_argument("--levels", required=True, help="Comma-separated admin levels to retain, e.g. 4,6")
    parser.add_argument("--profile", required=True, help="Profile name, e.g. photolens")
    parser.add_argument("--revision", required=True, type=int, help="Profile revision")
    args = parser.parse_args()
    levels = {int(value) for value in args.levels.split(",") if value.strip()}
    summary = derive_profile(
        args.source, args.output, levels=levels, profile=args.profile, revision=args.revision
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
