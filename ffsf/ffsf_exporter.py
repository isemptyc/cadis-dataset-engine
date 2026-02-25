from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Iterable

from shapely.geometry import shape
from shapely.prepared import prep


def _iter_features(admin_data: dict) -> list[dict]:
    """
    Iterate features from an admin dataset JSON file in deterministic order.

    Ordering rule (normative for exporter v0):
    - Features are iterated by admin level ascending (numeric sort of keys)
    - Within each level, original list order is preserved

    Notes:
    - The exporter MUST NOT reinterpret hierarchy or semantics.
    - This ordering is used only to produce a stable FeatureIndex layout.
    - Semantic meaning of a feature is bound to feature_id, not to index order.
    """
    admin_by_level = admin_data.get("admin_by_level", {})
    features: list[dict] = []
    for level in sorted(admin_by_level.keys(), key=int):
        features.extend(admin_by_level[level])
    return features


def _iter_outer_rings(geometry: dict) -> Iterable[list]:
    """
    Yield outer rings only from a Polygon or MultiPolygon geometry.

    FFSF v1 DESIGN DECISION:
    - Inner rings (holes) are intentionally NOT encoded.
    - Only the outer ring (ring[0]) of each polygon is emitted.
    - No ring boundaries or hole markers exist in GeometryData.

    Rationale:
    - FFSF is a frontend spatial ABI, not a full GIS geometry format.
    - Administrative point-in-polygon semantics do not depend on holes.

    To Be Decided (future versions):
    - Whether to support holes via explicit ring markers (FFSF v2+).
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        if coords:
            yield coords[0]
        return
    if geom_type == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly[0]
        return
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def _iter_parts_with_rings(geometry: dict) -> Iterable[list[list]]:
    """
    Yield polygon parts with all rings (outer + inner).

    For Polygon: yields a single list of rings.
    For MultiPolygon: yields one list of rings per polygon.
    """
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        if coords:
            yield coords
        return
    if geom_type == "MultiPolygon":
        for poly in coords:
            if poly:
                yield poly
        return
    raise ValueError(f"Unsupported geometry type: {geom_type}")


def _round_half_up(value: float) -> int:
    """
    Round-half-up implementation.

    This function exists to avoid Python's built-in round(), which uses
    banker's rounding and is therefore unsuitable for deterministic
    quantization.

    Rounding rule:
    - Values with fractional part >= 0.5 are rounded up.
    """
    return int(math.floor(value + 0.5))


def _quantize(value: float, min_value: float, span: float) -> int:
    """
    Quantize a single coordinate component into uint16 space.

    Quantization rule (FFSF §3.2):
    - Local, per-part bounding box quantization
    - Range mapped to [0, 65535]
    - Half-up rounding

    Edge cases:
    - If span == 0, returns 0 (degenerate geometry axis)
    - Values outside range are clamped to [0, 65535]
    """
    if span == 0:
        return 0
    scaled = (value - min_value) / span * 65535.0
    if scaled <= 0:
        return 0
    if scaled >= 65535:
        return 65535
    return _round_half_up(scaled)


def export_cadis_to_ffsf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    version: int = 3,
    country_geometry_path: str | Path | None = None,
) -> dict:
    """
    Convert an admin dataset JSON file into FFSF binary format.

    ROLE & RESPONSIBILITY
    ---------------------
    This function is a **pure geometry re-encoding compiler**.

    It MUST:
    - Preserve feature identities (feature_id) exactly as provided
    - Re-encode spatial geometry into FFSF binary layout

    It MUST NOT:
    - Perform semantic inference or cleanup
    - Modify hierarchy, admin_level, or naming
    - Query external services

    DATASET ASSUMPTIONS
    -------------------
    - Input JSON is a canonical admin dataset
    - Each feature has a stable and pre-assigned feature_id at feature["id"]
    - Geometry is Polygon or MultiPolygon (WGS84 lon/lat)

    FFSF v1 vs v2 DESIGN DECISIONS
    ------------------------------
    - v1 encodes only outer rings and drops holes
    - v2 encodes ring boundaries to preserve holes
    - GeometryData remains a flat uint16 stream
    - StringPool is omitted (StringOffset/StringLen are set to 0)
    - One .ffsf.bin output per input dataset (no chunking)

    To Be Decided (future versions):
    - Chunking strategy
    - StringPool usage (names, labels)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if version not in (1, 2, 3):
        raise ValueError(f"Unsupported FFSF version: {version}")

    admin_data = json.loads(input_path.read_text(encoding="utf-8"))
    meta = admin_data.get("meta", {})
    country = meta.get("country")
    if not country:
        raise ValueError("admin_data.meta.country is required for meta output")
    features = _iter_features(admin_data)
    prepared_country_geom = None
    if country_geometry_path is not None:
        country_geometry_path = Path(country_geometry_path)
        country_data = json.loads(country_geometry_path.read_text(encoding="utf-8"))
        prepared_country_geom = prep(shape(country_data["geometry"]))

    feature_index_entries: list[tuple[int, int, int, int]] = []
    part_bboxes: list[tuple[float, float, float, float]] = []
    geom_index_entries: list[tuple[int, ...]] = []
    ring_index_entries: list[int] = []
    geometry_data = bytearray()

    part_index = 0
    feature_ids: list[str] = []
    feature_meta_by_index: list[dict] = []

    for feature in features:
        # feature_id is treated as a stable semantic anchor and is NOT modified
        if "id" not in feature:
            raise ValueError("feature.id is required and must be stable")

        feature_id = feature.get("id")
        if not isinstance(feature_id, str) or not feature_id:
            raise ValueError("feature.id must be a non-empty string")

        feature_ids.append(feature_id)
        rep_point = shape(feature["geometry"]).representative_point()
        country_scope_flag = None
        if prepared_country_geom is not None:
            country_scope_flag = bool(prepared_country_geom.covers(rep_point))

        feature_meta_by_index.append({
            "feature_id": feature_id,
            "level": feature.get("level"),
            "name": feature.get("name"),
            "parent_id": feature.get("parent"),
            "representative_point_exact": [rep_point.x, rep_point.y],
            "country_scope_flag": country_scope_flag,
        })
        part_start = part_index
        part_count = 0

        for rings in _iter_parts_with_rings(feature["geometry"]):
            if version == 1:
                rings_to_encode = [rings[0]] if rings else []
            else:
                rings_to_encode = list(rings)

            rings_to_encode = [ring for ring in rings_to_encode if ring]
            if not rings_to_encode:
                continue

            # Bounding box is computed per polygon part (all rings)
            minx = min(pt[0] for ring in rings_to_encode for pt in ring)
            maxx = max(pt[0] for ring in rings_to_encode for pt in ring)
            miny = min(pt[1] for ring in rings_to_encode for pt in ring)
            maxy = max(pt[1] for ring in rings_to_encode for pt in ring)

            part_bboxes.append((minx, miny, maxx, maxy))

            spanx = maxx - minx
            spany = maxy - miny
            byte_offset = len(geometry_data)

            # NOTE: Rings may include a closing coordinate (first == last).
            # Frontend runtime MUST handle this correctly.
            if version == 1:
                ring = rings_to_encode[0]
                coords: list[int] = []
                for x, y in ring:
                    coords.append(_quantize(x, minx, spanx))
                    coords.append(_quantize(y, miny, spany))

                if coords:
                    geometry_data.extend(
                        struct.pack("<" + "H" * len(coords), *coords)
                    )
                byte_len = len(coords) * 2
                geom_index_entries.append((byte_offset, byte_len))
            else:
                ring_start = len(ring_index_entries)
                ring_count = 0

                for ring in rings_to_encode:
                    ring_index_entries.append(len(ring))
                    ring_count += 1
                    coords: list[int] = []
                    for x, y in ring:
                        coords.append(_quantize(x, minx, spanx))
                        coords.append(_quantize(y, miny, spany))
                    if coords:
                        geometry_data.extend(
                            struct.pack("<" + "H" * len(coords), *coords)
                        )

                byte_len = len(geometry_data) - byte_offset
                geom_index_entries.append(
                    (byte_offset, byte_len, ring_start, ring_count)
                )

            part_index += 1
            part_count += 1

        if part_count == 0:
            raise ValueError(
                f"Feature {feature.get('id')} has no polygon parts"
            )

        # FeatureIndex entry:
        # (StringOffset, StringLen, PartStartIdx, PartCount)
        # StringOffset/StringLen are reserved and set to 0 in FFSF v1/v2
        feature_index_entries.append((0, 0, part_start, part_count))

    # Header: Magic, Version, FeatureCount, TotalPartCount
    header = bytearray()
    header.extend(b"FFSF")
    header.extend(struct.pack("<I", version))
    header.extend(struct.pack("<I", len(feature_index_entries)))
    header.extend(struct.pack("<I", len(part_bboxes)))

    # Index Section
    index_section = bytearray()

    # FeatureIndex table
    for entry in feature_index_entries:
        index_section.extend(struct.pack("<4I", *entry))

    # PartBBoxTable
    for bbox in part_bboxes:
        index_section.extend(struct.pack("<4f", *bbox))

    # GeomIndex table
    if version == 1:
        for entry in geom_index_entries:
            index_section.extend(struct.pack("<2I", *entry))
    else:
        for entry in geom_index_entries:
            index_section.extend(struct.pack("<4I", *entry))

        # RingIndex table (v2)
        for count in ring_index_entries:
            index_section.extend(struct.pack("<I", count))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header + index_section + geometry_data)

    if len(feature_meta_by_index) != len(feature_index_entries):
        raise ValueError("feature meta count does not match FFSF feature count")

    meta_output_path = output_path.parent / f"{country}_feature_meta_by_index.json"
    meta_output_path.write_text(
        json.dumps(feature_meta_by_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "feature_count": len(feature_index_entries),
        "total_part_count": len(part_bboxes),
        "total_ring_count": len(ring_index_entries) if version >= 2 else 0,
        "geometry_bytes": len(geometry_data),
        "feature_ids": feature_ids,
    }
