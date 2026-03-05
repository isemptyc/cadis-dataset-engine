# Frontend-Friendly Spatial Format (FFSF) Specification
# Status: v3 (authoritative runtime + nearest)

---

## 1. Purpose and Scope

FFSF is a geometry-only spatial ABI optimized for frontend point-in-polygon
queries. It is designed for lightweight administrative lookup in photo
browsing scenarios.

FFSF answers exactly one question:
> Given a point, which feature indices contain it?

FFSF does NOT encode:
- Administrative semantics (levels, names, hierarchy)
- Address or descriptive location data

FFSF is a pure spatial fact store.

---

## 2. Design Goals

- Zero-parse / memory-mapped friendly binary layout
- Deterministic, stable encoding
- Efficient point queries with bbox filtering
- Correct polygon topology, including holes (v2)

Correctness is prioritized over compression.

---

## 3. Version Summary

### v1 (legacy)
- Encodes outer rings only
- Drops holes
- No ring boundaries in GeometryData

### v2 (legacy)
- Encodes outer + inner rings
- Explicit ring boundaries via RingIndex
- Preserves polygon holes for correct topology

### v3 (current)
- Same binary layout as v2
- Adds nearest-polygon runtime operator
- Intended as authoritative runtime dataset (no JSON geometry required)

Runtime must reject v1 unless explicitly enabled.

### Normative Requirement (v2+)

For administrative datasets, FFSF v2 MUST preserve all polygon rings
(outer and inner). Dropping inner rings (holes) is NOT permitted, as it
breaks topological equivalence with AdminLookup.

---

## 4. Binary Layout (v3)

All integers are little-endian.
StringPool is optional and unused in v2.

### 4.1 Header (16 bytes)

| Offset | Field          | Type   | Notes               |
|--------|----------------|--------|---------------------|
| 0      | Magic          | char[4]| "FFSF"              |
| 4      | Version        | uint32 | 3                   |
| 8      | FeatureCount   | uint32 | Number of features  |
| 12     | TotalPartCount | uint32 | Number of parts     |

### 4.2 Index Section (v3 order)

1. FeatureIndex (FeatureCount entries)
2. PartBBoxTable (TotalPartCount entries)
3. GeomIndexV2 (TotalPartCount entries)
4. RingIndex (TotalRingCount entries)

TotalRingCount is the sum of RingCount across all parts,
in part order.

#### FeatureIndex

Each entry is 4 x uint32:

| Field        | Type   | Notes |
|--------------|--------|-------|
| StringOffset | uint32 | Reserved (0) |
| StringLen    | uint32 | Reserved (0) |
| PartStartIdx | uint32 | First part index for feature |
| PartCount    | uint32 | Number of parts for feature |

#### PartBBoxTable

Each entry is 4 x float32:

| Field | Type    | Notes |
|-------|---------|-------|
| minX  | float32 | Part bbox min lon |
| minY  | float32 | Part bbox min lat |
| maxX  | float32 | Part bbox max lon |
| maxY  | float32 | Part bbox max lat |

#### GeomIndexV2 (v2/v3)

Each entry is 4 x uint32:

| Field        | Type   | Notes |
|--------------|--------|-------|
| ByteOffset   | uint32 | Part geometry start in GeometryData |
| ByteLen      | uint32 | Total bytes for all rings in part |
| RingStartIdx | uint32 | First ring index for this part |
| RingCount    | uint32 | Number of rings in this part |

#### RingIndex (v2/v3)

Each entry is 1 x uint32:

| Field      | Type   | Notes |
|------------|--------|-------|
| PointCount | uint32 | Number of points in the ring |

Ring order is outer ring first, then inner rings in source order.

### 4.3 Data Section

#### GeometryData

Flat uint16 stream:

```
[x1, y1, x2, y2, ...]
```

Rings are concatenated in order:
feature -> part -> ring.
Ring boundaries are defined by RingIndex point counts.

---

## 5. Coordinate Quantization (v1/v2/v3)

Quantization is local per polygon part using its bbox:

```
Xq = round_half_up((X - minX) / (maxX - minX) * 65535)
Yq = round_half_up((Y - minY) / (maxY - minY) * 65535)
```

Rules:
- Half-up rounding is required (no banker's rounding)
- If span is zero, quantized value is 0
- Values are clamped to [0, 65535]

All rings in a part use the same part bbox.

---

## 6. Geometry Rules

- Polygon and MultiPolygon are supported
- Each part is one polygon
- Rings may include a closing coordinate (first == last)
- The exporter preserves rings as-is
- Runtime must handle closing points correctly

Ring orientation (clockwise / counter-clockwise) is not semantically
significant; runtime relies on ring role (outer vs inner), not winding.

---

## 7. Runtime Semantics (v3)

For each part:
1. Early reject by PartBBoxTable
2. Quantize query point into part-local uint16 space
3. Test outer ring with even-odd ray casting
4. If NOT inside outer ring, the part does NOT match
5. If inside outer ring, test each inner ring
6. If inside any inner ring, the part does NOT match

Feature hit logic:
- A feature matches if any of its parts match
- Runtime returns feature indices (caller maps to feature_id)

Nearest-operator logic (v3):
- If no containment hits, runtime MAY perform nearest lookup within a declared threshold
- Candidate parts are filtered by expanded bbox search (threshold in degrees)
- Distance is computed to polygon boundaries using stored rings
- Nearest polygon per level is returned when within threshold

---

## 8. Exporter Rules (AdminLookup -> FFSF)

The exporter is a pure re-encoding compiler:

Must:
- Preserve feature_id exactly as provided by AdminLookup
- Preserve geometry topology (including holes in v2)
- Use deterministic feature order:
  - admin level ascending
  - original list order within each level

Must not:
- Modify geometry or hierarchy
- Infer semantics
- Query external services

Output:
- One .ffsf.bin per AdminLookup dataset
- v3 by default, v1 optional for testing

---

## 9. Backward Compatibility

- Version field distinguishes v1/v2/v3
- v3 runtime rejects v1/v2 unless explicitly enabled
- v1/v2 behavior is unchanged when legacy handling is enabled

---

## 10. Known Limitations

- No spatial indexing beyond bbox filtering
- No winding rule enforcement (assumes valid input)
- StringPool unused in v2/v3
- No normalization or validation of ring topology (assumes exporter input is valid)

---

## 11. Validation Criteria

For any GPS coordinate:

FFSF v3 containment hits == AdminLookup polygon_evidence hits
(excluding any supplemental or heuristic evidence)

FFSF v3 nearest hits == AdminLookup nearby evidence hits
within the configured distance threshold (excluding world-layer terminal
classification such as open_sea)

Example requirement:
- Taipei 101 hits Taipei City
- Taipei 101 does NOT hit New Taipei City
