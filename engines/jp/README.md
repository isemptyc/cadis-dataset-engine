# JapanAdminEngine (cadis-era) Specification

Version: **v2.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`JapanAdminEngine` in this repository:

`engines/jp/engine_jp.py`

`JapanAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Japan administrative lookup systems.

This document is descriptive (implementation-aligned), not normative.

## 2. Engine Identity

- `ENGINE`: `japan_admin`
- `VERSION`: `v2.0`
- `COUNTRY_ISO`: `JP`
- `COUNTRY_NAME`: `Japan`

## 3. Scope and Worldview

The engine handles these meaningful administrative levels:

- Level `3`: Prefectural level (都道府県)
- Level `4`: Municipality level (市・町・村)
- Level `7`: Sub-municipal level (町丁目・地域単位)

Configured levels:

- `LEVELS = [3, 4, 7]`

Allowed level shapes:

- `[3]`, `[4]`, `[7]`, `[3,4]`, `[3,7]`, `[4,7]`, `[3,4,7]`

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/japan/`)

Natural Earth (`dbf`) is **not required** in this cadis-era implementation.

## 5. Admin Profile (`JP_PROFILE`)

Name priority:

- `("name:ja", "name", "name:en")`

Per-level policies:

- Level `3`: simplify (`0.01`), fix invalid geometry, strict parent resolution
- Level `4`: simplify (`0.001`), fix invalid geometry, strict parent resolution
- Level `7`: no simplify, no fix-invalid, strict parent resolution

Parent fallback:

- `parent_fallback = False`

## 6. Build Pipeline

`JapanAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`japan_admin.json`) via `build_admin_dataset`
2. Extract relation hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`japan_admin.bin` + `JP_feature_meta_by_index.json`)
5. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
6. Emit runtime policy + build manifest

## 7. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[3, 4, 7]`
- `allowed_shapes`: all configured JP shapes
- `shape_status`:
  - `ok` for `[3,4]` and `[3,7]`
  - `partial` for other allowed shapes
- `layers`:
  - `hierarchy_required = false`
  - `repair_required = false`
- `nearby_policy`:
  - `enabled = true`
  - `max_distance_km = 2.0`
  - `offshore_max_distance_km = 20.0`

## 8. Release Boundary

The engine release boundary (for manifest packaging) is:

- `runtime_policy.json`
- `geometry.ffsf`
- `geometry_meta.json`

## 9. Non-Goals

This engine does not perform runtime lookup behavior, including:

- point query orchestration
- runtime status evaluation from live polygon hits
- reverse geocoding or label inference
- hierarchy repair or semantic fabrication

Those concerns belong to downstream runtime repositories.
