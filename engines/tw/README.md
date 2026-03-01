# TaiwanAdminEngine (cadis-era) Specification

Version: **v2.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`TaiwanAdminEngine` in this repository:

`engines/tw/engine_tw.py`

`TaiwanAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Taiwan administrative lookup systems.

This document is descriptive (implementation-aligned), not normative.

## 2. Engine Identity

- `ENGINE`: `taiwan_admin`
- `VERSION`: `v2.0`
- `COUNTRY_ISO`: `TW`
- `COUNTRY_NAME`: `Taiwan`

## 3. Scope and Worldview

The engine handles these meaningful administrative levels:

- Level `4`: City-level administrative unit
- Level `7`: District-level administrative unit
- Level `8`: Alternative sub-city administrative unit

Configured levels:

- `LEVELS = [4, 7, 8]`

Allowed level shapes:

- `[4]`, `[4,7]`, `[4,8]`, `[4,7,8]`

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/taiwan/`)

Natural Earth (`dbf`) is **not required** in this cadis-era implementation.

## 5. Admin Profile (`TW_PROFILE`)

Name priority:

- `("name:zh-Hant", "name:zh", "name")`

Per-level policies:

- Level `4`: simplify (`0.0005`), fix invalid geometry, strict parent resolution
- Level `7`: no simplify, no fix-invalid, strict parent resolution
- Level `8`: no simplify, no fix-invalid, strict parent resolution

Parent fallback:

- `parent_fallback = False`

## 6. Build Pipeline

`TaiwanAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`taiwan_admin.json`) via `build_admin_dataset`
2. Extract relation hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`taiwan_admin.bin` + `TW_feature_meta_by_index.json`)
5. Export semantic layer (`taiwan_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 7. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 7, 8]`
- `allowed_shapes`: all configured TW shapes
- `shape_status`: `ok` for all allowed shapes
- `layers`:
  - `hierarchy_required = true`
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
- `hierarchy.json`

## 9. Non-Goals

This engine does not perform runtime lookup behavior, including:

- point query orchestration
- runtime status evaluation from live polygon hits
- reverse geocoding or label inference
- hierarchy repair or semantic fabrication

Those concerns belong to downstream runtime repositories.
