# BelgiumAdminEngine (cadis-era) Specification

Version: **v1.0**
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`BelgiumAdminEngine` in this repository:

`engines/be/engine_be.py`

`BelgiumAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Belgium administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `be_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `BE`
- `COUNTRY_NAME`: `Belgium`

## 3. Scope and Worldview

The engine models the stable Belgium administrative spine observed in the
current OSM probe and geometry build:

- Level `4`: Region
- Level `6`: Province-equivalent unit
- Level `8`: Municipality

Configured levels:

- `LEVELS = [4, 6, 8]`

Allowed level shapes:

- `[4]`, `[4,6]`, `[4,6,8]`, `[4,8]`, `[6]`, `[6,8]`, `[8]`

Probe notes:

- the raw Belgium relation graph is noisy and contains neighboring-country
  entities plus dense `9+` local-unit inventory
- the engine therefore writes hierarchy artifacts from the exported dataset
  spine rather than trusting the raw relation tree as the release hierarchy
  source
- Belgium should be built with explicit country-scope geometry so FFSF export
  can emit `country_scope_flag` metadata from exact geometry precision

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/belgium/`)
- `country_geometry_path`

Natural Earth-derived boundary geometry is recommended for Belgium because
border-adjacent leakage has been observed in the OSM extract.

## 5. Build Pipeline

`BelgiumAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`belgium_admin.json`) via `build_admin_dataset`
2. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`belgium_admin.bin` + `BE_feature_meta_by_index.json`)
5. Export semantic layer (`belgium_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 6, 8]`
- `allowed_shapes`: all configured Belgium shapes
- `shape_status`:
  - `ok` for `[4,6]`, `[4,6,8]`, and `[4,8]`
  - `partial` for other allowed shapes
- `layers`:
  - `hierarchy_required = true`
  - `repair_required = false`
- `hierarchy_repair_rules`:
  - `parent_level = 4`
  - `child_levels = [6, 8]`
- `nearby_policy`:
  - `enabled = true`
  - `max_distance_km = 2.0`
  - `offshore_max_distance_km = 20.0`

## 7. Release Boundary

The engine release boundary (for manifest packaging) is:

- `runtime_policy.json`
- `geometry.ffsf`
- `geometry_meta.json`
- `hierarchy.json`
