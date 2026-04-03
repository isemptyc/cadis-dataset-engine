# DenmarkAdminEngine (cadis-era) Specification

Version: **v1.0**
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`DenmarkAdminEngine` in this repository:

`engines/dk/engine_dk.py`

`DenmarkAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Denmark administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `dk_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `DK`
- `COUNTRY_NAME`: `Denmark`

## 3. Scope and Worldview

The initial Denmark engine models the most likely stable administrative spine
for current OSM Denmark extracts:

- Level `4`: Region
- Level `7`: Municipality (`kommune`)

Configured levels:

- `LEVELS = [4, 7]`

Allowed level shapes:

- `[4]`, `[4,7]`, `[7]`

Probe expectations:

- Denmark should behave similarly to the current Sweden/Norway engines, but the
  actual release policy must be confirmed from the Denmark OSM probe artifacts
- if the probe shows unstable parentage, foreign-border leakage, or missing
  region-to-municipality coverage, this engine must be revised before release
- the engine writes hierarchy artifacts from the exported dataset spine so the
  build-stage hierarchy remains Denmark-scoped and aligned with the geometry
  dataset rather than with raw relation noise

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/denmark/`)

Natural Earth (`dbf`) is not required in the current implementation because the
expected source is a single-country Denmark extract.

## 5. Build Pipeline

`DenmarkAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`denmark_admin.json`) via `build_admin_dataset`
2. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`denmark_admin.bin` + `DK_feature_meta_by_index.json`)
5. Export semantic layer (`denmark_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 7]`
- `allowed_shapes`: `[4]`, `[4,7]`, `[7]`
- `shape_status`:
  - `ok` for `[4,7]`
  - `partial` for `[4]` and `[7]`
- `layers`:
  - `hierarchy_required = true`
  - `repair_required = false`
- `hierarchy_repair_rules`:
  - `parent_level = 4`
  - `child_levels = [7]`
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
