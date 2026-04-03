# NorwayAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`NorwayAdminEngine` in this repository:

`engines/no/engine_no.py`

`NorwayAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Norway administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `no_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `NO`
- `COUNTRY_NAME`: `Norway`

## 3. Scope and Worldview

The engine currently models the stable Norway administrative spine observed in
the initial probe:

- Level `4`: County (`fylke`)
- Level `7`: Municipality (`kommune`)

Configured levels:

- `LEVELS = [4, 7]`

Allowed level shapes:

- `[4]`, `[4,7]`, `[7]`

Probe notes:

- the raw relation scan includes neighboring-country entities, economic-zone
  entries, and sparse `9`-level borough-style units
- the runtime engine should therefore avoid using the raw relation tree as the
  release hierarchy source
- this engine writes hierarchy artifacts from the exported dataset spine so the
  build-stage hierarchy remains Norway-scoped and aligned with the geometry
  dataset rather than with noisy cross-border relation inventory

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/norway/`)

Natural Earth (`dbf`) is not required in the current implementation because the
expected source is a single-country Norway extract.

## 5. Build Pipeline

`NorwayAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`norway_admin.json`) via `build_admin_dataset`
2. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`norway_admin.bin` + `NO_feature_meta_by_index.json`)
5. Export semantic layer (`norway_admin_semantic.json`)
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
