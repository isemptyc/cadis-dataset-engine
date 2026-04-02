# SwedenAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`SwedenAdminEngine` in this repository:

`engines/se/engine_se.py`

`SwedenAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Sweden administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `se_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `SE`
- `COUNTRY_NAME`: `Sweden`

## 3. Scope and Worldview

The engine models the stable Sweden administrative spine observed in the OSM
probe and geometry build:

- Level `4`: County (`län`)
- Level `7`: Municipality (`kommun`)

Configured levels:

- `LEVELS = [4, 7]`

Allowed level shapes:

- `[4]`, `[4,7]`, `[7]`

Probe notes:

- the relation scan contains substantial `8+` noise from parishes, districts,
  neighborhoods, and similar local units
- the geometry-backed build for `[4,7]` yields `21` counties and `290`
  municipalities with full municipality-to-county parent coverage

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/sweden/`)

Natural Earth (`dbf`) is not required in the current implementation because the
expected source is a single-country Sweden extract.

## 5. Build Pipeline

`SwedenAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`sweden_admin.json`) via `build_admin_dataset`
2. Extract relation hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`sweden_admin.bin` + `SE_feature_meta_by_index.json`)
5. Export semantic layer (`sweden_admin_semantic.json`)
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
