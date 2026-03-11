# ItalyAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`ItalyAdminEngine` in this repository:

`engines/it/engine_it.py`

`ItalyAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Italy administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `it_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `IT`
- `COUNTRY_NAME`: `Italy`

## 3. Scope and Worldview

The engine models the national Italy administrative spine observed in the OSM
probe:

- Level `4`: Region
- Level `6`: Province / metropolitan-equivalent unit
- Level `8`: Municipality

Configured levels:

- `LEVELS = [4, 6, 8]`

Allowed level shapes:

- `[4]`, `[4,6]`, `[4,6,8]`, `[4,8]`, `[6]`, `[6,8]`, `[8]`

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/italy/`)

Natural Earth (`dbf`) is not required in the current implementation because the
expected source is a single-country Italy extract.

## 5. Build Pipeline

`ItalyAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`italy_admin.json`) via `build_admin_dataset`
2. Extract relation hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`italy_admin.bin` + `IT_feature_meta_by_index.json`)
5. Export semantic layer (`italy_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 6, 8]`
- `allowed_shapes`: all configured IT shapes
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
