# SouthKoreaAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`SouthKoreaAdminEngine` in this repository:

`engines/kr/engine_kr.py`

`SouthKoreaAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for South Korea administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `kr_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `KR`
- `COUNTRY_NAME`: `South Korea`

## 3. Scope and Worldview

The engine models the national South Korea administrative spine observed in the
OSM probe:

- Level `4`: Province / metropolitan-city-equivalent region
- Level `6`: City / county / district-equivalent intermediate unit
- Level `8`: Town / township / neighborhood-equivalent local unit

Configured levels:

- `LEVELS = [4, 6, 8]`

Allowed level shapes:

- `[4]`, `[4,6]`, `[4,6,8]`, `[4,8]`, `[6]`, `[6,8]`, `[8]`

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional input:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/south_korea/`)

Natural Earth (`dbf`) is not required in the current implementation because the
expected source is a single-country South Korea extract.

## 5. Build Pipeline

`SouthKoreaAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`south_korea_admin.json`) via `build_admin_dataset`
2. Extract relation hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`south_korea_admin.bin` + `KR_feature_meta_by_index.json`)
5. Export semantic layer (`south_korea_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 6, 8]`
- `allowed_shapes`: all configured KR shapes
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
