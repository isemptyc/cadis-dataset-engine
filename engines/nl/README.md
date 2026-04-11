# NetherlandsAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current build-only behavior of
`NetherlandsAdminEngine` in this repository:

`engines/nl/engine_nl.py`

`NetherlandsAdminEngine` prepares runtime artifacts for Netherlands
administrative lookup datasets.

## 2. Engine Identity

- `ENGINE`: `nl_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `NL`
- `COUNTRY_NAME`: `Netherlands`

## 3. Scope and Worldview

The engine models the stable Netherlands administrative spine observed in the
filtered dataset build:

- Level `4`: Province
- Level `8`: Municipality

Configured levels:

- `LEVELS = [4, 8]`

Allowed level shapes:

- `[4]`, `[4,8]`, `[8]`

Probe notes:

- the raw relation scan contains neighboring-country entities and church-style
  administrative relations that are not valid Cadis runtime units
- the engine therefore uses country-scoped dataset extraction and writes the
  runtime hierarchy from the cleaned dataset spine
- one Belgian enclave municipality (`Baarle-Hertog`) is explicitly excluded
  after build because it survives the general country-scope filter

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional inputs:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/netherlands/`)
- `country_geometry_path` (recommended for the Netherlands extract)

## 5. Build Pipeline

`NetherlandsAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`netherlands_admin.json`) via `build_admin_dataset`
2. Remove explicitly excluded enclave features
3. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
4. Render hierarchy text (`admin_tree.txt`)
5. Export spatial binary layer (`netherlands_admin.bin` + `NL_feature_meta_by_index.json`)
6. Export semantic layer (`netherlands_admin_semantic.json`)
7. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
8. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 8]`
- `allowed_shapes`: `[4]`, `[4,8]`, `[8]`
- `shape_status`:
  - `ok` for `[4,8]`
  - `partial` for `[4]` and `[8]`
- `layers`:
  - `hierarchy_required = true`
  - `repair_required = false`
- `hierarchy_repair_rules`:
  - `parent_level = 4`
  - `child_levels = [8]`
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
