# FranceAdminEngine (cadis-era) Specification

Version: **v1.0**
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`FranceAdminEngine` in this repository:

`engines/fr/engine_fr.py`

`FranceAdminEngine` prepares runtime artifacts for France administrative
lookup systems.

## 2. Engine Identity

- `ENGINE`: `fr_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `FR`
- `COUNTRY_NAME`: `France`

## 3. Scope and Worldview

The engine models the France administrative spine as:

- Level `4`: Region
- Level `6`: Department
- Level `8`: Municipality / commune-level unit

Configured levels:

- `LEVELS = [4, 6, 8]`

Allowed level shapes:

- `[4]`, `[4,6]`, `[4,6,8]`, `[4,8]`, `[6]`, `[6,8]`, `[8]`

The engine writes hierarchy artifacts from the finalized geometry-backed
dataset spine so release artifacts stay dataset-scoped even when the raw OSM
relation graph includes cross-border translation or relation noise.

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional inputs:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/france/`)
- `country_geometry_path`

## 5. Build Pipeline

`FranceAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`france_admin.json`) via `build_admin_dataset`
2. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
3. Render hierarchy text (`admin_tree.txt`)
4. Export spatial binary layer (`france_admin.bin` + `FR_feature_meta_by_index.json`)
5. Export semantic layer (`france_admin_semantic.json`)
6. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
7. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 6, 8]`
- `allowed_shapes`: all configured France shapes
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
