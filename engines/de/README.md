# GermanyAdminEngine (cadis-era) Specification

Version: **v1.0**
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`GermanyAdminEngine` in this repository:

`engines/de/engine_de.py`

`GermanyAdminEngine` is a country-specific dataset build engine that prepares
runtime artifacts for Germany administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `de_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `DE`
- `COUNTRY_NAME`: `Germany`

## 3. Scope and Worldview

The engine models the stable Germany administrative spine observed in the OSM
probe and geometry build:

- Level `4`: State (`Bundesland`)
- Level `5`: Administrative region (`Regierungsbezirk`) where present
- Level `6`: District / district-free city (`Kreis` / `kreisfreie Stadt`)
- Level `8`: Municipality (`Gemeinde`)

Configured levels:

- `LEVELS = [4, 5, 6, 8]`

Probe notes:

- the raw relation hierarchy contains substantial level `9+` local noise and
  sparse relation parentage
- the Germany Geofabrik extract includes a small number of neighboring border
  administrative relations, so the release flow must build with an explicit
  Natural Earth Germany boundary
- after scoped build plus deterministic exclusions, the current build yields
  `16` level-4 states, `19` level-5 regions, `398` level-6 districts, and
  `10815` level-8 municipalities
- the engine writes hierarchy artifacts from the exported dataset spine so the
  build-stage hierarchy remains Germany-scoped

## 4. Build Inputs

Required input:

- `osm_pbf_path`

Optional inputs:

- `work_dir` (default: `~/.cache/cadis_dataset_engine/germany/`)
- `country_geometry_path` for Germany boundary scoping

The release SOP should generate the Germany boundary with:

`/Users/isempty/Projects/my_cadis/scripts/build_de_boundaries.py`

## 5. Build Pipeline

`GermanyAdminEngine` performs deterministic dataset preparation:

1. Build polygon admin dataset (`germany_admin.json`) via `build_admin_dataset`
2. Remove known neighboring border relations that survive coarse boundary
   representative-point filtering
3. Project a dataset-scoped hierarchy (`admin_nodes.json`, `admin_edges.json`)
4. Render hierarchy text (`admin_tree.txt`)
5. Export spatial binary layer (`germany_admin.bin` + `DE_feature_meta_by_index.json`)
6. Export semantic layer (`germany_admin_semantic.json`)
7. Materialize runtime release layers:
   - `geometry.ffsf`
   - `geometry_meta.json`
   - `hierarchy.json`
8. Emit runtime policy + build manifest

## 6. Runtime Policy Metadata

`runtime_policy.json` includes:

- `allowed_levels`: `[4, 5, 6, 8]`
- `layers`:
  - `hierarchy_required = true`
  - `repair_required = false`
- `hierarchy_repair_rules`:
  - `parent_level = 4`
  - `child_levels = [5, 6, 8]`
- `nearby_policy`:
  - `enabled = true`
  - `max_distance_km = 2.0`
  - `offshore_max_distance_km = 20.0`

## 7. Release Boundary

The engine release boundary is:

- `runtime_policy.json`
- `geometry.ffsf`
- `geometry_meta.json`
- `hierarchy.json`
