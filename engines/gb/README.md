# GreatBritainAdminEngine (cadis-era) Specification

Version: **v1.0**  
Status: Draft

## 1. Purpose

This document describes the current **build-only** behavior of
`GreatBritainAdminEngine` in this repository:

`engines/gb/engine_gb.py`

`GreatBritainAdminEngine` is a country-specific dataset build engine that
prepares runtime artifacts for GB / United Kingdom administrative lookup systems.

## 2. Engine Identity

- `ENGINE`: `gb_admin`
- `VERSION`: `v1.0`
- `COUNTRY_ISO`: `GB`
- `COUNTRY_NAME`: `United Kingdom`

## 3. Scope and Worldview

- `LEVELS = [4, 5, 6, 8]`
- Runtime policy includes the supplied UK/GB level combinations plus the
  repo-standard `nearby_policy`

## 4. Build Pipeline

The engine builds:

- `gb_admin.json`
- `admin_tree.txt`
- `gb_admin.bin`
- `GB_feature_meta_by_index.json`
- runtime `geometry.ffsf`
- runtime `geometry_meta.json`
- runtime `hierarchy.json`
- runtime `runtime_policy.json`

## 5. Notes

- Build dispatch accepts both `gb` and `uk`, but output is written under `GB/`
- The current hierarchy export remains a first-pass inferred implementation
