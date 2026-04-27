# cadis-dataset-engine

Cadis dataset reproducibility repository.

This repository provides the deterministic build environment required to reproduce dataset releases published in the `cadis-dataset` repository.

It exists to ensure transparency, verifiability, and ODbL compliance of dataset generation.

This repository is not a runtime dependency and is not intended to be a pip-installable module.

---

## Purpose

This repository provides:

- Docker-based reproducibility environment
- Country-specific transformation logic
- Administrative hierarchy modeling policies
- Verification scripts for release parity

It does not publish dataset artifacts.

Published dataset releases are available in the `cadis-dataset` repository.

---

## Runtime Artifact Contract

Country engines materialize runtime release layers consumed by `cadis`:

- `geometry.ffsf`
- `geometry_meta.json`
- `hierarchy.json`
- `runtime_policy.json`

`hierarchy.json` is an interpretation aid, not a dataset mutation surface. Its
core semantic fields are:

- `id`
- `level`
- `name`
- `names`
- `parent_id`

For newly built capable datasets, `hierarchy.json` may also include explicit
branch identity metadata:

- top-level `branch_identity_version`
- top-level `branch_identity`
- per-node `root_id`
- per-node `branch_id`
- per-node `path_ids`
- per-node `path_signature`

These fields are derived deterministically from the dataset-scoped parent chain.
They allow the runtime to validate hierarchy repair by graph/path membership
instead of name inference. They do not alter geometry, administrative levels,
names, parent relationships, or feature IDs.

Older published datasets without branch identity metadata remain valid. Runtime
support is capability-driven: Cadis uses explicit branch identity when present
and valid, and otherwise falls back to the legacy guarded repair behavior.

---

## Reproducibility Model

To reproduce a dataset release:

1. Obtain the corresponding OpenStreetMap extract.
2. Clone this repository.
3. Checkout the engine commit referenced in the dataset release manifest.
4. Prepare Docker image (pull published image or build locally).
5. Run the build command with the specified country and dataset version.
6. Verify output checksums against the published release manifest.

Example:

```bash
git clone https://github.com/isemptyc/cadis-dataset-engine.git
cd cadis-dataset-engine

# Checkout the engine commit specified in the dataset release manifest
git checkout <commit_sha>

# Option A: pull published image
docker pull ghcr.io/isemptyc/cadis-dataset-engine:2026-02

# Option B: build locally from checked-out source
docker build -t cadis-engine .

# Use either image:
# - ghcr.io/isemptyc/cadis-dataset-engine:2026-02
# - cadis-engine
docker run --rm \
  -v $(pwd):/app \
  -v /path/to/osm:/data/osm:ro \
  -v /path/to/output:/data/out \
  ghcr.io/isemptyc/cadis-dataset-engine:2026-02 \
  --country tw \
  --version 1.0.0 \
  --osm /data/osm/taiwan.osm.pbf \
  --output /data/out
```

For large country extracts, low Docker memory limits may terminate the build with exit code `137` (OOM/SIGKILL).
Recommended Docker Desktop baseline for stable reproduction:
  - `CPU`: at least 2~4
  - `Memory`: at least 8~16 GB
  - `Swap`: at least 4 GB

The resulting release files must match the published manifest checksums in `cadis-dataset`.

### United States Reproduction

The United States dataset is a special build because the full-country OSM PBF is
too large for the normal single-extract workflow. Reproduce it from the
Geofabrik state/territory extract directory:

```bash
docker run --rm \
  -v $(pwd):/app \
  -v "/path/to/United States of America":/data/osm/us:ro \
  -v /path/to/output:/data/out \
  ghcr.io/isemptyc/cadis-dataset-engine:2026-02 \
  --country us \
  --version 0.2.4 \
  --osm /data/osm/us \
  --output /data/out
```

For US builds, `--osm` is a directory containing Geofabrik files such as
`california-latest.osm.pbf`, `new-york-latest.osm.pbf`, and
`puerto-rico-latest.osm.pbf`, not a single `.osm.pbf` file.

The US engine performs a deterministic stitching stage before polygon assembly:

```text
state/territory PBF extracts
→ relation/way/node union
→ global admin polygon assembly
→ standard Cadis runtime artifacts
```

To avoid repeating the expensive source scan, the engine may create an optional
source-side cache:

```text
<United States of America>/_stitch_cache/<fingerprint>/
├── relations.pkl
├── ways.pkl
└── nodes.pkl
```

This cache is derived entirely from the source PBF extracts and is not required
for public reproducibility. If `_stitch_cache` is missing, the engine regenerates
it from the PBF files. Deleting the cache is safe, but the next US build will be
slow. The cache must not be committed to this repository.

For strict reproducibility verification against pinned OSM identity metadata, add:

```bash
  --verified
```

Checksum verification against a released dataset manifest:

```bash
python verify.py \
  --country tw \
  --version 1.0.0 \
  --output /data/out \
  --release-manifest /path/to/cadis-dataset/releases/TW/engine.tw_admin/v1.0.0/dataset_release_manifest.json
```

---

## Repository Scope

This repository:

- Does not serve as a dataset CDN
- Does not distribute OpenStreetMap extracts
- Does not provide a runtime library
- Does not automatically fetch datasets

It exists solely as a public reproducibility reference.

---

## Relationship to Other Repositories

- `cadis` — runtime interpreter (consumes dataset artifacts)
- `cadis-dataset` — immutable dataset release repository (CDN surface)
- `cadis-dataset-engine` — deterministic rebuild reference

The runtime does not depend on this repository.

---

## Determinism Contract

For a fixed:

- Engine repository commit
- Country configuration version
- OpenStreetMap extract snapshot

The build output must be:

1. Deterministic
2. Checksum-level reproducible
3. Verifiable via release manifest checksums

Dataset releases must reference the exact engine commit used to produce them.

Commits referenced by published dataset releases must remain immutable.
History rewrites affecting referenced commits are prohibited.

---

## Repository Layout

```text
cadis-dataset-engine/
├── Dockerfile
├── __init__.py
├── build.py
├── verify.py
├── base.py
├── dataset.py
├── docs/
│   └── engine_playbook.md
├── engines/
│   └── <iso2>/
│       └── engine_<iso2>.py
└── ffsf/
    ├── ffsf_exporter.py
    ├── runtime_hierarchy.py
    └── semantic_dataset_exporter.py
```

---

## Licensing

Code in this repository is licensed under the Apache License 2.0. See `LICENSE`.

This repository builds derived dataset artifacts from third-party geospatial sources, primarily OpenStreetMap data, and in some engine pipelines may also use Natural Earth data.

OpenStreetMap data must be obtained separately and is licensed under the Open Database License (ODbL). Use of OpenStreetMap-derived data is subject to ODbL, and distribution of derivative databases or public use may trigger attribution and share-alike requirements under that license.

Natural Earth, where used, is a separate third-party input and is not covered by this repository's Apache 2.0 license. It must be obtained and used under its own published terms.

The separately published `cadis-dataset` repository remains a distinct artifact and is licensed under ODbL as applicable to the released dataset contents.

Using this repository to generate dataset artifacts does not by itself relicense those published dataset artifacts under Apache 2.0.

This repository does not provide any dataset for direct consumption. Dataset licensing obligations are defined by the corresponding dataset release repository.

---

## Data Source Notice

Build workflows require externally obtained OpenStreetMap extracts.

This repository does not host, distribute, or guarantee long-term availability of historical raw extracts.

OpenStreetMap data © OpenStreetMap contributors
Licensed under the Open Database License (ODbL) 1.0
https://www.openstreetmap.org/copyright
