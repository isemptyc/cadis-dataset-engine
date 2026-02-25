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

## Reproducibility Model

To reproduce a dataset release:

1. Obtain the corresponding OpenStreetMap extract.
2. Clone this repository.
3. Checkout the engine commit referenced in the dataset release manifest.
4. Build the Docker image.
5. Run the build command with the specified country and dataset version.
6. Verify output checksums against the published release manifest.

Example:

```bash
git clone https://github.com/<org>/cadis-dataset-engine.git
cd cadis-dataset-engine
git checkout <engine_commit>

docker build -t cadis-engine .
docker run --rm \
  -v /path/to/osm:/data/osm:ro \
  -v /path/to/output:/data/out \
  cadis-engine \
  --country tw \
  --version 1.0.0 \
  --osm /data/osm/taiwan.osm.pbf \
  --output /data/out
```

The resulting release files must match the published manifest checksums in `cadis-dataset`.

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
├── package.py
├── publish.py
├── base.py
├── core.py
├── dataset.py
├── engines/
│   └── <iso2>/
│       └── engine_<iso2>.py
└── ffsf/
    ├── ffsf_exporter.py
    └── semantic_dataset_exporter.py
```

---

## Licensing

Code in this repository is licensed under the MIT License.

Datasets generated using this repository may be subject to Open Database License (ODbL) obligations if derived from OpenStreetMap data.

---

## Data Source Notice

Build workflows require externally obtained OpenStreetMap extracts.

This repository does not host, distribute, or guarantee long-term availability of historical raw extracts.

OpenStreetMap data © OpenStreetMap contributors
Licensed under ODbL 1.0
