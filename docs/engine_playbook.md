# Cadis Country Dataset Engine Creation Playbook

This document describes the practical procedure for adding support for a new country dataset to Cadis.

Repository references:

- Cadis runtime: [github.com/isemptyc/cadis](https://github.com/isemptyc/cadis)
- Country dataset build/playbook home: [github.com/isemptyc/cadis-dataset-engine](https://github.com/isemptyc/cadis-dataset-engine)

Recommended Python stack:

- hierarchy probes, geometry builds, and validation commands require a working geospatial Python environment
- at minimum, install the modules used directly or transitively by the pipeline:
  - `pyosmium`
  - `pandas`
  - `numpy`
  - `geopandas`
  - `shapely`
  - `pyproj`
- in practice, make sure `pyproj` can resolve its PROJ data files correctly before running build commands

The key architectural point is that Cadis is dataset-driven.

This playbook is the only supported country onboarding procedure.

No per-country runtime code is allowed.

All country-specific behavior must be encoded in dataset artifacts.

That means country onboarding consists of producing a country dataset that satisfies Cadis runtime contracts, staging it in the expected cache layout, validating runtime behavior, and then enabling the new ISO2 in the public control layer.

## Mental Model

Cadis lookup works in two stages:

1. world resolution determines whether the point belongs to a country and which ISO2 should handle it
2. country runtime execution loads a bootstrapped dataset and interprets geometry, hierarchy, repair, and semantic overlays

That means country onboarding is primarily:

1. define the country's administrative model
2. compile runtime artifacts
3. stage the dataset
4. validate lookup behavior
5. expose the country in the public surface
6. release

In practice, all real country onboarding starts from source data evidence:

- OpenStreetMap extract(s) for the target country or region
- Natural Earth country boundary data when explicit country scoping is needed

Before running probe or build steps, confirm the Python environment includes the required geospatial modules. A typical setup looks like:

```bash
python -m pip install pyosmium pandas numpy geopandas shapely pyproj
```

If those inputs are not available yet, engine creation has not actually started.

## Architectural Rule

Cadis runtime must remain country-agnostic.

Unsupported approach:

- bespoke runtime logic for an individual country
- country-specific branching in lookup execution
- special-case repair logic implemented in Python for one country
- country-specific finalization behavior in runtime code

Supported approach:

- encode country-specific behavior in dataset artifacts only
- keep runtime interpretation generic across countries
- use policy, hierarchy, repair, geometry, and overlay artifacts to express country differences

Country-specific variation must be represented through artifacts such as:

- `runtime_policy.json`
- `geometry.ffsf`
- `geometry_meta.json`
- `hierarchy.json`
- `repair.json`
- optional deterministic overlay files
- dataset manifest metadata

## Final Deliverable

A country integration is operational when it produces a dataset directory like this:

```text
<cache_root>/
  GB/
    gb.admin/
      v1.0.0/
        dataset_release_manifest.json
        geometry.ffsf
        geometry_meta.json
        runtime_policy.json
        hierarchy.json
        repair.json
        overlays/
          names.json
```

Only some files are always required. The effective contract is:

- always required:
  - `dataset_release_manifest.json`
  - `geometry.ffsf`
  - `geometry_meta.json`
  - `runtime_policy.json`
- conditionally required:
  - `hierarchy.json` when `layers.hierarchy_required = true`
  - `repair.json` when `layers.repair_required = true`
  - any optional overlay files declared in `runtime_policy.json`

Cadis enforces this in `cadis/runtime/execution/pipeline.py` and `cadis/runtime/dataset/loader.py`.

## Structural Invariants

All country datasets must preserve these invariants in the finalized runtime hierarchy:

- monotonic level ordering
- stable `osm_id` ordering
- no duplicate levels in the final hierarchy
- no branching parents in the final result
- no post-finalization mutation

These are hard constraints, not stylistic guidance.

They govern:

- which observed OSM structures may be admitted into the runtime model
- which shapes may appear in `allowed_shapes`
- whether hierarchy reconstruction is valid
- whether explicit repair rules are valid
- what semantic overlays are allowed to do

The probe artifacts show what exists in source data. They do not automatically justify exposing those structures in Cadis. Every observed structure must be filtered through the final-result invariants first.

## Promotion Readiness Gates

A country dataset is promotion-ready only if all of the following are true:

- mass lookup sample passes 100%
- repeated runs produce identical output
- no nondeterministic repair anchors exist

These are release gates, not best-effort checks.

Operationally, this means:

- broad lookup sampling must show zero tolerated failures for the promotion set
- the same lookup corpus must yield identical output across repeated runs
- repair behavior must be deterministic, inspectable, and stable under repeated execution

If any of these conditions fail, the dataset may still be useful for development analysis, but it is not ready for promotion.

## Creation Procedure

### 1. Probe the Country's Administrative Hierarchy First

The first real step is not writing `runtime_policy.json`. It is generating the evidence needed to design it.

A practical probe step looks like this:

```python
NE_ROOT = Path("/path/to/nature_earth_dataset")
COUNTRY_DBF = NE_ROOT / "countries" / "ne_10m_admin_0_countries.dbf"
def build_admin_tree(
    country_pbf_path: Path,
    output_dir: Path,
    *,
    country_iso: str,
    country_name: str,
    apply_country_filter: bool = False,
) -> Path:
    """
    Step 0 data probe helper.
    Produces:
    - admin_nodes.json
    - admin_edges.json
    - admin_report.json
    - admin_tree.txt
    """
    from dataset import build_country_from_ne, extract_admin_hierarchy, render_admin_tree

    output_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_dir / "admin_tree.txt"
    country_path = None

    if apply_country_filter:
        country_path = output_dir / f"{country_iso.upper()}_COUNTRY.json"

        if not COUNTRY_DBF.exists():
            raise FileNotFoundError(f"Country DBF not found: {COUNTRY_DBF}")

        if not country_path.exists():
            build_country_from_ne(
                dbf_path=COUNTRY_DBF,
                output_path=country_path,
                country_iso=country_iso.upper(),
                country_name=country_name,
            )

    extract_admin_hierarchy(
        pbf_path=str(country_pbf_path),
        output_dir=output_dir,
        name_keys=("name", "name:en"),
        country_geometry_path=country_path,
    )
    render_admin_tree(
        nodes_path=output_dir / "admin_nodes.json",
        edges_path=output_dir / "admin_edges.json",
        output_path=target_file,
    )
    return target_file
```

This step is an OSM administrative hierarchy scan. It does not build the final runtime dataset yet. It extracts and renders the country's administrative graph so you can decide:

- which admin levels actually exist in the source data
- which of those levels are useful for Cadis
- which parent-child relationships are stable enough to rely on
- which level combinations should be considered complete or partial
- whether a hierarchy repair layer is needed

The important probe artifacts are:

- `admin_nodes.json`
  - discovered administrative entities and their level/name/identifier data
- `admin_edges.json`
  - inferred parent-child relationships between the discovered entities
- `admin_report.json`
  - summary diagnostics such as level distribution, unresolved edges, and graph quality signals
- `admin_tree.txt`
  - human-readable hierarchy rendering used to decide policy

In practice, `admin_tree.txt` and `admin_report.json` are the two most useful review artifacts for determining levels and shapes.

This means a true engine build starts from data, not from a prewritten policy file.
If a `runtime_policy.json` already exists, that should be treated as a downstream contract artifact produced from an earlier probe/design step, not as the real beginning of country onboarding.

### 2. Understand What `apply_country_filter` Is For

`apply_country_filter` matters when the source PBF is regional rather than country-specific.

Example:

- `malaysia-singapore-brunei-latest.osm.pbf`

In that case, the probe step should first build a target-country boundary geometry and use it to constrain extraction to the intended country. Without that filter, the hierarchy scan can be polluted by neighboring countries in the same PBF.

Operationally, this means:

- build or load a country boundary geometry for the target ISO2
- pass that boundary into the hierarchy extraction step
- only then analyze nodes, edges, and level distribution

When the source PBF is already country-specific, `apply_country_filter=False` is usually fine.
When using combined extracts or when relation hierarchy contamination is a risk, Natural Earth country geometry (or an equivalent country boundary source) should be treated as part of the standard probe toolkit rather than an optional afterthought.

Important qualification:

- a single-country extract does not guarantee a clean administrative relation graph
- border-adjacent or foreign features can still leak into probes or builds even when the file name suggests a country-only scope
- if that happens, treat the country-specific extract as a strong hint, not absolute proof that no country scoping work is needed

#### When Natural Earth Is Needed

Natural Earth is needed when the engine requires explicit country-scope truth that cannot be trusted to emerge from the OSM input alone.

Typical cases:

- the input PBF contains multiple countries or a cross-border region
- the relation hierarchy probe is picking up neighboring-country admin structures
- the runtime integration expects `geometry_meta.json` to contain country-scope metadata such as `country_scope_flag`
- boundary-sensitive behavior such as nearby/offshore handling depends on a stable country outline external to the admin polygons themselves
- the country itself has a non-trivial scope question and the engine must decide what the dataset's country boundary means before any scoped metadata is exported

Concrete example:

- target country: Singapore
- source extract: `malaysia-singapore-brunei-latest.osm.pbf`
- why Natural Earth is needed:
  - without an external Singapore boundary, the probe and build steps can admit Malaysia or Brunei administrative relations into the observed graph
  - level distribution, allowed-shape decisions, and any scope-sensitive metadata would then be derived from a polluted multi-country input rather than Singapore alone

Another example:

- target country: France
- design question:
  - should `FR` mean metropolitan France only, or should it include overseas departments / collectivities and other sovereign territories
- why Natural Earth may be needed:
  - once the engine decides the intended lookup scope, it needs an external boundary source that matches that product decision closely enough for scoped probe/build/export behavior
  - the hard part is not only geometry processing; it is deciding what the France dataset is supposed to include before country-scope metadata is emitted

Typical non-cases:

- the input is already a single-country Geofabrik-style extract
- the engine can be modeled and validated cleanly without probe-time country gating
- runtime does not consume exported country-scope metadata for that country

Important caution:

- Natural Earth is a boundary reference, not a guarantee of semantic correctness for every administrative feature
- before enabling export-time country-scope metadata, verify that the chosen boundary source does not wrongly exclude valid features such as islands, enclaves, lagoon municipalities, or coastal admin units
- if the boundary source causes false exclusions, keep the engine unscoped until a better boundary source or country-specific scoping rule is available

Practical fallback when scoped export is not trustworthy:

- keep the engine unscoped at export time
- generate hierarchy artifacts from the finalized dataset rather than from the polluted raw relation graph
- explicitly prune known foreign leakage during engine build if the contamination is sparse and well understood

This is preferable to emitting `country_scope_flag` from an inaccurate boundary source and silently excluding legitimate in-country features.

### 3. Derive Runtime Policy from the Probe Artifacts

`runtime_policy.json` should be derived from what the probe reveals, not written from intuition alone.

A practical derivation workflow is:

1. read `admin_report.json` to see the level distribution and unresolved graph signals
2. inspect `admin_tree.txt` to understand the real parent-child structure
3. choose the levels Cadis should expose
4. decide which level combinations represent structurally acceptable outcomes
5. assign `ok` vs `partial` semantics to those combinations
6. decide whether missing parent levels can be repaired from hierarchy alone
7. decide whether explicit semantic repair anchors are necessary

Before any of those choices are finalized, project the observed country graph into the runtime's final hierarchy model and reject anything that cannot satisfy the structural invariants.

#### Choosing `allowed_levels`

`allowed_levels` should reflect the levels that are both:

- materially present in the source data
- meaningful for the country's runtime output
- capable of participating in an invariant-preserving final hierarchy

Do not include levels only because they exist occasionally in OSM. Include levels that represent stable output structure.

Use:

- `admin_report.json` for frequency and coverage
- `admin_tree.txt` for whether a level behaves consistently in the hierarchy

#### Choosing `allowed_shapes`

`allowed_shapes` should represent the level combinations that Cadis may legitimately observe after geometry lookup and deterministic reconstruction.

Examples of legitimate shapes:

- the full expected path for a well-structured area
- a reduced but still valid path for areas where some levels do not exist
- a sparse path that is known to occur in the country's real admin model

Do not define shapes based only on theoretical hierarchy depth or raw OSM observations. A shape is only valid if it can be normalized into a final hierarchy that preserves:

- monotonic level ordering
- stable `osm_id` ordering
- no duplicate levels
- no branching parentage

Observed source structures that violate those constraints should be normalized earlier or excluded from `allowed_shapes`.

#### Choosing `shape_status`

`shape_status` maps an observed shape to:

- `ok` when the shape is considered complete enough for the country's intended output
- `partial` when the shape is structurally valid but incomplete
- `failed` is not explicitly listed in the file for valid shapes; shapes outside `allowed_shapes` fail at runtime

This is a product decision grounded in data.

A common pattern is:

- full expected structures are `ok`
- sparse but real structures are `partial`
- impossible or untrusted structures are excluded from `allowed_shapes`

A shape should never be marked `ok` or `partial` if producing it would require violating final hierarchy invariants.

#### Choosing Hierarchy Repair Rules

Use hierarchy repair when a missing parent can be recovered deterministically from child-level evidence already present in the dataset.

That usually means:

- a child level has a stable parent in `admin_tree.txt`
- the child-to-parent mapping is consistent enough to serialize into `hierarchy.json`
- the recovered parent does not introduce duplicate levels or branching parentage
- the repaired hierarchy still has stable level and `osm_id` ordering

If this is true, set:

- `layers.hierarchy_required = true`
- `hierarchy_repair_rules.parent_level`
- `hierarchy_repair_rules.child_levels`

#### Choosing Semantic Repair Rules

Use explicit repair rules only when hierarchy-based reconstruction is not enough.

This is appropriate when:

- source data has recurring gaps or ambiguity
- certain child entities need explicit anchoring
- the repair is deterministic and inspectable
- the repaired result still satisfies all final hierarchy invariants

If the country can be modeled cleanly without this layer, keep:

- `layers.repair_required = false`

That is what the UK build appears to have done.

### 4. Define the Administrative Contract

Before generating files, decide what the country should return.

Questions to settle first:

- which admin levels are valid in this country
- which combinations of levels count as `ok`, `partial`, or invalid
- whether missing parent levels can be reconstructed deterministically from hierarchy
- whether known gaps require an explicit semantic repair layer
- whether near-border fallback and offshore classification should be enabled

In Cadis, these decisions are encoded in dataset policy, not handwritten per-country runtime code.

### 5. Choose Canonical Source Data

Country onboarding is primarily a taxonomy alignment problem, not a geometry indexing problem.

Pick the geographic and naming sources that will define:

- polygon boundaries
- stable identifiers
- canonical display names
- multilingual query aliases when the dataset exports them
- parent-child administrative relationships
- any explicit exception rules

The main difficulty is deciding how the country's real administrative system should be normalized into a stable Cadis hierarchy. That means:

- deciding which levels are semantically meaningful
- deciding which entities belong at those levels
- normalizing parent-child relationships
- resolving ambiguous or inconsistent source naming
- deciding which exceptions can be admitted without violating structural invariants

Geometry indexing is downstream of those choices. If taxonomy alignment is weak, a high-quality spatial index does not rescue the dataset.

So when choosing source data, prioritize sources that support semantic consistency and hierarchy normalization first. Geometry richness matters, but it is not the primary modeling problem.

### 5.1 Canonical Naming vs Multilingual Aliases

Cadis datasets may export multilingual naming metadata, but runtime remains language-agnostic.

When a dataset uses the multilingual naming contract:

- `name` is the canonical deterministic label
- `names` is auxiliary alias metadata for query recall
- runtime must pass `names` through unchanged
- runtime must not select language, reorder aliases, or depend on locale

Country engines must define canonical naming and alias extraction separately.

Canonical naming policy:

- is required for every engine
- must be deterministic
- should prefer the dataset's intended local/default administrative label rather than a global language preference

Alias extraction policy:

- is optional
- must be explicitly enabled by the engine
- must use a country-specific narrower alias set rather than exporting every available `name:<lang>` tag

Examples:

- Belgium:
  - canonical policy: `("name", "name:nl", "name:fr", "name:de", "name:en", "official_name")`
  - alias set: `nl`, `fr`, `de`

This narrower alias set is a hard design rule, not a convenience preference.
The goal is to keep alias metadata useful for recall without turning the dataset into an unconstrained dump of multilingual OSM tags.

### 6. Normalize the Source into Cadis Runtime Artifacts

Transform the raw source into the files Cadis can load.

Artifact responsibilities:

- `geometry.ffsf` and `geometry_meta.json`
  - spatial lookup index and feature metadata
  - loaded by `load_geometry_index()`
- `hierarchy.json`
  - name-based parent reconstruction
  - loaded by `load_hierarchy_parent_map()`
- `repair.json`
  - explicit fallback mapping for recurring semantic gaps
  - loaded by `load_repair_anchor_map()`
- `runtime_policy.json`
  - structural rules, layer requirements, allowed shapes, lookup status semantics, nearby/offshore behavior
- `dataset_release_manifest.json`
  - dataset identity, country metadata, version metadata

If the data cannot be represented cleanly in these artifacts, the country model is not ready for Cadis yet.

Important note:

- `hierarchy.json` identifiers do not need to be in the same namespace as geometry feature identifiers
- a mismatch such as `r12345` in hierarchy vs `gb_r12345` in geometry metadata is not, by itself, a runtime defect
- treat identifier mismatch as a problem only if the actual runtime contract for that layer requires direct cross-reference and the dataset fails that contract

### 7. Build `runtime_policy.json` Carefully

This file is the main country-specific runtime contract. From `cadis/runtime/dataset/loader.py`, it needs to define:

- `runtime_policy_version`
- `allowed_levels`
- `allowed_shapes`
- `shape_status`
- `layers.hierarchy_required`
- `layers.repair_required`
- `hierarchy_repair_rules.parent_level`
- `hierarchy_repair_rules.child_levels`
- `repair_rules.parent_level`
- `repair_rules.child_levels`
- required `nearby_policy`
- optional `optional_layers`

What these mean operationally:

- `allowed_levels` defines the levels that exist in the country model
- `allowed_shapes` defines which combinations of discovered levels are structurally legal
- `shape_status` maps those combinations to `ok`, `partial`, or `failed`
- hierarchy rules define when a missing parent can be synthesized from a child node
- repair rules define explicit exception handling when hierarchy alone is insufficient
- nearby policy defines how Cadis behaves near the country boundary and in offshore cases

Weak policy design usually causes more runtime problems than geometry indexing.

`nearby_policy` is not optional in practice. All engines should emit it.
The default policy is:

```json
"nearby_policy": {
  "enabled": true,
  "max_distance_km": 2.0,
  "offshore_max_distance_km": 20.0
}
```

Country engines should treat this as the standard baseline contract.
If a country ever needs different nearby behavior, that should be an explicit,
justified deviation from the default rather than omission of the field.

### 8. Generate the Geometry Index

Compile the country's polygons into:

- `geometry.ffsf`
- `geometry_meta.json`

Cadis runtime expects files compatible with `FFSFSpatialIndexV3`. Every feature should map cleanly to:

- a known admin level
- a stable identifier
- a canonical name

If nearby/offshore behavior matters, include accurate country-scope geometry. Boundary quality directly affects fallback behavior.

If the dataset uses multilingual naming:

- `geometry_meta.json` should carry canonical `name`
- optional `names` should contain only the engine's explicitly allowed alias set
- alias export should follow the dataset's declared `name_schema`

If runtime needs country-scope metadata in `geometry_meta.json`, compute that scope truth at export time from the original full-precision geometry using the external country boundary source. Do not derive country-scope inclusion later from reconstructed or quantized FFSF geometry.

### 9. Add Hierarchy and Repair Layers Only When They Add Deterministic Value

Use `hierarchy.json` when the parent can be derived deterministically from child-level evidence.

Use `repair.json` when the source data has recurring holes or ambiguities that need explicit anchoring.

Do not use semantic overlays to compensate for missing structural logic. Hierarchy and repair layers are the structural tools; overlays are not.

Additional practical rule:

- if the raw hierarchy probe is polluted by cross-country relations, it is valid to derive `hierarchy.json` from the finalized dataset artifact set instead of the raw probe output
- the goal is deterministic parent reconstruction for the released dataset, not loyalty to a contaminated intermediate graph

### 10. Use Semantic Overlays Only for Deterministic Post-Processing

Optional overlay files are declared in `runtime_policy.json` and loaded by `cadis/runtime/dataset/loader.py`.

They may:

- attach extra result metadata
- override display names by `osm_id`

They may not:

- change `lookup_status`
- change hierarchy length
- reorder nodes
- change structural levels
- change the hierarchy `osm_id` sequence

This is the runtime expression of the same structural invariant set. Overlays must preserve the finalized hierarchy, not reinterpret it.

If you need structural correction, fix geometry, hierarchy, repair, or policy instead.

### 11. Write a Correct `dataset_release_manifest.json`

The dataset manifest should identify the dataset cleanly enough for runtime labeling and release management.

At minimum it should correctly represent:

- `country_iso`
- `country_name`
- `dataset_id`
- `dataset_version`

If the dataset exports multilingual naming metadata, the manifest should also declare:

- `name_schema`

Current schema:

- `multilingual_v1`
  - `name` is canonical
  - `names` is optional alias metadata
  - alias keys are representative metadata only and must not be treated as authoritative language guarantees

If the dataset will be distributed through Cadis bootstrap/CDN flow, the manifest and checksums must also satisfy `cadis/cdn/bootstrap.py`.

### 12. Stage the Dataset in the Real Cache Layout

Before changing product metadata, prove the dataset works by staging it exactly as Cadis expects it on disk.

Example:

```text
/tmp/_release_stage/
  GB/
    gb.admin/
      v1.0.0/
        ...
```

Then run lookups against that staged root:

```bash
CADIS_CACHE_DIR="/tmp/_release_stage" cadis lookup 51.38088980891408 -0.06495895248855296 --json
```

This is the fastest high-signal validation loop because it exercises the real runtime without requiring release infrastructure.

Important limitation:

- staging a dataset under `CADIS_CACHE_DIR` does not bypass product-level country support gating
- Cadis currently also checks a hardcoded supported ISO list in `cadis/_api.py`
- if the installed Cadis build does not include the target ISO2 in `SUPPORTED_ISO2`, lookup will still fail even when the staged dataset is valid

So this staged-cache command is a validation step for a Cadis build that already includes the target country support. It is not a way to make an older PyPI release support an unreleased country dataset.

### 13. Enable the Country in the Public Control Layer

Once the staged dataset behaves correctly, expose the ISO2 in the public API surface.

For example, GB support was enabled by updating the supported ISO list in `cadis/_api.py`.

This should remain a small change. If onboarding a country requires broad runtime code changes, the architecture is probably drifting away from dataset-driven design.

### 14. Validate Behavior, Not Just One Successful Point

A country is not validated by a single happy-path lookup.

Minimum validation set:

- a clear onshore success case
- a boundary-adjacent point
- an offshore-but-near point when offshore handling is enabled
- a case that exercises hierarchy reconstruction
- a case that exercises repair anchors, if `repair.json` exists
- invalid input behavior
- missing-dataset behavior
- blocked-by-policy behavior when `CADIS_ALLOWED_ISO2` is in use

Validate both:

- correctness of `result.admin_hierarchy`
- correctness of `execution.lookup_status` and `resolution_state`

This development validation step is necessary but not sufficient for release. Promotion requires the stronger dataset-wide gates defined in "Promotion Readiness Gates."

Operational note:

- evaluation failures caused by insufficient random sampling budget are not automatically dataset failures
- if the harness cannot obtain the intended inside/outside corpus, raise the sampler attempt cap and rerun before concluding that runtime behavior is defective

## What Usually Goes Wrong

In practice, country integrations usually fail for these reasons:

- the level taxonomy is underspecified
- parent-child naming is inconsistent
- canonical identifiers are unstable
- geometry does not align with the semantic hierarchy
- repair rules are incomplete
- overlays are misused to patch structural problems

The Cadis-side code changes are usually the easy part. Data modeling discipline is the hard part.

## Promotion Checklist

Before promoting a country dataset, confirm all of the following:

- finalized hierarchy satisfies all structural invariants
- `allowed_levels`, `allowed_shapes`, and repair rules were derived from probe evidence rather than assumption
- mass lookup sample passes 100%
- the evaluation harness can reliably produce the intended sample corpus without starving on inside-point generation
- repeated runs of the same lookup corpus produce identical output
- no nondeterministic repair anchors exist
- staged dataset behaves correctly from the real cache layout
- public ISO enablement is in place
- package version, artifact build, tag, and release notes are prepared

## UK Example Framing

For the United Kingdom support release, the effective procedure looked like this:

0. start from source data:
   - OSM extract for the target country build
   - Natural Earth country boundary data when explicit country scoping is needed
1. probe the country hierarchy first and review the evidence:
   - `admin_tree.txt`
   - `admin_report.json`
   - level distribution
   - parent-child quality
2. derive the intended runtime policy contract from that evidence
3. implement an initial GB engine against that contract
4. build a staged `GB/gb.admin/<version>/` dataset
5. inspect the generated runtime artifacts directly:
   - `geometry_meta.json`
   - `hierarchy.json`
   - `runtime_policy.json`
6. run real Cadis lookup and evaluation against the staged cache root
7. identify structural problems in the generated artifacts, especially hierarchy scope, parent quality, and country-boundary contamination
8. revise hierarchy generation and rebuild until:
   - runtime behavior is correct
   - artifact scope is coherent
   - the finalized hierarchy is structurally acceptable
9. verify broader evaluation metrics and repeated rebuild determinism

That is the expected pattern for future country onboarding unless Cadis architecture changes materially.
