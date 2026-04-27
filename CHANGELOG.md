# Changelog

## Unreleased

- Added explicit branch identity metadata for runtime `hierarchy.json` payloads.
- Added `ffsf/runtime_hierarchy.py` to materialize deterministic `root_id`,
  `branch_id`, `path_ids`, and `path_signature` fields from dataset-scoped
  parent chains.
- Updated the Portugal engine to emit branch identity metadata for newly built
  hierarchy artifacts.
- Preserved dataset semantics: geometry, runtime policy, node IDs, node levels,
  names, and parent relationships are unchanged apart from the additional
  branch identity fields.
- Existing published datasets remain valid and do not require regeneration.

## License Change

- Changed the repository source code license from MIT to Apache License 2.0.
- Clarified that `cadis-dataset-engine` contains reproducibility and build code only.
- Clarified that published dataset artifacts are licensed separately in the `cadis-dataset` repository under ODbL as applicable.
- No functional changes were made to engine behavior or dataset generation logic.
