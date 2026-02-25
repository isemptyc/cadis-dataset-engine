from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

"""
Administrative Semantic Dataset Exporter
=======================================

This module exports a *preprocessed administrative semantic dataset* from
a rendered admin_tree (e.g. admin_tree.txt) into a frontend-friendly,
flat, ID-keyed structure.

Design Intent
-------------

This dataset represents **semantic administrative relationships only**:
- parent / child hierarchy
- administrative level
- canonical display name

It intentionally contains **no geometry** and **no spatial logic**.

The exported result is meant to be consumed by frontend runtime engines
as *semantic evidence*, optionally supplementing polygon-based lookup results
(e.g. FFSF).

Scope & Non-Goals
-----------------

This module:
- Validates hierarchy integrity (missing parents, cycles)
- Produces a deterministic, flat mapping keyed by feature_id
- Preserves semantic facts as-is from preprocessing

This module does NOT:
- Perform spatial reasoning
- Resolve administrative conflicts
- Decide runtime resolution correctness
- Enforce country-specific policies

All semantic *interpretation* and *final authority* belongs to the
country-specific runtime engine on the frontend.

Country-Specific Usage Notes
----------------------------

Different runtime engines may choose to use or ignore this dataset:

- Taiwan:
  Semantic supplementation is REQUIRED due to known polygon gaps
  (e.g. city-level geometries missing in OSM-derived datasets).

- Japan / United Kingdom:
  Semantic supplementation is currently NOT ENABLED.
  Although multiple valid administrative hierarchies may exist for a feature,
  no verified lookup failure currently requires admin_tree-based correction.

The presence of this dataset does NOT imply it must be used.

Activation, traversal rules, and conflict resolution strategies are
explicitly delegated to each AdminEngine implementation.

Design Principle
----------------

This module follows a strict separation of concerns:

    "Data provides facts.
     Engines decide meaning."

Any future change in semantic supplementation policy MUST be implemented
at the runtime engine layer, not here.
"""

def _require_field(node: dict, field: str):
    if field not in node:
        raise ValueError(f"Missing required field: {field}")
    return node[field]


def _build_node_map(nodes: list[dict]) -> dict[str, dict]:
    node_map: dict[str, dict] = {}
    for node in nodes:
        feature_id = _require_field(node, "feature_id")
        if feature_id in node_map:
            raise ValueError(f"Duplicate feature_id: {feature_id}")
        node_map[feature_id] = node
    return node_map


def _validate_parent_links(node_map: dict[str, dict]) -> None:
    for feature_id, node in node_map.items():
        parent_id = node.get("parent_id")
        if parent_id is None:
            continue
        if parent_id not in node_map:
            raise ValueError(
                f"Missing parent_id reference: {feature_id} -> {parent_id}"
            )


def _detect_cycles(node_map: dict[str, dict]) -> None:
    '''
    Cycles are considered invalid input and will cause export to fail.
    '''
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise ValueError(f"Cycle detected at: {node_id}")

        visiting.add(node_id)
        parent_id = node_map[node_id].get("parent_id")
        if parent_id is not None:
            visit(parent_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in node_map:
        visit(node_id)


def export_admin_semantic_dataset(
    nodes: list[dict],
    output_path: str | Path,
    *,
    version: str,
    country: str,
    source: str = "admin_tree.txt",
) -> Path:
    """
    Export an Administrative Semantic Dataset from parsed admin_tree nodes.

    Input expectations:
    - nodes is a list of dicts, each with:
      - feature_id (string)
      - level (number)
      - name (string)
      - parent_id (string or None)

    Output guarantees:
    - Top-level structure matches the Administrative Semantic Dataset v1.0
    - nodes table is a flat dict keyed by feature_id
    - values are fixed-position arrays: [level, name, parent_id]

    Failure conditions:
    - Duplicate feature_id entries
    - Missing required fields
    - parent_id references missing nodes
    - Cycles in parent relationships
    """
    output_path = Path(output_path)

    node_map = _build_node_map(nodes)
    _validate_parent_links(node_map)
    _detect_cycles(node_map)

    # Nodes are serialized in sorted feature_id order to ensure deterministic output
    serialized_nodes: dict[str, list] = {}
    for feature_id in sorted(node_map.keys()):
        node = node_map[feature_id]
        level = _require_field(node, "level")
        name = _require_field(node, "name")
        parent_id = node.get("parent_id")
        serialized_nodes[feature_id] = [level, name, parent_id]

    payload = {
        "version": version,
        "country": country,
        "source": source,
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "nodes": serialized_nodes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        #json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    return output_path
