from __future__ import annotations

import hashlib
from typing import Any


BRANCH_IDENTITY_VERSION = "1.0"


def _stable_path_signature(path_ids: list[str]) -> str:
    payload = "\x1f".join(path_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_to_root(node_id: str, node_by_id: dict[str, dict[str, Any]]) -> list[str]:
    path: list[str] = []
    seen: set[str] = set()
    current_id: str | None = node_id
    while current_id is not None:
        if current_id in seen:
            raise ValueError(f"Cycle detected at: {current_id}")
        seen.add(current_id)
        node = node_by_id.get(current_id)
        if node is None:
            raise ValueError(f"Missing hierarchy node: {current_id}")
        path.append(current_id)
        parent_id = node.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"Invalid parent_id for node: {current_id}")
        current_id = parent_id
    return path


def build_runtime_hierarchy_payload(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build hierarchy.json with explicit branch identity for new datasets."""
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("Runtime hierarchy node id must be a non-empty string.")
        if node_id in node_by_id:
            raise ValueError(f"Duplicate runtime hierarchy node id: {node_id}")
        node_by_id[node_id] = node

    annotated_nodes: list[dict[str, Any]] = []
    for node in nodes:
        node_id = node["id"]
        path_ids = list(reversed(_path_to_root(node_id, node_by_id)))
        root_id = path_ids[0]
        annotated = dict(node)
        annotated["root_id"] = root_id
        annotated["branch_id"] = root_id
        annotated["path_ids"] = path_ids
        annotated["path_signature"] = _stable_path_signature(path_ids)
        annotated_nodes.append(annotated)

    return {
        "branch_identity_version": BRANCH_IDENTITY_VERSION,
        "branch_identity": {
            "type": "root_path_signature",
            "path_order": "root_to_node",
            "hash": "sha256",
        },
        "nodes": annotated_nodes,
    }
