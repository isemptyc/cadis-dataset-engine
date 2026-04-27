from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

from base import DatasetBuildEngineBase
from dataset import (
    AdminLevelPolicy,
    AdminProfile,
    build_admin_dataset,
    render_admin_tree,
)
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "portugal"

PT_PROFILE = AdminProfile(
    name_keys=("name:pt", "name", "official_name", "name:en"),
    level_policies={
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        6: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.003,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        7: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.001,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        8: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=("pt", "en", "es", "gl", "fr"),
)


class PortugalAdminEngine(DatasetBuildEngineBase):
    ENGINE = "pt_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 6, 7, 8]
    ALLOWED_SHAPES = {
        (4,),
        (4, 7),
        (4, 7, 8),
        (4, 8),
        (6,),
        (6, 7),
        (6, 7, 8),
        (6, 8),
        (7,),
        (7, 8),
        (8,),
    }

    COUNTRY_ISO = "PT"
    COUNTRY_NAME = "Portugal"
    RUNTIME_POLICY_VERSION = "1.0"

    def __init__(
        self,
        *,
        osm_pbf_path: str | Path | None = None,
        work_dir: Path | None = None,
        country_geometry_path: str | Path | None = None,
    ):
        self._work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._country_geometry_path = (
            Path(country_geometry_path) if country_geometry_path is not None else None
        )

        self._admin_dataset_path = self._work_dir / "portugal_admin.json"
        self._ffsf_dataset_path = self._work_dir / "portugal_admin.bin"
        self._ffsf_meta_path = self._work_dir / "PT_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "portugal_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "PortugalAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=PT_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_autonomous_region",
                    6: "admin_district",
                    7: "admin_municipality",
                    8: "admin_civil_parish",
                },
                id_prefix="pt",
                country_geometry_path=self._country_geometry_path,
            )

        if not self._admin_hierarchy_path.exists():
            self._write_dataset_scoped_hierarchy_artifacts()

        if not self._ffsf_dataset_path.exists() or not self._ffsf_meta_path.exists():
            if not self._admin_dataset_path.exists():
                raise FileNotFoundError(
                    f"Missing admin dataset required for FFSF export: {self._admin_dataset_path}"
                )
            export_cadis_to_ffsf(
                input_path=self._admin_dataset_path,
                output_path=self._ffsf_dataset_path,
                version=3,
                country_geometry_path=self._country_geometry_path,
            )

        if not self._semantic_dataset_path.exists():
            semantic_nodes = self._build_semantic_nodes()
            export_admin_semantic_dataset(
                nodes=semantic_nodes,
                output_path=self._semantic_dataset_path,
                version="pt-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _release_dataset_paths(self) -> list[Path]:
        paths = []
        for p in (
            self._runtime_policy_path(),
            self._runtime_geometry_path,
            self._runtime_geometry_meta_path,
            self._runtime_hierarchy_path,
        ):
            if p.exists():
                paths.append(p)
        return paths

    def _write_dataset_build_manifest(self) -> Path:
        self._ensure_runtime_release_layers()
        return super()._write_dataset_build_manifest()

    def _normalize_runtime_feature_id(self, feature_id: str | None) -> str | None:
        if feature_id is None:
            return None
        prefix = f"{self.COUNTRY_ISO.lower()}_"
        if feature_id.startswith(prefix):
            return feature_id[len(prefix):]
        return feature_id

    def _load_admin_dataset_nodes(self) -> list[dict]:
        payload = json.loads(self._admin_dataset_path.read_text(encoding="utf-8"))
        admin_by_level = payload.get("admin_by_level", {})
        if not isinstance(admin_by_level, dict):
            return []

        nodes: list[dict] = []
        for level in self.LEVELS:
            rows = admin_by_level.get(str(level), [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                feature_id = row.get("id")
                name = row.get("name")
                if not isinstance(feature_id, str) or not isinstance(name, str) or not name:
                    continue
                parent_id = row.get("parent")
                if parent_id is not None and not isinstance(parent_id, str):
                    parent_id = None
                nodes.append(
                    {
                        "id": self._normalize_runtime_feature_id(feature_id),
                        "level": level,
                        "name": name,
                        "names": row.get("names") if isinstance(row.get("names"), dict) else None,
                        "parent_id": self._normalize_runtime_feature_id(parent_id),
                    }
                )
        return nodes

    def _write_dataset_scoped_hierarchy_artifacts(self) -> None:
        dataset_nodes = self._load_admin_dataset_nodes()
        level_counts: dict[int, int] = {}
        hierarchy_nodes: list[dict] = []
        hierarchy_edges: list[dict] = []

        for node in dataset_nodes:
            node_id = node["id"]
            raw_id = f"{self.COUNTRY_ISO.lower()}_{node_id}"
            level = node["level"]
            parent_id = node.get("parent_id")
            raw_parent_id = None
            if isinstance(parent_id, str) and parent_id:
                raw_parent_id = f"{self.COUNTRY_ISO.lower()}_{parent_id}"

            hierarchy_nodes.append(
                {
                    "id": raw_id,
                    "osm_id": raw_id[3:] if raw_id.startswith(f"{self.COUNTRY_ISO.lower()}_") else raw_id,
                    "name": node["name"],
                    "names": node.get("names"),
                    "admin_level": level,
                    "tags": {
                        "boundary": "administrative",
                        "admin_level": str(level),
                    },
                }
            )
            level_counts[level] = level_counts.get(level, 0) + 1

            if raw_parent_id is not None:
                hierarchy_edges.append(
                    {
                        "parent": raw_parent_id,
                        "child": raw_id,
                        "method": "dataset_parent",
                        "confidence": 1.0,
                    }
                )

        nodes_path = self._work_dir / "admin_nodes.json"
        edges_path = self._work_dir / "admin_edges.json"
        report_path = self._work_dir / "admin_report.json"

        nodes_path.write_text(
            json.dumps(hierarchy_nodes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        edges_path.write_text(
            json.dumps(hierarchy_edges, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "country_geometry_filter_applied": self._country_geometry_path is not None,
                    "dataset_scope_projection_applied": True,
                    "node_count": len(hierarchy_nodes),
                    "edge_count": len(hierarchy_edges),
                    "unresolved_is_in_edges": 0,
                    "admin_level_distribution": {
                        str(level): count for level, count in sorted(level_counts.items())
                    },
                    "unresolved_samples": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        render_admin_tree(
            nodes_path=nodes_path,
            edges_path=edges_path,
            output_path=self._admin_hierarchy_path,
        )

    def _load_tree_parent_candidates(self) -> dict[str, list[dict]]:
        hierarchy_nodes = self._load_admin_hierarchy(self._admin_hierarchy_path)
        candidates: dict[str, list[dict]] = {}
        for node in hierarchy_nodes:
            if node["level"] not in self.LEVELS:
                continue
            node_id = self._normalize_runtime_feature_id(node["id"])
            parent_id = self._normalize_runtime_feature_id(node.get("parent_id"))
            candidates.setdefault(node_id, []).append(
                {
                    "level": node["level"],
                    "parent_id": parent_id,
                }
            )
        return candidates

    def _build_semantic_nodes(self) -> list[dict]:
        dataset_nodes = self._load_admin_dataset_nodes()
        node_by_id = {node["id"]: node for node in dataset_nodes}
        node_ids = set(node_by_id.keys())
        tree_candidates = self._load_tree_parent_candidates()

        supplemented_parents: dict[str, str | None] = {}
        for node in dataset_nodes:
            parent_id = node.get("parent_id")
            if parent_id in node_ids:
                supplemented_parents[node["id"]] = parent_id
                continue

            child_level = node["level"]
            valid_candidates: list[tuple[int, str]] = []
            for candidate in tree_candidates.get(node["id"], []):
                candidate_parent = candidate.get("parent_id")
                if candidate_parent not in node_ids:
                    continue
                parent_node = node_by_id.get(candidate_parent)
                if parent_node is None or parent_node["level"] >= child_level:
                    continue
                valid_candidates.append((parent_node["level"], candidate_parent))

            if valid_candidates:
                valid_candidates.sort(key=lambda item: (-item[0], item[1]))
                supplemented_parents[node["id"]] = valid_candidates[0][1]
            else:
                supplemented_parents[node["id"]] = None

        semantic_nodes: list[dict] = []
        for node in sorted(dataset_nodes, key=lambda n: (n["level"], n["name"], n["id"])):
            semantic_nodes.append(
                {
                    "feature_id": node["id"],
                    "level": node["level"],
                    "name": node["name"],
                    "names": node.get("names") if isinstance(node.get("names"), dict) else None,
                    "parent_id": supplemented_parents[node["id"]],
                }
            )
        return semantic_nodes

    def _load_semantic_nodes(self) -> list[dict]:
        payload = json.loads(self._semantic_dataset_path.read_text(encoding="utf-8"))
        nodes_raw = payload.get("nodes", {})
        out = []
        if not isinstance(nodes_raw, dict):
            return out
        for feature_id, row in nodes_raw.items():
            if not isinstance(row, list) or len(row) not in {3, 4}:
                continue
            if len(row) == 3:
                level, name, parent_id = row
                names = None
            else:
                level, name, parent_id, names = row
            if not isinstance(level, int):
                continue
            if not isinstance(name, str) or not name:
                continue
            if parent_id is not None and not isinstance(parent_id, str):
                continue
            if names is not None and not isinstance(names, dict):
                continue
            out.append(
                {
                    "id": feature_id,
                    "level": level,
                    "name": name,
                    "names": names if isinstance(names, dict) and names else None,
                    "parent_id": parent_id,
                }
            )
        return out

    def _ensure_runtime_release_layers(self) -> None:
        if not self._ffsf_dataset_path.exists() or not self._ffsf_meta_path.exists():
            return

        if (
            not self._runtime_geometry_path.exists()
            or self._runtime_geometry_path.stat().st_size != self._ffsf_dataset_path.stat().st_size
        ):
            shutil.copy2(self._ffsf_dataset_path, self._runtime_geometry_path)

        if (
            not self._runtime_geometry_meta_path.exists()
            or self._runtime_geometry_meta_path.stat().st_size != self._ffsf_meta_path.stat().st_size
        ):
            shutil.copy2(self._ffsf_meta_path, self._runtime_geometry_meta_path)

        semantic_nodes = self._load_semantic_nodes()
        self._runtime_hierarchy_path.write_text(
            json.dumps({"nodes": semantic_nodes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [4, 6, 7, 8],
            "allowed_shapes": [
                [4],
                [4, 7],
                [4, 7, 8],
                [4, 8],
                [6],
                [6, 7],
                [6, 7, 8],
                [6, 8],
                [7],
                [7, 8],
                [8],
            ],
            "shape_status": [
                {"levels": [4], "status": "partial"},
                {"levels": [4, 7], "status": "ok"},
                {"levels": [4, 7, 8], "status": "ok"},
                {"levels": [4, 8], "status": "partial"},
                {"levels": [6], "status": "partial"},
                {"levels": [6, 7], "status": "ok"},
                {"levels": [6, 7, 8], "status": "ok"},
                {"levels": [6, 8], "status": "partial"},
                {"levels": [7], "status": "partial"},
                {"levels": [7, 8], "status": "partial"},
                {"levels": [8], "status": "partial"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 6,
                "child_levels": [7, 8],
            },
            "repair_rules": {
                "parent_level": 6,
                "child_levels": [],
            },
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 2.0,
                "offshore_max_distance_km": 20.0,
            },
        }
