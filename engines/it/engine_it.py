from pathlib import Path
import json
import shutil

from base import DatasetBuildEngineBase
from dataset import (
    AdminLevelPolicy,
    AdminProfile,
    build_admin_dataset,
    extract_admin_hierarchy,
    render_admin_tree,
)
from ffsf import (
    export_cadis_to_ffsf,
)
from ffsf.semantic_dataset_exporter import (
    export_admin_semantic_dataset,
)

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "italy"

IT_PROFILE = AdminProfile(
    name_keys=("name:it", "name", "name:en", "official_name"),
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
        8: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
)


class ItalyAdminEngine(DatasetBuildEngineBase):
    ENGINE = "it_admin"
    VERSION = "v1.0"

    LEVELS = [4, 6, 8]
    ALLOWED_SHAPES = {
        (4,),
        (4, 6),
        (4, 6, 8),
        (4, 8),
        (6,),
        (6, 8),
        (8,),
    }

    COUNTRY_ISO = "IT"
    COUNTRY_NAME = "Italy"
    RUNTIME_POLICY_VERSION = "1.0"

    def __init__(
        self,
        *,
        osm_pbf_path: str | Path | None = None,
        work_dir: Path | None = None,
    ):
        self._work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._admin_dataset_path = self._work_dir / "italy_admin.json"
        self._ffsf_dataset_path = self._work_dir / "italy_admin.bin"
        self._ffsf_meta_path = self._work_dir / "IT_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "italy_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "ItalyAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=IT_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_region",
                    6: "admin_province",
                    8: "admin_municipality",
                },
                id_prefix="it",
                country_geometry_path=None,
            )

        if not self._admin_hierarchy_path.exists():
            nodes_path = self._work_dir / "admin_nodes.json"
            edges_path = self._work_dir / "admin_edges.json"

            if not nodes_path.exists() or not edges_path.exists():
                extract_admin_hierarchy(
                    pbf_path=osm_pbf_path,
                    output_dir=self._work_dir,
                    name_keys=IT_PROFILE.name_keys,
                    target_levels=self.LEVELS,
                )

            render_admin_tree(
                nodes_path=nodes_path,
                edges_path=edges_path,
                output_path=self._admin_hierarchy_path,
            )

        if not self._ffsf_dataset_path.exists() or not self._ffsf_meta_path.exists():
            if not self._admin_dataset_path.exists():
                raise FileNotFoundError(
                    f"Missing admin dataset required for FFSF export: {self._admin_dataset_path}"
                )
            export_cadis_to_ffsf(
                input_path=self._admin_dataset_path,
                output_path=self._ffsf_dataset_path,
                version=3,
            )

        if not self._semantic_dataset_path.exists():
            semantic_nodes = self._build_semantic_nodes()
            export_admin_semantic_dataset(
                nodes=semantic_nodes,
                output_path=self._semantic_dataset_path,
                version="it-admin-semantic-1.0.0",
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
                        "parent_id": self._normalize_runtime_feature_id(parent_id),
                    }
                )
        return nodes

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
            if not isinstance(row, list) or len(row) != 3:
                continue
            level, name, parent_id = row
            if not isinstance(level, int):
                continue
            if not isinstance(name, str) or not name:
                continue
            if parent_id is not None and not isinstance(parent_id, str):
                continue
            out.append(
                {
                    "id": feature_id,
                    "level": level,
                    "name": name,
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
            "allowed_levels": [4, 6, 8],
            "allowed_shapes": [
                [4],
                [4, 6],
                [4, 6, 8],
                [4, 8],
                [6],
                [6, 8],
                [8],
            ],
            "shape_status": [
                {"levels": [4], "status": "partial"},
                {"levels": [4, 6], "status": "ok"},
                {"levels": [4, 6, 8], "status": "ok"},
                {"levels": [4, 8], "status": "ok"},
                {"levels": [6], "status": "partial"},
                {"levels": [6, 8], "status": "partial"},
                {"levels": [8], "status": "partial"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [6, 8],
            },
            "repair_rules": {
                "parent_level": 4,
                "child_levels": [],
            },
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 2.0,
                "offshore_max_distance_km": 20.0,
            },
        }
