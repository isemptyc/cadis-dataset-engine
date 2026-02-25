from pathlib import Path
import json
import shutil

from base import LookupSystemBase
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

"""
TaiwanAdminLookup
├── SpatialGate            (country bbox + geometry)
├── PolygonEvidence        (level → polygons)
├── HierarchyKnowledge     (admin_tree.txt → parent truth)
├── SemanticCorrection     (Level 4 via admin_tree when polygons fail)
└── CanonicalProjection    (rank-based output + status)
"""

DEFAULT_WORK_DIR = Path.home() / ".cache" / "admin_lookup" / "taiwan"

# ==================================================
# Admin Profile (Taiwan)
# ==================================================

TW_PROFILE = AdminProfile(
    name_keys=("name:zh-Hant", "name:zh", "name"),
    level_policies={
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.0005,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        7: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
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

# ==================================================
# Taiwan Admin Lookup Engine
# ==================================================

class TaiwanAdminEngine(LookupSystemBase):
    ENGINE = "taiwan_admin"
    VERSION = "v2.0"

    """
    Levels are fixed to [4, 7, 8] and represent the only semantically meaningful
    administrative levels for Taiwan in this engine.
    """
    LEVELS = [4, 7, 8]
    ALLOWED_SHAPES = {
        (4,),
        (4, 7),
        (4, 8),
        (4, 7, 8),
    }

    COUNTRY_ISO = "TW"
    COUNTRY_NAME = "Taiwan"
    RUNTIME_POLICY_VERSION = "1.0"

    def __init__(
        self,
        *,
        osm_pbf_path: str | Path | None = None,
        work_dir: Path | None = None,
        use_ffsf: bool = True,  # retained for API compatibility; unused in build-only mode
    ):
        self._use_ffsf = use_ffsf
        self._work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # ---- dataset paths (external, writable) ----
        self._admin_dataset_path = self._work_dir / "taiwan_admin.json"
        self._ffsf_dataset_path = self._work_dir / "taiwan_admin.bin"
        self._ffsf_meta_path = self._work_dir / "TW_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "taiwan_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "TaiwanAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(
            osm_pbf_path=str(osm_pbf_path),
        )

    # ==================================================
    # dataset preparation (engine responsibility)
    # ==================================================

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        """
        Ensure all engine-required datasets exist on disk.
        This method is idempotent and may perform heavy I/O on first run.
        """
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=TW_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={4: "admin_city", 7: "admin_district", 8: "admin_subcity"},
                id_prefix="tw",
                country_geometry_path=None,
            )

        if not self._admin_hierarchy_path.exists():
            nodes_path = self._work_dir / "admin_nodes.json"
            edges_path = self._work_dir / "admin_edges.json"

            if not nodes_path.exists() or not edges_path.exists():
                extract_admin_hierarchy(
                    pbf_path=osm_pbf_path,
                    output_dir=self._work_dir,
                    name_keys=TW_PROFILE.name_keys,
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
            hierarchy_nodes = self._load_admin_hierarchy(self._admin_hierarchy_path)
            semantic_nodes = [
                {
                    "feature_id": n["id"],
                    "level": n["level"],
                    "name": n["name"],
                    "parent_id": n.get("parent_id"),
                }
                for n in hierarchy_nodes
            ]
            export_admin_semantic_dataset(
                nodes=semantic_nodes,
                output_path=self._semantic_dataset_path,
                version="tw-admin-semantic-1.0.0",
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
            "allowed_levels": list(self.LEVELS),
            "allowed_shapes": [list(shape) for shape in sorted(self.ALLOWED_SHAPES)],
            "shape_status": [
                {"levels": [4], "status": "ok"},
                {"levels": [4, 7], "status": "ok"},
                {"levels": [4, 8], "status": "ok"},
                {"levels": [4, 7, 8], "status": "ok"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [7, 8],
            },
            "repair_rules": {
                "parent_level": 4,
                "child_levels": [],
            },
        }
