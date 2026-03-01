from pathlib import Path
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

"""
JapanAdminDatasetBuild
├── Polygon extraction dataset (japan_admin.json)
├── Hierarchy text rendering   (admin_tree.txt)
└── Geometry runtime layer     (geometry.ffsf + geometry_meta.json)
"""

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "japan"

# ==================================================
# Admin Profile (Japan)
# ==================================================

JP_PROFILE = AdminProfile(
    name_keys=("name:ja", "name", "name:en"),
    level_policies={
        3: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.001,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        7: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
)


# ==================================================
# Japan Admin Dataset Build Engine
# ==================================================

class JapanAdminEngine(DatasetBuildEngineBase):
    ENGINE = "japan_admin"
    VERSION = "v2.0"

    """
    Levels are fixed to [3, 4, 7] and represent the only semantically meaningful
    administrative levels for Japan in this engine.
    """
    LEVELS = [3, 4, 7]
    ALLOWED_SHAPES = {
        (3,),
        (4,),
        (7,),
        (3, 4),
        (3, 7),
        (4, 7),
        (3, 4, 7),
    }

    COUNTRY_ISO = "JP"
    COUNTRY_NAME = "Japan"
    RUNTIME_POLICY_VERSION = "1.0"

    def __init__(
        self,
        *,
        osm_pbf_path: str | Path | None = None,
        work_dir: Path | None = None,
    ):
        self._work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # ---- dataset paths (external, writable) ----
        self._admin_dataset_path = self._work_dir / "japan_admin.json"
        self._ffsf_dataset_path = self._work_dir / "japan_admin.bin"
        self._ffsf_meta_path = self._work_dir / "JP_feature_meta_by_index.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"

        if osm_pbf_path is None:
            raise ValueError(
                "JapanAdminEngine in cadis-dataset-engine is build-only. "
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
                profile=JP_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    3: "admin_prefecture",
                    4: "admin_municipality",
                    7: "admin_subarea",
                },
                id_prefix="jp",
                country_geometry_path=None,
            )

        if not self._admin_hierarchy_path.exists():
            nodes_path = self._work_dir / "admin_nodes.json"
            edges_path = self._work_dir / "admin_edges.json"

            if not nodes_path.exists() or not edges_path.exists():
                extract_admin_hierarchy(
                    pbf_path=osm_pbf_path,
                    output_dir=self._work_dir,
                    name_keys=JP_PROFILE.name_keys,
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

        self._ensure_runtime_release_layers()

    def _release_dataset_paths(self) -> list[Path]:
        paths = []
        for p in (
            self._runtime_policy_path(),
            self._runtime_geometry_path,
            self._runtime_geometry_meta_path,
        ):
            if p.exists():
                paths.append(p)
        return paths

    def _write_dataset_build_manifest(self) -> Path:
        self._ensure_runtime_release_layers()
        return super()._write_dataset_build_manifest()

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

    def _runtime_policy_payload(self) -> dict:
        allowed_shapes = sorted(self.ALLOWED_SHAPES)
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": list(self.LEVELS),
            "allowed_shapes": [list(s) for s in allowed_shapes],
            "shape_status": [
                {
                    "levels": list(s),
                    "status": "ok" if ((3 in s and 4 in s) or s == (3, 7)) else "partial",
                }
                for s in allowed_shapes
            ],
            "layers": {
                "hierarchy_required": False,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 3,
                "child_levels": [4, 7],
            },
            "repair_rules": {
                "parent_level": 3,
                "child_levels": [],
            },
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 2.0,
                "offshore_max_distance_km": 20.0,
            },
        }
