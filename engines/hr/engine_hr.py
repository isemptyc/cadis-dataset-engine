from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.de.engine_de import GermanyAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "croatia"

HR_PROFILE = AdminProfile(
    name_keys=("name:hr", "name", "name:en", "official_name"),
    level_policies={
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01,
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
    multilingual_allowed_languages=("hr", "it", "sr", "sr-latn", "en"),
)


class CroatiaAdminEngine(GermanyAdminEngine):
    ENGINE = "hr_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 7, 8]
    ALLOWED_SHAPES = {
        (4,),
        (4, 7),
        (4, 7, 8),
        (4, 8),
        (7,),
        (7, 8),
        (8,),
    }

    COUNTRY_ISO = "HR"
    COUNTRY_NAME = "Croatia"
    RUNTIME_POLICY_VERSION = "1.0"
    EXCLUDED_FEATURE_IDS: set[str] = set()

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

        self._admin_dataset_path = self._work_dir / "croatia_admin.json"
        self._ffsf_dataset_path = self._work_dir / "croatia_admin.bin"
        self._ffsf_meta_path = self._work_dir / "HR_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "croatia_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "CroatiaAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=HR_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_county",
                    7: "admin_city_or_municipality",
                    8: "admin_settlement",
                },
                id_prefix="hr",
                country_geometry_path=self._country_geometry_path,
            )

        self._apply_dataset_overrides()

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
                version="hr-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [4, 7, 8],
            "allowed_shapes": [
                [4],
                [4, 7],
                [4, 7, 8],
                [4, 8],
                [7],
                [7, 8],
                [8],
            ],
            "shape_status": [
                {"levels": [4], "status": "ok"},
                {"levels": [4, 7], "status": "ok"},
                {"levels": [4, 7, 8], "status": "ok"},
                {"levels": [4, 8], "status": "ok"},
                {"levels": [7], "status": "partial"},
                {"levels": [7, 8], "status": "partial"},
                {"levels": [8], "status": "partial"},
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
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 2.0,
                "offshore_max_distance_km": 20.0,
            },
        }
