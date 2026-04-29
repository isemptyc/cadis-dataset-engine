from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.de.engine_de import GermanyAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "singapore"

SG_PROFILE = AdminProfile(
    name_keys=("name:en", "name", "official_name"),
    level_policies={
        6: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.001,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        11: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
)


class SingaporeAdminEngine(GermanyAdminEngine):
    ENGINE = "sg_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = None

    LEVELS = [6, 11]
    ALLOWED_SHAPES = {
        (6,),
        (6, 11),
        (11,),
    }

    COUNTRY_ISO = "SG"
    COUNTRY_NAME = "Singapore"
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

        self._admin_dataset_path = self._work_dir / "singapore_admin.json"
        self._ffsf_dataset_path = self._work_dir / "singapore_admin.bin"
        self._ffsf_meta_path = self._work_dir / "SG_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "singapore_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "SingaporeAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=SG_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    6: "admin_region",
                    11: "admin_neighborhood",
                },
                id_prefix="sg",
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
                version="sg-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [6, 11],
            "allowed_shapes": [
                [6],
                [6, 11],
                [11],
            ],
            "shape_status": [
                {"levels": [6], "status": "partial"},
                {"levels": [6, 11], "status": "ok"},
                {"levels": [11], "status": "partial"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 6,
                "child_levels": [11],
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
