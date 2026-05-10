from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.br.engine_br import BrazilAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "india"

IN_PROFILE = AdminProfile(
    name_keys=("name:en", "name", "official_name"),
    level_policies={
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        5: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.005,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        6: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.003,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        9: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=("en", "kn", "mr", "te", "ur"),
)


class IndiaAdminEngine(BrazilAdminEngine):
    ENGINE = "in_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 5, 6, 9]
    ALLOWED_SHAPES = {
        (4,),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 9),
        (4, 5, 9),
        (4, 6),
        (4, 6, 9),
        (4, 9),
        (5,),
        (5, 6),
        (5, 6, 9),
        (5, 9),
        (6,),
        (6, 9),
        (9,),
    }

    COUNTRY_ISO = "IN"
    COUNTRY_NAME = "India"
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

        self._admin_dataset_path = self._work_dir / "india_admin.json"
        self._ffsf_dataset_path = self._work_dir / "india_admin.bin"
        self._ffsf_meta_path = self._work_dir / "IN_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "india_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "IndiaAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=IN_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_state_union_territory",
                    5: "admin_district",
                    6: "admin_subdistrict",
                    9: "admin_village_locality",
                },
                id_prefix="in",
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
                version="in-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [4, 5, 6, 9],
            "allowed_shapes": [
                [4],
                [4, 5],
                [4, 5, 6],
                [4, 5, 6, 9],
                [4, 5, 9],
                [4, 6],
                [4, 6, 9],
                [4, 9],
                [5],
                [5, 6],
                [5, 6, 9],
                [5, 9],
                [6],
                [6, 9],
                [9],
            ],
            "shape_status": [
                {"levels": [4], "status": "partial"},
                {"levels": [4, 5], "status": "partial"},
                {"levels": [4, 5, 6], "status": "ok"},
                {"levels": [4, 5, 6, 9], "status": "ok"},
                {"levels": [4, 5, 9], "status": "partial"},
                {"levels": [4, 6], "status": "partial"},
                {"levels": [4, 6, 9], "status": "partial"},
                {"levels": [4, 9], "status": "partial"},
                {"levels": [5], "status": "partial"},
                {"levels": [5, 6], "status": "partial"},
                {"levels": [5, 6, 9], "status": "partial"},
                {"levels": [5, 9], "status": "partial"},
                {"levels": [6], "status": "partial"},
                {"levels": [6, 9], "status": "partial"},
                {"levels": [9], "status": "partial"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [5, 6, 9],
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
