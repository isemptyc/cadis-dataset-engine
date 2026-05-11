from itertools import combinations
from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.de.engine_de import GermanyAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "monaco"

MC_PROFILE = AdminProfile(
    name_keys=("name:fr", "name", "name:en", "official_name"),
    level_policies={
        8: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.0001,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        10: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=("ar", "fr", "it", "oc", "ru"),
)


def _all_nonempty_level_shapes(levels: tuple[int, ...]) -> set[tuple[int, ...]]:
    return {
        shape
        for size in range(1, len(levels) + 1)
        for shape in combinations(levels, size)
    }


class MonacoAdminEngine(GermanyAdminEngine):
    ENGINE = "mc_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [8, 10]
    ALLOWED_SHAPES = _all_nonempty_level_shapes((8, 10))

    COUNTRY_ISO = "MC"
    COUNTRY_NAME = "Monaco"
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

        self._admin_dataset_path = self._work_dir / "monaco_admin.json"
        self._ffsf_dataset_path = self._work_dir / "monaco_admin.bin"
        self._ffsf_meta_path = self._work_dir / "MC_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "monaco_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "MonacoAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=MC_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    8: "admin_city",
                    10: "admin_quarter",
                },
                id_prefix="mc",
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
                version="mc-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        allowed_shapes = [list(shape) for shape in sorted(self.ALLOWED_SHAPES)]
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [8, 10],
            "allowed_shapes": allowed_shapes,
            "shape_status": [
                {
                    "levels": shape,
                    "status": "ok" if 8 in shape else "partial",
                }
                for shape in allowed_shapes
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 8,
                "child_levels": [10],
            },
            "repair_rules": {
                "parent_level": 8,
                "child_levels": [],
            },
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 1.0,
                "offshore_max_distance_km": 5.0,
            },
        }
