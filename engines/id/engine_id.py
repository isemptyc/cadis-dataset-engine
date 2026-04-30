from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.de.engine_de import GermanyAdminEngine

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "indonesia"

ID_PROFILE = AdminProfile(
    name_keys=("name:id", "name", "name:en", "official_name"),
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
        7: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=("ar", "en", "id", "ko", "ms-arab"),
)


class IndonesiaAdminEngine(GermanyAdminEngine):
    ENGINE = "id_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 5, 6, 7]
    ALLOWED_SHAPES = {
        (4,),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 7),
        (4, 5, 7),
        (4, 6),
        (4, 6, 7),
        (4, 7),
        (5,),
        (5, 6),
        (5, 6, 7),
        (5, 7),
        (6,),
        (6, 7),
        (7,),
    }

    COUNTRY_ISO = "ID"
    COUNTRY_NAME = "Indonesia"
    RUNTIME_POLICY_VERSION = "1.0"
    EXCLUDED_FEATURE_IDS = {
        "id_r3725861",   # Oe-Cusse Ambeno (East Timor)
        "id_r4631017",   # Aileu (East Timor)
        "id_r4631018",   # Ainaro (East Timor)
        "id_r4631019",   # Baucau (East Timor)
        "id_r4631020",   # Bobonaro (East Timor)
        "id_r4631021",   # Cova-Lima (East Timor)
        "id_r4631022",   # Dili (East Timor)
        "id_r4631023",   # Ermera (East Timor)
        "id_r4631024",   # Lautem (East Timor)
        "id_r4631025",   # Liquica (East Timor)
        "id_r4631027",   # Manatuto (East Timor)
        "id_r4631028",   # Manufahi (East Timor)
        "id_r4631029",   # Viqueque (East Timor)
        "id_r12363798",  # Atauro (East Timor)
    }

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

        self._admin_dataset_path = self._work_dir / "indonesia_admin.json"
        self._ffsf_dataset_path = self._work_dir / "indonesia_admin.bin"
        self._ffsf_meta_path = self._work_dir / "ID_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "indonesia_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "IndonesiaAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=ID_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_province",
                    5: "admin_regency_city",
                    6: "admin_district",
                    7: "admin_village",
                },
                id_prefix="id",
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
            from ffsf import export_cadis_to_ffsf

            export_cadis_to_ffsf(
                input_path=self._admin_dataset_path,
                output_path=self._ffsf_dataset_path,
                version=3,
                country_geometry_path=self._country_geometry_path,
            )

        if not self._semantic_dataset_path.exists():
            from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

            semantic_nodes = self._build_semantic_nodes()
            export_admin_semantic_dataset(
                nodes=semantic_nodes,
                output_path=self._semantic_dataset_path,
                version="id-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": list(self.LEVELS),
            "allowed_shapes": [list(shape) for shape in sorted(self.ALLOWED_SHAPES)],
            "shape_status": [
                {
                    "levels": list(shape),
                    "status": "ok"
                    if 4 in shape and any(level in shape for level in (5, 6, 7))
                    else "partial",
                }
                for shape in sorted(self.ALLOWED_SHAPES)
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [5, 6, 7],
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
