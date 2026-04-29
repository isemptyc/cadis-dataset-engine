from pathlib import Path
import json

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.de.engine_de import GermanyAdminEngine

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "malaysia"

MY_PROFILE = AdminProfile(
    name_keys=("name:ms", "name", "name:en", "official_name"),
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
        10: AdminLevelPolicy(
            simplify=False,
            simplify_tolerance=None,
            fix_invalid=False,
            parent_resolution="strict",
        ),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=("zh", "ar", "en", "ms", "ta"),
)


class MalaysiaAdminEngine(GermanyAdminEngine):
    ENGINE = "my_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 5, 6, 7, 8, 10]
    ALLOWED_SHAPES = {
        (4,),
        (4, 5),
        (4, 5, 6),
        (4, 5, 6, 7),
        (4, 5, 6, 7, 8),
        (4, 5, 6, 7, 8, 10),
        (4, 5, 6, 7, 10),
        (4, 5, 6, 8),
        (4, 5, 6, 8, 10),
        (4, 5, 6, 10),
        (4, 5, 7),
        (4, 5, 7, 8),
        (4, 5, 7, 8, 10),
        (4, 5, 7, 10),
        (4, 5, 8),
        (4, 5, 8, 10),
        (4, 5, 10),
        (4, 6),
        (4, 6, 7),
        (4, 6, 7, 8),
        (4, 6, 7, 8, 10),
        (4, 6, 7, 10),
        (4, 6, 8),
        (4, 6, 8, 10),
        (4, 6, 10),
        (4, 7),
        (4, 7, 8),
        (4, 7, 8, 10),
        (4, 7, 10),
        (4, 8),
        (4, 8, 10),
        (4, 10),
        (5,),
        (5, 6),
        (5, 6, 7),
        (5, 6, 7, 8),
        (5, 6, 7, 8, 10),
        (5, 6, 7, 10),
        (5, 6, 8),
        (5, 6, 8, 10),
        (5, 6, 10),
        (5, 7),
        (5, 7, 8),
        (5, 7, 8, 10),
        (5, 7, 10),
        (5, 8),
        (5, 8, 10),
        (5, 10),
        (6,),
        (6, 7),
        (6, 7, 8),
        (6, 7, 8, 10),
        (6, 7, 10),
        (6, 8),
        (6, 8, 10),
        (6, 10),
        (7,),
        (7, 8),
        (7, 8, 10),
        (7, 10),
        (8,),
        (8, 10),
        (10,),
    }

    COUNTRY_ISO = "MY"
    COUNTRY_NAME = "Malaysia"
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

        self._admin_dataset_path = self._work_dir / "malaysia_admin.json"
        self._ffsf_dataset_path = self._work_dir / "malaysia_admin.bin"
        self._ffsf_meta_path = self._work_dir / "MY_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "malaysia_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "MalaysiaAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=MY_PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels={
                    4: "admin_state_federal_territory",
                    5: "admin_division",
                    6: "admin_district",
                    7: "admin_local_authority",
                    8: "admin_subdistrict",
                    10: "admin_neighborhood",
                },
                id_prefix="my",
                country_geometry_path=self._country_geometry_path,
            )

        self._apply_dataset_overrides()
        self._inject_missing_admin1_supplements()

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
                version="my-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _inject_missing_admin1_supplements(self) -> None:
        if self._country_geometry_path is None or not self._country_geometry_path.exists():
            return
        if not self._admin_dataset_path.exists():
            return

        boundary_payload = json.loads(self._country_geometry_path.read_text(encoding="utf-8"))
        supplements = boundary_payload.get("admin1_supplements")
        if not isinstance(supplements, list) or not supplements:
            return

        payload = json.loads(self._admin_dataset_path.read_text(encoding="utf-8"))
        admin_by_level = payload.get("admin_by_level")
        if not isinstance(admin_by_level, dict):
            return

        level4_rows = admin_by_level.setdefault("4", [])
        if not isinstance(level4_rows, list):
            return

        existing_names = {
            row.get("name")
            for row in level4_rows
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        }
        existing_ids = {
            row.get("id")
            for row in level4_rows
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }

        changed = False
        for row in supplements:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            feature_id = row.get("id")
            geometry = row.get("geometry")
            if not isinstance(name, str) or not isinstance(feature_id, str):
                continue
            if not isinstance(geometry, dict):
                continue
            if name in existing_names or feature_id in existing_ids:
                continue
            level4_rows.append(
                {
                    "id": feature_id,
                    "osm_id": row.get("osm_id", feature_id),
                    "name": name,
                    "level": 4,
                    "bbox": row.get("bbox"),
                    "names": row.get("names") if isinstance(row.get("names"), dict) else None,
                    "geometry": geometry,
                }
            )
            existing_names.add(name)
            existing_ids.add(feature_id)
            changed = True

        if changed:
            level4_rows.sort(key=lambda item: (str(item.get("name", "")), str(item.get("id", ""))))
            self._admin_dataset_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _runtime_policy_payload(self) -> dict:
        allowed_shapes = [list(shape) for shape in sorted(self.ALLOWED_SHAPES)]
        shape_status = []
        for shape in sorted(self.ALLOWED_SHAPES):
            has_state = 4 in shape
            has_district_or_better = any(level in shape for level in (5, 6, 7, 8, 10))
            status = "ok" if has_state and has_district_or_better else "partial"
            shape_status.append({"levels": list(shape), "status": status})

        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": list(self.LEVELS),
            "allowed_shapes": allowed_shapes,
            "shape_status": shape_status,
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [5, 6, 7, 8, 10],
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
