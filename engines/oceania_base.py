from __future__ import annotations

from itertools import combinations
from pathlib import Path

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.br.engine_br import BrazilAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset


def all_nonempty_level_shapes(levels: tuple[int, ...]) -> set[tuple[int, ...]]:
    return {
        shape
        for size in range(1, len(levels) + 1)
        for shape in combinations(levels, size)
    }


def build_oceania_profile(*, name_keys: tuple[str, ...], levels: tuple[int, ...], languages: tuple[str, ...]) -> AdminProfile:
    level_policies = {
        level: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01 if level in {2, 3, 4} else 0.001,
            fix_invalid=True,
            parent_resolution="strict",
        )
        for level in levels
    }
    return AdminProfile(
        name_keys=name_keys,
        level_policies=level_policies,
        parent_fallback=False,
        multilingual_names_enabled=True,
        multilingual_allowed_languages=languages,
    )


def default_level_labels(levels: tuple[int, ...]) -> dict[int, str]:
    labels = {
        2: "admin_country",
        3: "admin_region",
        4: "admin_region",
        5: "admin_district",
        6: "admin_municipality",
        7: "admin_locality",
        8: "admin_locality",
        9: "admin_locality",
        10: "admin_detail",
    }
    return {level: labels.get(level, f"admin_level_{level}") for level in levels}


class OceaniaAdminEngineBase(BrazilAdminEngine):
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"
    RUNTIME_POLICY_VERSION = "1.0"

    PROFILE: AdminProfile
    LEVEL_LABELS: dict[int, str]
    WORK_DIR_NAME: str
    FILE_STEM: str

    def __init__(
        self,
        *,
        osm_pbf_path: str | Path | None = None,
        work_dir: Path | None = None,
        country_geometry_path: str | Path | None = None,
    ):
        self._work_dir = (
            Path(work_dir)
            if work_dir
            else Path.home() / ".cache" / "cadis_dataset_engine" / self.WORK_DIR_NAME
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._country_geometry_path = (
            Path(country_geometry_path) if country_geometry_path is not None else None
        )

        self._admin_dataset_path = self._work_dir / f"{self.FILE_STEM}_admin.json"
        self._ffsf_dataset_path = self._work_dir / f"{self.FILE_STEM}_admin.bin"
        self._ffsf_meta_path = self._work_dir / f"{self.COUNTRY_ISO}_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / f"{self.FILE_STEM}_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                f"{self.__class__.__name__} in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path."
            )
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            build_admin_dataset(
                pbf_path=osm_pbf_path,
                output_path=self._admin_dataset_path,
                levels=self.LEVELS,
                profile=self.PROFILE,
                fallback_policy=None,
                country_code=self.COUNTRY_ISO,
                country_name=self.COUNTRY_NAME,
                level_labels=self.LEVEL_LABELS,
                id_prefix=self.COUNTRY_ISO.lower(),
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
            export_admin_semantic_dataset(
                nodes=self._build_semantic_nodes(),
                output_path=self._semantic_dataset_path,
                version=f"{self.COUNTRY_ISO.lower()}-admin-semantic-1.0.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )
        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        allowed_shapes = [list(shape) for shape in sorted(self.ALLOWED_SHAPES)]
        parent_level = self.LEVELS[0]
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": self.LEVELS,
            "allowed_shapes": allowed_shapes,
            "shape_status": [
                {
                    "levels": shape,
                    "status": "ok" if parent_level in shape else "partial",
                }
                for shape in allowed_shapes
            ],
            "layers": {"hierarchy_required": True, "repair_required": False},
            "hierarchy_repair_rules": {
                "parent_level": parent_level,
                "child_levels": [level for level in self.LEVELS if level != parent_level],
            },
            "repair_rules": {"parent_level": parent_level, "child_levels": []},
            "nearby_policy": {
                "enabled": True,
                "max_distance_km": 2.0,
                "offshore_max_distance_km": 20.0,
            },
        }
