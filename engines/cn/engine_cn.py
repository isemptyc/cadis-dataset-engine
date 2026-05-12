from itertools import combinations
import json
from pathlib import Path
from datetime import datetime

from dataset import AdminLevelPolicy, AdminProfile, build_admin_dataset
from engines.br.engine_br import BrazilAdminEngine
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "china"

CN_PROFILE = AdminProfile(
    name_keys=("name:en", "name", "official_name", "name:zh", "name:bo", "name:ru"),
    level_policies={
        4: AdminLevelPolicy(simplify=True, simplify_tolerance=0.01, fix_invalid=True, parent_resolution="strict"),
        5: AdminLevelPolicy(simplify=True, simplify_tolerance=0.002, fix_invalid=True, parent_resolution="strict"),
        6: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        7: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        8: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        9: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        10: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        11: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        12: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
        14: AdminLevelPolicy(simplify=True, simplify_tolerance=0.001, fix_invalid=True, parent_resolution="strict"),
    },
    parent_fallback=False,
    multilingual_names_enabled=True,
    multilingual_allowed_languages=('en', 'zh', 'bo', 'ru'),
)


def _all_nonempty_level_shapes(levels: tuple[int, ...]) -> set[tuple[int, ...]]:
    return {shape for size in range(1, len(levels) + 1) for shape in combinations(levels, size)}


class ChinaAdminEngine(BrazilAdminEngine):
    ENGINE = "cn_admin"
    VERSION = "v1.0"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    ALLOWED_SHAPES = _all_nonempty_level_shapes((4, 5, 6, 7, 8, 9, 10, 11, 12, 14,))

    COUNTRY_ISO = "CN"
    COUNTRY_NAME = "China"
    RUNTIME_POLICY_VERSION = "1.0"

    def __init__(self, *, osm_pbf_path: str | Path | None = None, work_dir: Path | None = None, country_geometry_path: str | Path | None = None):
        self._work_dir = Path(work_dir) if work_dir else DEFAULT_WORK_DIR
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._country_geometry_path = Path(country_geometry_path) if country_geometry_path is not None else None
        self._admin_dataset_path = self._work_dir / "china_admin.json"
        self._ffsf_dataset_path = self._work_dir / "china_admin.bin"
        self._ffsf_meta_path = self._work_dir / "CN_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "china_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"
        if osm_pbf_path is None:
            raise ValueError("ChinaAdminEngine in cadis-dataset-engine is build-only. Provide osm_pbf_path.")
        self._ensure_datasets(osm_pbf_path=str(osm_pbf_path))

    @staticmethod
    def _source_input_paths(osm_pbf_path: str | Path) -> list[Path]:
        path = Path(osm_pbf_path)
        if not path.is_dir():
            return [path]

        source_paths: list[Path] = []
        for stem in ("china", "tibet"):
            matches = sorted(path.glob(f"{stem}-*.osm.pbf"))
            if not matches:
                raise FileNotFoundError(f"Missing CN source component matching {stem}-*.osm.pbf in {path}")
            source_paths.append(matches[-1])
        return source_paths

    def _build_single_source_admin_dataset(self, *, pbf_path: Path, output_path: Path) -> dict:
        build_admin_dataset(
            pbf_path=str(pbf_path),
            output_path=output_path,
            levels=self.LEVELS,
            profile=CN_PROFILE,
            fallback_policy=None,
            country_code=self.COUNTRY_ISO,
            country_name=self.COUNTRY_NAME,
            level_labels={
                4: "admin_region",
                5: "admin_district",
                6: "admin_municipality",
                7: "admin_locality",
                8: "admin_locality",
                9: "admin_locality",
                10: "admin_detail",
                11: "admin_unit",
                12: "admin_unit",
                14: "admin_unit",
            },
            id_prefix="cn",
            country_geometry_path=self._country_geometry_path,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))

    def _build_admin_dataset(self, *, osm_pbf_path: str) -> None:
        source_paths = self._source_input_paths(osm_pbf_path)
        if len(source_paths) == 1:
            self._build_single_source_admin_dataset(
                pbf_path=source_paths[0],
                output_path=self._admin_dataset_path,
            )
            return

        source_dir = self._work_dir / "_source_builds"
        source_dir.mkdir(parents=True, exist_ok=True)
        merged_payload: dict | None = None
        processing_time_sec = 0.0
        source_names: list[str] = []
        seen_ids: set[str] = set()

        for source_path in source_paths:
            payload = self._build_single_source_admin_dataset(
                pbf_path=source_path,
                output_path=source_dir / f"{source_path.stem}_admin.json",
            )
            source_names.append(source_path.name)
            meta = payload.get("meta")
            if isinstance(meta, dict):
                processing_time = meta.get("processing_time_sec")
                if isinstance(processing_time, (int, float)):
                    processing_time_sec += float(processing_time)
            if merged_payload is None:
                merged_payload = payload
                seen_ids = {
                    row["id"]
                    for rows in merged_payload.get("admin_by_level", {}).values()
                    if isinstance(rows, list)
                    for row in rows
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                }
                continue

            merged_by_level = merged_payload.setdefault("admin_by_level", {})
            for level in self.LEVELS:
                level_key = str(level)
                target_rows = merged_by_level.setdefault(level_key, [])
                for row in payload.get("admin_by_level", {}).get(level_key, []):
                    if not isinstance(row, dict):
                        continue
                    row_id = row.get("id")
                    if not isinstance(row_id, str) or row_id in seen_ids:
                        continue
                    target_rows.append(row)
                    seen_ids.add(row_id)

        if merged_payload is None:
            raise RuntimeError("No China source components were built")

        label_keys = {
            4: "admin_region",
            5: "admin_district",
            6: "admin_municipality",
            7: "admin_locality",
            8: "admin_locality",
            9: "admin_locality",
            10: "admin_detail",
            11: "admin_unit",
            12: "admin_unit",
            14: "admin_unit",
        }
        for key in sorted(set(label_keys.values())):
            merged_payload[key] = []
        for level, key in label_keys.items():
            merged_payload[key].extend(merged_payload.get("admin_by_level", {}).get(str(level), []))

        meta = merged_payload.setdefault("meta", {})
        meta["source"] = "OpenStreetMap composite"
        meta["source_components"] = source_names
        meta["generated_at"] = datetime.utcnow().isoformat()
        meta["processing_time_sec"] = processing_time_sec
        self._admin_dataset_path.write_text(
            json.dumps(merged_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _ensure_datasets(self, osm_pbf_path: str) -> None:
        if not self._admin_dataset_path.exists():
            self._build_admin_dataset(osm_pbf_path=osm_pbf_path)
        if not self._admin_hierarchy_path.exists():
            self._write_dataset_scoped_hierarchy_artifacts()
        if not self._ffsf_dataset_path.exists() or not self._ffsf_meta_path.exists():
            if not self._admin_dataset_path.exists():
                raise FileNotFoundError(f"Missing admin dataset required for FFSF export: {self._admin_dataset_path}")
            export_cadis_to_ffsf(input_path=self._admin_dataset_path, output_path=self._ffsf_dataset_path, version=3, country_geometry_path=self._country_geometry_path)
        if not self._semantic_dataset_path.exists():
            export_admin_semantic_dataset(nodes=self._build_semantic_nodes(), output_path=self._semantic_dataset_path, version="cn-admin-semantic-1.0.0", country=self.COUNTRY_ISO, source="admin_tree.txt")
        self._ensure_runtime_release_layers()

    def _runtime_policy_payload(self) -> dict:
        allowed_shapes = [list(shape) for shape in sorted(self.ALLOWED_SHAPES)]
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [4, 5, 6, 7, 8, 9, 10, 11, 12, 14],
            "allowed_shapes": allowed_shapes,
            "shape_status": [{"levels": shape, "status": "ok" if 4 in shape or 5 in shape or 6 in shape else "partial"} for shape in allowed_shapes],
            "layers": {"hierarchy_required": True, "repair_required": False},
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [level for level in [4, 5, 6, 7, 8, 9, 10, 11, 12, 14] if level != 4],
                "sub_engine_anchors": [{"source": "tibet", "parent_levels": [5, 6]}],
            },
            "repair_rules": {"parent_level": 4, "child_levels": []},
            "nearby_policy": {"enabled": True, "max_distance_km": 2.0, "offshore_max_distance_km": 20.0},
        }
