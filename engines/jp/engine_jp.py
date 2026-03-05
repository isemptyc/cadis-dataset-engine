from pathlib import Path
import json
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
├── Geometry runtime layer     (geometry.ffsf + geometry_meta.json)
└── Runtime hierarchy layer    (hierarchy.json)
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
    MODERN_REGION_NAMES = {
        "北海道地方",
        "東北地方",
        "関東地方",
        "中部地方",
        "近畿地方",
        "中国地方",
        "四国地方",
        "九州地方",
    }

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
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

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
            self._runtime_hierarchy_path,
        ):
            if p.exists():
                paths.append(p)
        return paths

    def _write_dataset_build_manifest(self) -> Path:
        self._ensure_runtime_release_layers()
        return super()._write_dataset_build_manifest()

    def _load_feature_meta_rows(self) -> list[dict]:
        payload = json.loads(self._ffsf_meta_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("features", [])
            if isinstance(rows, dict):
                rows = list(rows.values())
        else:
            rows = []
        return [r for r in rows if isinstance(r, dict)]

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

        feature_rows = self._load_feature_meta_rows()
        level4_name_to_id = {
            r["name"]: r["feature_id"]
            for r in feature_rows
            if r.get("level") == 4 and isinstance(r.get("name"), str) and isinstance(r.get("feature_id"), str)
        }
        level3_name_to_id = {
            r["name"]: r["feature_id"]
            for r in feature_rows
            if r.get("level") == 3 and isinstance(r.get("name"), str) and isinstance(r.get("feature_id"), str)
        }

        hierarchy_nodes = self._load_admin_hierarchy(self._admin_hierarchy_path)
        node_by_id = {n["id"]: n for n in hierarchy_nodes}
        pref_to_region: dict[str, str] = {}
        for n in hierarchy_nodes:
            if n.get("level") != 4:
                continue
            pref_name = n.get("name")
            if pref_name not in level4_name_to_id:
                continue
            parent = node_by_id.get(n.get("parent_id"))
            if parent is None or parent.get("level") != 3:
                continue
            region_name = parent.get("name")
            if region_name not in self.MODERN_REGION_NAMES:
                continue
            existing = pref_to_region.get(pref_name)
            if existing is None:
                pref_to_region[pref_name] = region_name
            elif existing != region_name:
                # Deterministic tie-breaker for malformed hierarchy: keep lexicographically smaller region.
                pref_to_region[pref_name] = min(existing, region_name)

        region_names = sorted(set(pref_to_region.values()))
        region_name_to_runtime_id: dict[str, str] = {}
        for idx, region_name in enumerate(region_names, start=1):
            region_name_to_runtime_id[region_name] = level3_name_to_id.get(
                region_name,
                f"jp_region_{idx:02d}",
            )

        runtime_nodes = []
        for region_name in region_names:
            runtime_nodes.append(
                {
                    "id": region_name_to_runtime_id[region_name],
                    "level": 3,
                    "name": region_name,
                    "parent_id": None,
                }
            )

        for pref_name in sorted(pref_to_region.keys()):
            runtime_nodes.append(
                {
                    "id": level4_name_to_id[pref_name],
                    "level": 4,
                    "name": pref_name,
                    "parent_id": region_name_to_runtime_id[pref_to_region[pref_name]],
                }
            )

        self._runtime_hierarchy_path.write_text(
            json.dumps({"nodes": runtime_nodes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 3,
                "child_levels": [4],
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
