from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from pathlib import Path

from base import DatasetBuildEngineBase
from dataset import (
    AdminLevelPolicy,
    AdminProfile,
    build_admin_dataset,
    render_admin_tree,
)
from ffsf import export_cadis_to_ffsf
from ffsf.semantic_dataset_exporter import export_admin_semantic_dataset
from engines.us.stitch_admin_dataset import build_stitched_admin_dataset

DEFAULT_WORK_DIR = Path.home() / ".cache" / "cadis_dataset_engine" / "united_states"

US_REGIONS = (
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "district-of-columbia",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new-hampshire",
    "new-jersey",
    "new-mexico",
    "new-york",
    "north-carolina",
    "north-dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "puerto-rico",
    "rhode-island",
    "south-carolina",
    "south-dakota",
    "tennessee",
    "texas",
    "us-virgin-islands",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west-virginia",
    "wisconsin",
    "wyoming",
)

US_REGION_NAMES = {
    "alabama": "Alabama",
    "alaska": "Alaska",
    "arizona": "Arizona",
    "arkansas": "Arkansas",
    "california": "California",
    "colorado": "Colorado",
    "connecticut": "Connecticut",
    "delaware": "Delaware",
    "district-of-columbia": "District of Columbia",
    "florida": "Florida",
    "georgia": "Georgia",
    "hawaii": "Hawaii",
    "idaho": "Idaho",
    "illinois": "Illinois",
    "indiana": "Indiana",
    "iowa": "Iowa",
    "kansas": "Kansas",
    "kentucky": "Kentucky",
    "louisiana": "Louisiana",
    "maine": "Maine",
    "maryland": "Maryland",
    "massachusetts": "Massachusetts",
    "michigan": "Michigan",
    "minnesota": "Minnesota",
    "mississippi": "Mississippi",
    "missouri": "Missouri",
    "montana": "Montana",
    "nebraska": "Nebraska",
    "nevada": "Nevada",
    "new-hampshire": "New Hampshire",
    "new-jersey": "New Jersey",
    "new-mexico": "New Mexico",
    "new-york": "New York",
    "north-carolina": "North Carolina",
    "north-dakota": "North Dakota",
    "ohio": "Ohio",
    "oklahoma": "Oklahoma",
    "oregon": "Oregon",
    "pennsylvania": "Pennsylvania",
    "puerto-rico": "Puerto Rico",
    "rhode-island": "Rhode Island",
    "south-carolina": "South Carolina",
    "south-dakota": "South Dakota",
    "tennessee": "Tennessee",
    "texas": "Texas",
    "us-virgin-islands": "United States Virgin Islands",
    "utah": "Utah",
    "vermont": "Vermont",
    "virginia": "Virginia",
    "washington": "Washington",
    "west-virginia": "West Virginia",
    "wisconsin": "Wisconsin",
    "wyoming": "Wyoming",
}

US_PROFILE = AdminProfile(
    name_keys=("name:en", "name", "official_name"),
    level_policies={
        4: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.01,
            fix_invalid=True,
            parent_resolution="strict",
        ),
        6: AdminLevelPolicy(
            simplify=True,
            simplify_tolerance=0.003,
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
    multilingual_allowed_languages=("en", "es", "fr"),
)


class UnitedStatesAdminEngine(DatasetBuildEngineBase):
    ENGINE = "us_admin"
    VERSION = "v0.1"
    NAME_SCHEMA = "multilingual_v1"

    LEVELS = [4, 6, 8]
    ALLOWED_SHAPES = {
        (4,),
        (4, 6),
        (4, 6, 8),
        (4, 8),
        (6,),
        (6, 8),
        (8,),
    }

    COUNTRY_ISO = "US"
    COUNTRY_NAME = "United States of America"
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

        self._intermediate_dir = self._work_dir / "state_builds"
        self._admin_dataset_path = self._work_dir / "united_states_admin.json"
        self._stitch_report_path = self._work_dir / "us_stitch_report.json"
        self._ffsf_dataset_path = self._work_dir / "united_states_admin.bin"
        self._ffsf_meta_path = self._work_dir / "US_feature_meta_by_index.json"
        self._semantic_dataset_path = self._work_dir / "united_states_admin_semantic.json"
        self._admin_hierarchy_path = self._work_dir / "admin_tree.txt"
        self._runtime_geometry_path = self._work_dir / "geometry.ffsf"
        self._runtime_geometry_meta_path = self._work_dir / "geometry_meta.json"
        self._runtime_hierarchy_path = self._work_dir / "hierarchy.json"

        if osm_pbf_path is None:
            raise ValueError(
                "UnitedStatesAdminEngine in cadis-dataset-engine is build-only. "
                "Provide osm_pbf_path as a directory of state PBF extracts or one PBF file."
            )
        self._ensure_datasets(osm_pbf_path=Path(osm_pbf_path))

    def _selected_pbf_paths(self, osm_pbf_path: Path) -> list[Path]:
        if osm_pbf_path.is_file():
            return [osm_pbf_path]
        if not osm_pbf_path.is_dir():
            raise FileNotFoundError(f"US OSM input must be a PBF file or directory: {osm_pbf_path}")

        available = {path.name: path for path in osm_pbf_path.glob("*-latest.osm.pbf")}
        selected: list[Path] = []
        missing: list[str] = []
        for region in US_REGIONS:
            name = f"{region}-latest.osm.pbf"
            path = available.get(name)
            if path is None:
                missing.append(name)
            else:
                selected.append(path)
        if missing:
            raise FileNotFoundError(
                "US build missing required state extracts: " + ", ".join(missing)
            )
        return selected

    def _ensure_datasets(self, osm_pbf_path: Path) -> None:
        pbf_paths = self._selected_pbf_paths(osm_pbf_path)

        if not self._admin_dataset_path.exists():
            if osm_pbf_path.is_dir():
                stitch_cache_dir = osm_pbf_path / "_stitch_cache"
                build_stitched_admin_dataset(
                    pbf_paths=pbf_paths,
                    output_path=self._admin_dataset_path,
                    report_path=self._stitch_report_path,
                    levels=self.LEVELS,
                    profile=US_PROFILE,
                    country_code=self.COUNTRY_ISO,
                    country_name=self.COUNTRY_NAME,
                    level_labels={
                        4: "admin_state",
                        6: "admin_county",
                        8: "admin_municipality",
                    },
                    id_prefix="us",
                    cache_dir=stitch_cache_dir,
                    allowed_level4_names=set(US_REGION_NAMES.values()),
                )
            else:
                build_admin_dataset(
                    pbf_path=str(pbf_paths[0]),
                    output_path=self._admin_dataset_path,
                    levels=self.LEVELS,
                    profile=US_PROFILE,
                    fallback_policy=None,
                    country_code=self.COUNTRY_ISO,
                    country_name=self.COUNTRY_NAME,
                    level_labels={
                        4: "admin_state",
                        6: "admin_county",
                        8: "admin_municipality",
                    },
                    id_prefix="us",
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
                version="us-admin-semantic-0.1.0",
                country=self.COUNTRY_ISO,
                source="admin_tree.txt",
            )

        self._ensure_runtime_release_layers()

    def _ensure_state_level_payload(self, region: str, payload: dict) -> dict:
        admin_by_level = payload.get("admin_by_level")
        if not isinstance(admin_by_level, dict):
            return payload

        state_name = US_REGION_NAMES.get(region)
        if not state_name:
            return payload

        level4_rows = admin_by_level.setdefault("4", [])
        if not isinstance(level4_rows, list):
            return payload
        if any(isinstance(row, dict) and row.get("name") == state_name for row in level4_rows):
            return payload

        level6_rows = admin_by_level.get("6", [])
        if not isinstance(level6_rows, list) or not level6_rows:
            return payload

        from shapely.geometry import mapping, shape
        from shapely.ops import unary_union

        county_geometries = []
        for row in level6_rows:
            if not isinstance(row, dict):
                continue
            geom = row.get("geometry")
            if not isinstance(geom, dict):
                continue
            county_geometries.append(shape(geom))
        if not county_geometries:
            return payload

        state_id = f"us_s_{region.replace('-', '_')}"
        state_geom = unary_union(county_geometries)
        level4_rows.append(
            {
                "id": state_id,
                "osm_id": state_id,
                "level": 4,
                "name": state_name,
                "geometry": mapping(state_geom),
            }
        )

        for row in level6_rows:
            if not isinstance(row, dict):
                continue
            if not row.get("parent"):
                row["parent"] = state_id

        for label in ("admin_state", "admin_county"):
            rows = payload.get(label)
            if label == "admin_state":
                payload[label] = level4_rows
            elif label == "admin_county" and isinstance(rows, list):
                payload[label] = level6_rows
        return payload

    def _write_merged_admin_dataset(self, state_payloads: list[tuple[str, dict]]) -> None:
        merged_by_level: dict[str, list[dict]] = {str(level): [] for level in self.LEVELS}
        seen_ids: set[str] = set()
        source_regions: list[str] = []

        for region, payload in state_payloads:
            source_regions.append(region)
            admin_by_level = payload.get("admin_by_level", {})
            if not isinstance(admin_by_level, dict):
                continue
            for level in self.LEVELS:
                rows = admin_by_level.get(str(level), [])
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    feature_id = row.get("id")
                    if not isinstance(feature_id, str) or not feature_id:
                        continue
                    if feature_id in seen_ids:
                        continue
                    seen_ids.add(feature_id)
                    merged_by_level[str(level)].append(row)

        for level_key, rows in merged_by_level.items():
            rows.sort(key=lambda row: (str(row.get("name") or ""), str(row.get("id") or "")))

        payload = {
            "meta": {
                "country": self.COUNTRY_ISO,
                "country_name": self.COUNTRY_NAME,
                "country_geometry_filter_applied": self._country_geometry_path is not None,
                "levels": self.LEVELS,
                "source": "OpenStreetMap Geofabrik US state extracts",
                "source_regions": source_regions,
                "state_extract_count": len(source_regions),
                "generated_at": datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
            },
            "admin_by_level": merged_by_level,
        }
        payload["admin_state"] = merged_by_level["4"]
        payload["admin_county"] = merged_by_level["6"]
        payload["admin_municipality"] = merged_by_level["8"]

        self._admin_dataset_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
        manifest_path = super()._write_dataset_build_manifest()
        if self._stitch_report_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            build_evidence = payload.setdefault("build_evidence", {})
            build_evidence["us_stitch_report"] = {
                "path": self._stitch_report_path.relative_to(self._work_dir).as_posix(),
                "sha256": self._sha256_file(self._stitch_report_path),
                "size": self._stitch_report_path.stat().st_size,
            }
            try:
                stitch_report = json.loads(self._stitch_report_path.read_text(encoding="utf-8"))
            except Exception:
                stitch_report = {}
            if isinstance(stitch_report, dict):
                for key in (
                    "relation_count",
                    "required_way_count",
                    "present_way_count",
                    "missing_way_count",
                    "required_node_count",
                    "present_node_count",
                    "missing_node_count",
                    "assembled_count",
                    "failure_count",
                    "failure_counts",
                    "scope_filter",
                    "cache",
                ):
                    if key in stitch_report:
                        build_evidence[f"us_stitch_{key}"] = stitch_report[key]
            manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return manifest_path

    def _normalize_runtime_feature_id(self, feature_id: str | None) -> str | None:
        if feature_id is None:
            return None
        prefix = f"{self.COUNTRY_ISO.lower()}_"
        if feature_id.startswith(prefix):
            return feature_id[len(prefix):]
        return feature_id

    def _load_admin_dataset_nodes(self) -> list[dict]:
        payload = json.loads(self._admin_dataset_path.read_text(encoding="utf-8"))
        admin_by_level = payload.get("admin_by_level", {})
        if not isinstance(admin_by_level, dict):
            return []

        nodes: list[dict] = []
        for level in self.LEVELS:
            rows = admin_by_level.get(str(level), [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                feature_id = row.get("id")
                name = row.get("name")
                if not isinstance(feature_id, str) or not isinstance(name, str) or not name:
                    continue
                parent_id = row.get("parent")
                if parent_id is not None and not isinstance(parent_id, str):
                    parent_id = None
                nodes.append(
                    {
                        "id": self._normalize_runtime_feature_id(feature_id),
                        "level": level,
                        "name": name,
                        "names": row.get("names") if isinstance(row.get("names"), dict) else None,
                        "parent_id": self._normalize_runtime_feature_id(parent_id),
                    }
                )
        return nodes

    def _write_dataset_scoped_hierarchy_artifacts(self) -> None:
        dataset_nodes = self._load_admin_dataset_nodes()
        level_counts: dict[int, int] = {}
        hierarchy_nodes: list[dict] = []
        hierarchy_edges: list[dict] = []

        for node in dataset_nodes:
            node_id = node["id"]
            raw_id = f"{self.COUNTRY_ISO.lower()}_{node_id}"
            level = node["level"]
            parent_id = node.get("parent_id")
            raw_parent_id = None
            if isinstance(parent_id, str) and parent_id:
                raw_parent_id = f"{self.COUNTRY_ISO.lower()}_{parent_id}"

            hierarchy_nodes.append(
                {
                    "id": raw_id,
                    "osm_id": raw_id[3:] if raw_id.startswith(f"{self.COUNTRY_ISO.lower()}_") else raw_id,
                    "name": node["name"],
                    "names": node.get("names"),
                    "admin_level": level,
                    "tags": {
                        "boundary": "administrative",
                        "admin_level": str(level),
                    },
                }
            )
            level_counts[level] = level_counts.get(level, 0) + 1

            if raw_parent_id is not None:
                hierarchy_edges.append(
                    {
                        "parent": raw_parent_id,
                        "child": raw_id,
                        "method": "dataset_parent",
                        "confidence": 1.0,
                    }
                )

        nodes_path = self._work_dir / "admin_nodes.json"
        edges_path = self._work_dir / "admin_edges.json"
        report_path = self._work_dir / "admin_report.json"

        nodes_path.write_text(
            json.dumps(hierarchy_nodes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        edges_path.write_text(
            json.dumps(hierarchy_edges, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "country_geometry_filter_applied": self._country_geometry_path is not None,
                    "dataset_scope_projection_applied": True,
                    "state_extract_count": len(US_REGIONS),
                    "node_count": len(hierarchy_nodes),
                    "edge_count": len(hierarchy_edges),
                    "unresolved_is_in_edges": 0,
                    "admin_level_distribution": {
                        str(level): count for level, count in sorted(level_counts.items())
                    },
                    "unresolved_samples": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        render_admin_tree(
            nodes_path=nodes_path,
            edges_path=edges_path,
            output_path=self._admin_hierarchy_path,
        )

    def _load_tree_parent_candidates(self) -> dict[str, list[dict]]:
        hierarchy_nodes = self._load_admin_hierarchy(self._admin_hierarchy_path)
        candidates: dict[str, list[dict]] = {}
        for node in hierarchy_nodes:
            if node["level"] not in self.LEVELS:
                continue
            node_id = self._normalize_runtime_feature_id(node["id"])
            parent_id = self._normalize_runtime_feature_id(node.get("parent_id"))
            candidates.setdefault(node_id, []).append(
                {
                    "level": node["level"],
                    "parent_id": parent_id,
                }
            )
        return candidates

    def _build_semantic_nodes(self) -> list[dict]:
        dataset_nodes = self._load_admin_dataset_nodes()
        node_by_id = {node["id"]: node for node in dataset_nodes}
        node_ids = set(node_by_id.keys())
        tree_candidates = self._load_tree_parent_candidates()

        supplemented_parents: dict[str, str | None] = {}
        for node in dataset_nodes:
            parent_id = node.get("parent_id")
            if parent_id in node_ids:
                supplemented_parents[node["id"]] = parent_id
                continue

            child_level = node["level"]
            valid_candidates: list[tuple[int, str]] = []
            for candidate in tree_candidates.get(node["id"], []):
                candidate_parent = candidate.get("parent_id")
                if candidate_parent not in node_ids:
                    continue
                parent_node = node_by_id.get(candidate_parent)
                if parent_node is None or parent_node["level"] >= child_level:
                    continue
                valid_candidates.append((parent_node["level"], candidate_parent))

            if valid_candidates:
                valid_candidates.sort(key=lambda item: (-item[0], item[1]))
                supplemented_parents[node["id"]] = valid_candidates[0][1]
            else:
                supplemented_parents[node["id"]] = None

        semantic_nodes: list[dict] = []
        for node in sorted(dataset_nodes, key=lambda n: (n["level"], n["name"], n["id"])):
            semantic_nodes.append(
                {
                    "feature_id": node["id"],
                    "level": node["level"],
                    "name": node["name"],
                    "names": node.get("names") if isinstance(node.get("names"), dict) else None,
                    "parent_id": supplemented_parents[node["id"]],
                }
            )
        return semantic_nodes

    def _load_semantic_nodes(self) -> list[dict]:
        payload = json.loads(self._semantic_dataset_path.read_text(encoding="utf-8"))
        nodes_raw = payload.get("nodes", {})
        out = []
        if not isinstance(nodes_raw, dict):
            return out
        for feature_id, row in nodes_raw.items():
            if not isinstance(row, list) or len(row) not in {3, 4}:
                continue
            if len(row) == 3:
                level, name, parent_id = row
                names = None
            else:
                level, name, parent_id, names = row
            if not isinstance(level, int):
                continue
            if not isinstance(name, str) or not name:
                continue
            if parent_id is not None and not isinstance(parent_id, str):
                continue
            if names is not None and not isinstance(names, dict):
                continue
            out.append(
                {
                    "id": feature_id,
                    "level": level,
                    "name": name,
                    "names": names if isinstance(names, dict) and names else None,
                    "parent_id": parent_id,
                }
            )
        return out

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

        semantic_nodes = self._load_semantic_nodes()
        self._runtime_hierarchy_path.write_text(
            json.dumps({"nodes": semantic_nodes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _runtime_policy_payload(self) -> dict:
        return {
            "runtime_policy_version": self.RUNTIME_POLICY_VERSION,
            "allowed_levels": [4, 6, 8],
            "allowed_shapes": [
                [4],
                [4, 6],
                [4, 6, 8],
                [4, 8],
                [6],
                [6, 8],
                [8],
            ],
            "shape_status": [
                {"levels": [4], "status": "partial"},
                {"levels": [4, 6], "status": "ok"},
                {"levels": [4, 6, 8], "status": "ok"},
                {"levels": [4, 8], "status": "ok"},
                {"levels": [6], "status": "partial"},
                {"levels": [6, 8], "status": "partial"},
                {"levels": [8], "status": "partial"},
            ],
            "layers": {
                "hierarchy_required": True,
                "repair_required": False,
            },
            "hierarchy_repair_rules": {
                "parent_level": 4,
                "child_levels": [6, 8],
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
