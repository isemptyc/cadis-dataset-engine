import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Optional

"""
Dataset Design Note: Polygon vs Hierarchy Consistency
----------------------------------------------------

This dataset pipeline intentionally produces *two complementary but not
guaranteed-consistent views* of administrative units:

1. Polygon dataset (geometry-driven)
2. Hierarchy dataset (relation-driven)

These two datasets MAY reference different OpenStreetMap relations for what
humans consider to be the "same" administrative unit (same name, same
admin_level). This is not a bug, but a direct consequence of OpenStreetMap's
data model and editing history.

Why this happens
----------------
In OpenStreetMap, administrative units are represented as relations.
However, OSM does NOT guarantee a one-to-one mapping between:

    (name, admin_level)  <->  relation ID

In practice, it is common for a single real-world administrative unit to be
represented by multiple relations, for example:

- an older relation with correct hierarchy semantics (subarea / is_in),
  but incomplete or invalid geometry
- a newer relation with valid multipolygon geometry,
  but weaker or missing hierarchical links

Pipeline consequences
---------------------
Because of this, the two extraction pipelines apply *different validity gates*:

- Polygon extraction (`build_admin_dataset`)
    * Requires valid OSM Area geometry
    * Discards relations that cannot form a multipolygon
    * Prioritizes geometric correctness

- Hierarchy extraction (`extract_admin_hierarchy`)
    * Operates on relations only (no geometry required)
    * Preserves subarea / is_in semantic structure
    * Prioritizes administrative relationships

As a result, polygon and hierarchy datasets may legitimately select
different OSM relations for the same named unit.

Design contract
---------------
This divergence is an expected and accepted outcome.

Therefore:

- Polygon evidence represents *geographic fact*
- Hierarchy data represents *administrative knowledge*

Hierarchy data MUST NOT be used to invalidate or discard polygon evidence
during lookup. It may only be used to supplement, interpret, or evaluate
semantic completeness at a later stage.

Any lookup engine that assumes polygon OSM IDs and hierarchy OSM IDs are
globally consistent is making an invalid assumption, particularly for regions
such as Taiwan.

This behavior is data-driven, not implementation-driven.
"""

def _pick_name(tags: dict, name_keys: Iterable[str]) -> Optional[str]:
    for key in name_keys:
        value = tags.get(key)
        if value:
            return value
    return None


def _should_apply_country_filter(pbf_path: str) -> bool:
    # Heuristic:
    # Geofabrik single-country dumps typically end with "-latest.osm.pbf".
    # Custom/combined regional extracts usually do not.
    return not Path(pbf_path).name.endswith("-latest.osm.pbf")

@dataclass(frozen=True)
class AdminLevelPolicy:
    simplify: bool
    simplify_tolerance: Optional[float]
    fix_invalid: bool
    parent_resolution: str  # "strict" | "relaxed"


@dataclass(frozen=True)
class AdminProfile:
    name_keys: tuple[str, ...]
    level_policies: dict[int, AdminLevelPolicy]
    parent_fallback: bool


def _resolve_level_policy(
    level: int,
    profile: AdminProfile,
    fallback_policy: Optional[AdminLevelPolicy],
):
    if level in profile.level_policies:
        return profile.level_policies[level], "level"
    if fallback_policy is not None:
        return fallback_policy, "fallback"
    return None, "none"


def _resolve_parent_resolution(
    level: int,
    profile: AdminProfile,
    fallback_parent_resolution: Optional[str],
) -> Optional[str]:
    if level in profile.level_policies:
        return profile.level_policies[level].parent_resolution
    if fallback_parent_resolution is not None:
        return fallback_parent_resolution
    return None


# ==================================================
# admin hierarchy (relations-only)
# ==================================================

def extract_admin_hierarchy(
    pbf_path: str,
    output_dir: Path,
    *,
    name_keys: Iterable[str] = ("name",),
    country_geometry_path: Optional[Path] = None,
):
    """
    Extract administrative hierarchy from OSM relations only.

    Notes
    -----
    This function intentionally ignores geometry validity and operates purely
    on relation semantics (admin_level, subarea, is_in tags).

    As a result, the relations selected here may differ from those selected
    by geometry-based polygon extraction.

    This is expected behavior and reflects the fact that OpenStreetMap may
    contain multiple relations for the same administrative unit with differing
    geometry quality and hierarchy completeness.
    """
    import hashlib
    import osmium

    apply_country_filter = (
        country_geometry_path is not None and _should_apply_country_filter(pbf_path)
    )

    if country_geometry_path is not None and not apply_country_filter:
        print("Country hierarchy filter skipped (single-country PBF heuristic).")

    def build_country_relation_allowlist():
        if not apply_country_filter:
            return None

        from shapely import wkb as shapely_wkb
        from shapely.geometry import shape
        from shapely.prepared import prep

        country_path = Path(country_geometry_path)
        if not country_path.exists():
            raise FileNotFoundError(f"country_geometry_path not found: {country_path}")

        country_raw = json.loads(country_path.read_text(encoding="utf-8"))
        if "geometry" not in country_raw:
            raise ValueError(f"Invalid country geometry file (missing geometry): {country_path}")

        country_geom = prep(shape(country_raw["geometry"]))

        try:
            pbf_stat = Path(pbf_path).stat()
            country_stat = country_path.stat()
            fingerprint_input = (
                f"{Path(pbf_path).resolve()}|{pbf_stat.st_size}|{pbf_stat.st_mtime_ns}|"
                f"{country_path.resolve()}|{country_stat.st_size}|{country_stat.st_mtime_ns}"
            )
        except FileNotFoundError:
            fingerprint_input = f"{Path(pbf_path)}|{country_path}"

        fingerprint = hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest()[:16]
        cache_path = output_dir / f".allowlist_{country_path.stem}_{fingerprint}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            allowlist = set(cached.get("allowlist_relation_ids", []))
            print(f"Country hierarchy allowlist cache hit: {cache_path.name} ({len(allowlist)} IDs)")
            return allowlist

        allowlist = set()
        wkbfab = osmium.geom.WKBFactory()
        fp = osmium.FileProcessor(pbf_path).with_locations().with_areas()

        for obj in fp:
            if not isinstance(obj, osmium.osm.Area):
                continue

            if obj.from_way():
                continue

            tags = dict(obj.tags)
            if tags.get("boundary") != "administrative":
                continue

            try:
                wkb = wkbfab.create_multipolygon(obj)
                poly = shapely_wkb.loads(wkb, hex=True)
            except Exception:
                continue

            rep = poly.representative_point()
            if not country_geom.covers(rep):
                continue

            rel_id = obj.orig_id() if hasattr(obj, "orig_id") else obj.id
            allowlist.add(f"r{rel_id}")

        cache_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.utcnow().isoformat(),
                    "pbf_path": str(Path(pbf_path)),
                    "country_geometry_path": str(country_path),
                    "allowlist_relation_ids": sorted(allowlist),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Country hierarchy filter enabled. Allowed relations: {len(allowlist)}")
        return allowlist

    class AdminRelationExtractor(osmium.SimpleHandler):
        def __init__(self, allowed_relation_ids=None):
            super().__init__()
            self.nodes = {}
            self.edges = []
            self.name_index = defaultdict(list)
            self.level_stats = defaultdict(int)
            self.allowed_relation_ids = allowed_relation_ids

        def relation(self, r):
            tags = dict(r.tags)

            if tags.get("boundary") != "administrative" and "admin_level" not in tags:
                return

            rid = f"r{r.id}"
            if self.allowed_relation_ids is not None and rid not in self.allowed_relation_ids:
                return

            admin_level = tags.get("admin_level")

            node = {
                "id": rid,
                "osm_id": r.id,
                "name": _pick_name(tags, name_keys),
                "admin_level": int(admin_level) if admin_level and admin_level.isdigit() else None,
                "tags": tags,
            }

            self.nodes[rid] = node

            if node["name"]:
                self.name_index[node["name"]].append(rid)

            if node["admin_level"] is not None:
                self.level_stats[node["admin_level"]] += 1

            for m in r.members:
                if m.type == "r" and m.role == "subarea":
                    self.edges.append({
                        "parent": rid,
                        "child": f"r{m.ref}",
                        "method": "subarea",
                        "confidence": 1.0,
                    })

            for k, v in tags.items():
                if k.startswith("is_in"):
                    self.edges.append({
                        "parent_name": v,
                        "child": rid,
                        "method": "is_in",
                        "confidence": 0.7,
                    })

    def resolve_is_in_edges(nodes, edges, name_index):
        resolved = []
        unresolved = []

        for e in edges:
            if e.get("method") != "is_in":
                resolved.append(e)
                continue

            candidates = name_index.get(e["parent_name"], [])
            if len(candidates) == 1:
                resolved.append({
                    "parent": candidates[0],
                    "child": e["child"],
                    "method": "is_in",
                    "confidence": e["confidence"],
                })
            else:
                unresolved.append(e)

        return resolved, unresolved

    print("Extracting administrative hierarchy (relation-only)...")

    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_relation_ids = build_country_relation_allowlist()

    handler = AdminRelationExtractor(allowed_relation_ids=allowed_relation_ids)
    handler.apply_file(pbf_path, locations=False)

    print(f"Extracted nodes: {len(handler.nodes)}")
    print(f"Raw edges: {len(handler.edges)}")

    resolved_edges, unresolved_edges = resolve_is_in_edges(
        handler.nodes, handler.edges, handler.name_index
    )

    if apply_country_filter:
        node_ids = set(handler.nodes.keys())
        resolved_edges = [
            e for e in resolved_edges
            if e.get("parent") in node_ids and e.get("child") in node_ids
        ]

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "country_geometry_filter_applied": apply_country_filter,
        "node_count": len(handler.nodes),
        "edge_count": len(resolved_edges),
        "unresolved_is_in_edges": len(unresolved_edges),
        "admin_level_distribution": dict(sorted(handler.level_stats.items())),
        "unresolved_samples": unresolved_edges[:20],
    }

    (output_dir / "admin_nodes.json").write_text(
        json.dumps(list(handler.nodes.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "admin_edges.json").write_text(
        json.dumps(resolved_edges, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "admin_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print("Output written to:", output_dir.resolve())

    return {
        "nodes": output_dir / "admin_nodes.json",
        "edges": output_dir / "admin_edges.json",
        "report": output_dir / "admin_report.json",
    }


# ==================================================
# admin tree renderer
# ==================================================

def render_admin_tree(
    nodes_path: Path,
    edges_path: Path,
    output_path: Path,
    *,
    indent: str = "  ",
):
    from collections import defaultdict

    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    edges = json.loads(edges_path.read_text(encoding="utf-8"))

    node_by_id = {n["id"]: n for n in nodes}

    children_by_parent = defaultdict(list)
    parent_by_child = {}

    for e in edges:
        parent = e.get("parent")
        child = e.get("child")
        if not parent or not child:
            continue

        children_by_parent[parent].append(child)
        if child not in parent_by_child:
            parent_by_child[child] = parent

    roots = [node_id for node_id in node_by_id if node_id not in parent_by_child]

    def format_node(node):
        parts = [node["name"] or "(unnamed)"]
        if node.get("admin_level") is not None:
            parts.append(f"[level={node['admin_level']}]")
        parts.append(f"[id={node['id']}]")
        return " ".join(parts)

    def render_tree(node_id, depth=0, visited=None, lines=None):
        if visited is None:
            visited = set()
        if lines is None:
            lines = []

        if node_id in visited:
            return lines

        visited.add(node_id)

        node = node_by_id.get(node_id)
        if not node:
            return lines

        lines.append(f"{indent * depth}- {format_node(node)}")

        children = sorted(
            children_by_parent.get(node_id, []),
            key=lambda cid: (node_by_id[cid].get("name") or "")
            if cid in node_by_id
            else "",
        )

        for child_id in children:
            render_tree(child_id, depth + 1, visited, lines)

        return lines

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    print(f"Roots detected: {len(roots)}")

    all_lines = []
    for root_id in sorted(roots, key=lambda rid: (node_by_id[rid].get("name") or "")):
        all_lines.extend(render_tree(root_id))
        all_lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(all_lines), encoding="utf-8")

    print("Admin tree written to:")
    print(output_path.resolve())

    return output_path


def build_country_from_ne(
    dbf_path: Path,
    output_path: Path,
    *,
    country_iso: Optional[str] = None,
    country_name: Optional[str] = None,
):
    """
    Build a country boundary JSON from Natural Earth admin-0 data.

    This is intended for combined/multi-country OSM inputs where the build
    needs an explicit country polygon gate (for example SG/MY/BN regional data).
    For single-country OSM extracts, this step is typically unnecessary.
    """
    import geopandas as gpd
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    output_path = Path(output_path)
    print(f"Loading Natural Earth DBF: {dbf_path}")
    gdf = gpd.read_file(dbf_path)

    filters = []
    if country_iso:
        filters.append(gdf["ISO_A2"] == country_iso)
    if country_name:
        filters.append(gdf["ADMIN"] == country_name)

    if filters:
        match = filters[0]
        for f in filters[1:]:
            match = match | f
        filtered = gdf[match]
    else:
        filtered = gdf

    if filtered.empty:
        raise RuntimeError("Country not found in Natural Earth DBF")

    geom = unary_union(filtered.geometry)

    if filtered.crs is None:
        filtered = filtered.set_crs(epsg=4326)
    elif filtered.crs.to_epsg() != 4326:
        geom = gpd.GeoSeries([geom], crs=filtered.crs).to_crs(epsg=4326).iloc[0]

    minx, miny, maxx, maxy = geom.bounds
    bbox = [minx, miny, maxx, maxy]

    output = {
        "id": country_iso,
        "name": country_name,
        "bbox": bbox,
        "geometry": mapping(geom),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Country polygon written to: {output_path}")
    print(f"bbox = {bbox}")

import osmium
from shapely import wkb as wkb_loader

# IMPORTANT:
# Identity is relation-driven.
# Only relation-based administrative boundaries are used.
# Way-based boundaries are ignored to ensure stable entity identity across rebuilds.

class AdminExtractor(osmium.SimpleHandler):
    def __init__(self, target_levels, profile):
        super().__init__()
        self.target_levels = set(target_levels)
        self.profile = profile
        self.data = []
        self.wkbfab = osmium.geom.WKBFactory()

    def area(self, a):
        # ---- boundary filter ----
        if a.tags.get("boundary") != "administrative":
            return

        # ---- ignore way-based areas (identity instability) ----
        if a.from_way():
            return

        # ---- level filter ----
        level_tag = a.tags.get("admin_level")
        if not level_tag or not level_tag.isdigit():
            return

        lvl = int(level_tag)
        if lvl not in self.target_levels:
            return

        try:
            # geometry from assembled multipolygon
            wkb = self.wkbfab.create_multipolygon(a)

            # IMPORTANT:
            # Use ORIGINAL RELATION ID (stable across rebuilds)
            rel_id = a.orig_id()

            self.data.append({
                "osm_id": f"r{rel_id}",
                "level": lvl,
                "name": _pick_name(dict(a.tags), self.profile.name_keys),
                "wkb": wkb
            })

        except Exception:
            # geometry occasionally fails on broken multipolygons
            pass

import pandas as pd
import geopandas as gpd

class TopologyEngine:
    @staticmethod
    def build_gdf(raw_data):
        df = pd.DataFrame(raw_data)
        # 修正修正：loads(bytes) 確保相容性
        df['geometry'] = df['wkb'].apply(lambda x: wkb_loader.loads(x))
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        return gdf.drop(columns=['wkb'])

    @staticmethod
    def infer_hierarchy(gdf, levels):
        levels = sorted(levels)
        # 預計算代表點，增加 Robustness (處理島嶼/洞)
        gdf['rep_point'] = gdf.geometry.representative_point()
        gdf['parent'] = None
        
        # 建立空間索引 (SJoin 底層使用 STRtree)
        # 
        for i in range(len(levels) - 1):
            p_lvl, c_lvl = levels[i], levels[i+1]
            parents = gdf[gdf.level == p_lvl][['id', 'geometry']]
            children = gdf[gdf.level == c_lvl][['id', 'rep_point']]
            
            c_pts = gpd.GeoDataFrame(children, geometry='rep_point', crs="EPSG:4326")
            joined = gpd.sjoin(c_pts, parents, predicate='within', how='left')
            
            # 修正修正：精確 Mapping 避免污染
            parent_map = joined.set_index('id_left')['id_right'].to_dict()
            mask = gdf.level == c_lvl
            gdf.loc[mask, 'parent'] = gdf.loc[mask, 'id'].map(parent_map)
        
        return gdf
    
def serialize_output(gdf, levels, profile, meta_info):
    output = {
        "meta": meta_info,
        "admin_by_level": {}
    }
    
    # 預計算 BBOX
    gdf['bbox'] = gdf.geometry.bounds.apply(lambda b: [b.minx, b.miny, b.maxx, b.maxy], axis=1)

    for lvl in levels:
        lvl_df = gdf[gdf.level == lvl]
        items = []
        for _, row in lvl_df.iterrows():
            item = {
                "id": row['id'],
                "osm_id": row['osm_id'],
                "name": row['name'],
                "level": lvl,
                "bbox": row['bbox'],
            }
            # 性能修正：根據 profile 決定是否匯出龐大的幾何數據
            if getattr(profile, 'export_geometry', True):
                item["geometry"] = row['geometry'].__geo_interface__
            
            if row['parent'] or profile.parent_fallback:
                item["parent"] = row['parent']
            items.append(item)
            
        output["admin_by_level"][str(lvl)] = items
    return output

#build_admin_dataset_v2
def build_admin_dataset(
    pbf_path: str,
    output_path: Path,
    *,
    levels: Iterable[int],
    profile: AdminProfile,
    fallback_policy: Optional[AdminLevelPolicy] = None, #backward compatible only, useless
    fallback_parent_resolution: Optional[str] = None, #backward compatible only, useless
    country_code: Optional[str] = None,
    country_name: Optional[str] = None,
    country_geometry_path: Optional[Path] = None,
    level_labels: Optional[dict[int, str]] = None,
    id_prefix: Optional[str] = None,
):
    """
    Build an administrative boundary dataset from OSM PBF input using an explicit
    AdminProfile to drive all semantic decisions.

    Semantic notes (important):

    - Fallback behavior is NOT a default.
        * Fallbacks are applied only when explicitly provided.
        * Absence of a level-specific policy with no fallback means an explicit no-op.

    - Fallback behavior is NOT auto-fix.
        * The pipeline will not attempt to "correct" geometry or hierarchy unless
          explicitly instructed by the profile or fallback settings.

    - Parent-child resolution semantics are independent from geometry processing.
        * A level may participate in parent resolution even if no geometry operations
          are applied.

    - Unresolved parents are valid outcomes.
        * If `profile.parent_fallback` is False, the `parent` key is omitted.
        * If `profile.parent_fallback` is True, unresolved parents are represented
          explicitly as `null`.

    This function must not infer caller intent. All non-trivial behavior must be
    traceable to the provided AdminProfile or explicit fallback parameters.
    """        
    import numpy as np
    """
    Entry Point: 協調整個 ETL 流程
    """
    start_time = datetime.now()
    levels = sorted(list(set(levels)))
    output_path = Path(output_path)

    # --- Step 1: Extraction (OSM PBF -> Raw Data) ---
    print(f"[{datetime.now()}] 階段 1: 開始從 PBF 提取 WKB 數據...")
    extractor = AdminExtractor(target_levels=levels, profile=profile)
    extractor.apply_file(pbf_path, locations=True)
    
    if not extractor.data:
        print("錯誤: 未能從 PBF 提取到任何行政邊界數據。")
        return None

    # --- Step 2: Topology & Geometry Processing (Raw Data -> GeoDataFrame) ---
    print(f"[{datetime.now()}] 階段 2: 建立地理空間索引與拓撲分析...")
    engine = TopologyEngine()
    gdf = engine.build_gdf(extractor.data)
    
    # 釋放不再需要的原始數據以節省 RAM
    del extractor.data 

    apply_country_filter = (
        country_geometry_path is not None and _should_apply_country_filter(pbf_path)
    )
    if country_geometry_path is not None and not apply_country_filter:
        print("Country geometry filter skipped (single-country PBF heuristic).")

    # 可選：按國界幾何過濾，支援「同一份多國 PBF 分別建檔」
    if apply_country_filter:
        from shapely.geometry import shape
        from shapely.prepared import prep

        country_geometry_path = Path(country_geometry_path)
        if not country_geometry_path.exists():
            raise FileNotFoundError(f"country_geometry_path not found: {country_geometry_path}")

        country_raw = json.loads(country_geometry_path.read_text(encoding="utf-8"))
        if "geometry" not in country_raw:
            raise ValueError(f"Invalid country geometry file (missing geometry): {country_geometry_path}")

        country_geom = prep(shape(country_raw["geometry"]))
        gdf["_country_rep_point"] = gdf.geometry.representative_point()
        gdf = gdf[gdf["_country_rep_point"].apply(country_geom.covers)].copy()
        gdf = gdf.drop(columns=["_country_rep_point"])

        if gdf.empty:
            print("錯誤: 國界過濾後沒有任何行政邊界。請檢查 pbf 與 country_geometry_path 是否匹配。")
            return None

    # 預生成 ID 前綴
    if id_prefix is None:
        id_prefix = (country_code or "adm").lower()
    gdf['id'] = f"{id_prefix}_" + gdf['osm_id']

    # 執行幾何優化 (Simplify / Fix Invalid)
    # 使用 Feedback 建議的 Groupby 優化方式
    for lvl, group in gdf.groupby('level'):
        policy, _ = _resolve_level_policy(lvl, profile, None) # 你的策略解析函數
        if policy:
            mask = gdf.level == lvl
            if policy.simplify:
                gdf.loc[mask, 'geometry'] = gdf.loc[mask, 'geometry'].simplify(
                    policy.simplify_tolerance, preserve_topology=True
                )
            if policy.fix_invalid:
                invalid_mask = mask & (~gdf.geometry.is_valid)
                gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)

    # 計算行政層級父子關係
    gdf = engine.infer_hierarchy(gdf, levels)

    # --- Step 3: Serialization (GeoDataFrame -> JSON) ---
    print(f"[{datetime.now()}] 階段 3: 序列化數據並寫入磁碟...")
    meta_info = {
        "country": country_code,
        "country_name": country_name,
        "country_geometry_filter_applied": apply_country_filter,
        "levels": levels,
        "source": "OpenStreetMap",
        "generated_at": datetime.utcnow().isoformat(),
        "processing_time_sec": (datetime.now() - start_time).total_seconds()
    }

    gdf = gdf.replace({np.nan: None, pd.NA: None})
    final_json = serialize_output(gdf, levels, profile, meta_info)
    
    # 如果有 Label 對照表，加入額外 Key
    if level_labels:
        for lvl, label in level_labels.items():
            if str(lvl) in final_json["admin_by_level"]:
                final_json[label] = final_json["admin_by_level"][str(lvl)]

    # 執行寫入
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[{datetime.now()}] 任務完成！總耗時: {datetime.now() - start_time}")
    return output_path
