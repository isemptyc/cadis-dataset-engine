from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import osmium

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.us.engine_us import US_REGIONS


class TargetRelationScanner(osmium.SimpleHandler):
    def __init__(self, target_relation_ids: set[int]):
        super().__init__()
        self.target_relation_ids = target_relation_ids
        self.relations: dict[int, dict] = {}

    def relation(self, relation):
        if relation.id not in self.target_relation_ids:
            return
        tags = dict(relation.tags)
        way_members = []
        other_members = []
        for member in relation.members:
            row = {
                "type": member.type,
                "ref": int(member.ref),
                "role": str(member.role or ""),
            }
            if member.type == "w":
                way_members.append(row)
            else:
                other_members.append(row)
        self.relations[int(relation.id)] = {
            "id": int(relation.id),
            "tags": tags,
            "way_members": way_members,
            "other_members": other_members,
        }


class RequiredWayScanner(osmium.SimpleHandler):
    def __init__(self, required_way_ids: set[int]):
        super().__init__()
        self.required_way_ids = required_way_ids
        self.ways: dict[int, dict] = {}

    def way(self, way):
        if way.id not in self.required_way_ids:
            return
        node_refs = [int(node.ref) for node in way.nodes]
        self.ways[int(way.id)] = {
            "id": int(way.id),
            "node_refs": node_refs,
            "node_count": len(node_refs),
            "closed": bool(node_refs) and node_refs[0] == node_refs[-1],
            "tags": dict(way.tags),
        }


class RequiredNodeScanner(osmium.SimpleHandler):
    def __init__(self, required_node_ids: set[int]):
        super().__init__()
        self.required_node_ids = required_node_ids
        self.nodes: set[int] = set()

    def node(self, node):
        if node.id in self.required_node_ids:
            self.nodes.add(int(node.id))


def _parse_relation_ids(raw: str) -> set[int]:
    out = {int(part.strip().removeprefix("r")) for part in raw.split(",") if part.strip()}
    if not out:
        raise ValueError("--relation-ids must include at least one relation id")
    return out


def _scan_target_relations(pbf_paths: list[Path], target_relation_ids: set[int]) -> dict:
    by_relation: dict[int, dict] = {}
    relation_regions: dict[int, list[str]] = defaultdict(list)
    for path in pbf_paths:
        scanner = TargetRelationScanner(target_relation_ids)
        scanner.apply_file(str(path), locations=False)
        region = path.name.removesuffix("-latest.osm.pbf")
        for relation_id, row in scanner.relations.items():
            relation_regions[relation_id].append(region)
            existing = by_relation.get(relation_id)
            if existing is None:
                by_relation[relation_id] = row
                continue
            if existing["way_members"] != row["way_members"]:
                existing.setdefault("variant_regions", []).append(region)
    return {
        "relations": by_relation,
        "relation_regions": {str(k): v for k, v in sorted(relation_regions.items())},
    }


def _scan_required_ways(pbf_paths: list[Path], required_way_ids: set[int]) -> tuple[dict[int, dict], dict[int, list[str]]]:
    union: dict[int, dict] = {}
    regions_by_way: dict[int, list[str]] = defaultdict(list)
    for path in pbf_paths:
        scanner = RequiredWayScanner(required_way_ids)
        scanner.apply_file(str(path), locations=False)
        region = path.name.removesuffix("-latest.osm.pbf")
        for way_id, row in scanner.ways.items():
            regions_by_way[way_id].append(region)
            existing = union.get(way_id)
            if existing is None:
                union[way_id] = row
            elif existing["node_refs"] != row["node_refs"]:
                existing.setdefault("variant_regions", []).append(region)
    return union, regions_by_way


def _scan_required_nodes(pbf_paths: list[Path], required_node_ids: set[int]) -> dict[int, list[str]]:
    regions_by_node: dict[int, list[str]] = defaultdict(list)
    for path in pbf_paths:
        scanner = RequiredNodeScanner(required_node_ids)
        scanner.apply_file(str(path), locations=False)
        region = path.name.removesuffix("-latest.osm.pbf")
        for node_id in scanner.nodes:
            regions_by_node[node_id].append(region)
    return regions_by_node


def _way_endpoint_graph(ways: dict[int, dict]) -> dict[str, object]:
    degree = defaultdict(int)
    for way in ways.values():
        refs = way.get("node_refs") or []
        if len(refs) < 2:
            continue
        degree[refs[0]] += 1
        degree[refs[-1]] += 1
    odd = sorted(node_id for node_id, count in degree.items() if count % 2)
    degree_counts = defaultdict(int)
    for count in degree.values():
        degree_counts[str(count)] += 1
    return {
        "endpoint_node_count": len(degree),
        "odd_endpoint_count": len(odd),
        "odd_endpoint_sample": odd[:50],
        "endpoint_degree_distribution": dict(sorted(degree_counts.items(), key=lambda item: int(item[0]))),
    }


def _relation_report(
    relation_id: int,
    relation: dict | None,
    relation_regions: list[str],
    way_union: dict[int, dict],
    regions_by_way: dict[int, list[str]],
    regions_by_node: dict[int, list[str]],
) -> dict:
    if relation is None:
        return {
            "relation_id": relation_id,
            "relation_found": False,
        }

    required_way_ids = [int(row["ref"]) for row in relation["way_members"]]
    present_way_ids = [way_id for way_id in required_way_ids if way_id in way_union]
    missing_way_ids = [way_id for way_id in required_way_ids if way_id not in way_union]
    required_node_ids = sorted(
        {
            node_id
            for way_id in present_way_ids
            for node_id in way_union[way_id].get("node_refs", [])
        }
    )
    missing_node_ids = [node_id for node_id in required_node_ids if node_id not in regions_by_node]
    present_ways = {way_id: way_union[way_id] for way_id in present_way_ids}

    return {
        "relation_id": relation_id,
        "relation_found": True,
        "name": relation["tags"].get("name") or relation["tags"].get("name:en"),
        "admin_level": relation["tags"].get("admin_level"),
        "relation_regions": relation_regions,
        "required_way_count": len(required_way_ids),
        "present_way_count": len(present_way_ids),
        "missing_way_count": len(missing_way_ids),
        "missing_way_ids": missing_way_ids[:200],
        "required_node_count_from_present_ways": len(required_node_ids),
        "missing_node_count_from_present_ways": len(missing_node_ids),
        "missing_node_ids": missing_node_ids[:200],
        "closed_individual_way_count": sum(1 for way_id in present_way_ids if way_union[way_id].get("closed")),
        "endpoint_graph": _way_endpoint_graph(present_ways),
        "way_region_coverage": {
            str(way_id): regions_by_way.get(way_id, [])
            for way_id in required_way_ids[:200]
        },
        "sample_missing_way_regions_expected_from_relation_regions": relation_regions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relation-ids", default="61320,2552485")
    parser.add_argument("--regions", default=",".join(US_REGIONS))
    args = parser.parse_args()

    osm_dir = Path(args.osm_dir)
    relation_ids = _parse_relation_ids(args.relation_ids)
    regions = tuple(part.strip() for part in args.regions.split(",") if part.strip())
    pbf_paths = [osm_dir / f"{region}-latest.osm.pbf" for region in regions]
    missing_pbf = [str(path) for path in pbf_paths if not path.exists()]
    if missing_pbf:
        raise FileNotFoundError(", ".join(missing_pbf))

    print(f"[{datetime.now()}] scanning target relations...", flush=True)
    relation_scan = _scan_target_relations(pbf_paths, relation_ids)
    relations = relation_scan["relations"]
    required_way_ids = {
        int(member["ref"])
        for relation in relations.values()
        for member in relation["way_members"]
    }

    print(f"[{datetime.now()}] scanning {len(required_way_ids)} required ways...", flush=True)
    way_union, regions_by_way = _scan_required_ways(pbf_paths, required_way_ids)
    required_node_ids = {
        int(node_id)
        for way in way_union.values()
        for node_id in way.get("node_refs", [])
    }

    way_regions = {
        region
        for regions_for_way in regions_by_way.values()
        for region in regions_for_way
    }
    node_pbf_paths = [
        osm_dir / f"{region}-latest.osm.pbf"
        for region in regions
        if region in way_regions
    ]
    print(
        f"[{datetime.now()}] scanning {len(required_node_ids)} required nodes "
        f"from {len(node_pbf_paths)} extracts with required ways...",
        flush=True,
    )
    regions_by_node = _scan_required_nodes(node_pbf_paths, required_node_ids)

    relation_reports = []
    for relation_id in sorted(relation_ids):
        relation_reports.append(
            _relation_report(
                relation_id,
                relations.get(relation_id),
                relation_scan["relation_regions"].get(str(relation_id), []),
                way_union,
                regions_by_way,
                regions_by_node,
            )
        )

    output = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "osm_dir": str(osm_dir),
        "region_count": len(regions),
        "relation_ids": sorted(relation_ids),
        "union_required_way_count": len(required_way_ids),
        "union_present_way_count": len(way_union),
        "union_missing_way_count": len(required_way_ids - set(way_union)),
        "union_required_node_count": len(required_node_ids),
        "union_present_node_count": len(regions_by_node),
        "union_missing_node_count": len(required_node_ids - set(regions_by_node)),
        "relations": relation_reports,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
