from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import osmium

from engines.ca.engine_ca import CA_REGIONS


class AdminLevelProbe(osmium.SimpleHandler):
    def __init__(self, *, sample_limit: int):
        super().__init__()
        self.sample_limit = sample_limit
        self.counts: Counter[int] = Counter()
        self.samples: dict[int, list[dict]] = defaultdict(list)

    def relation(self, relation):
        tags = dict(relation.tags)
        if tags.get("boundary") != "administrative":
            return
        raw_level = tags.get("admin_level")
        if not isinstance(raw_level, str) or not raw_level.isdigit():
            return
        level = int(raw_level)
        self.counts[level] += 1
        if len(self.samples[level]) < self.sample_limit:
            self.samples[level].append(
                {
                    "osm_id": f"r{int(relation.id)}",
                    "name": tags.get("name:en") or tags.get("name") or tags.get("official_name"),
                    "admin_level": level,
                    "boundary": tags.get("boundary"),
                    "type": tags.get("type"),
                }
            )


def _selected_paths(osm_path: Path) -> list[Path]:
    if osm_path.is_file():
        return [osm_path]
    available = {path.name: path for path in osm_path.glob("*-latest.osm.pbf")}
    missing = []
    selected = []
    for region in CA_REGIONS:
        name = f"{region}-latest.osm.pbf"
        path = available.get(name)
        if path is None:
            missing.append(name)
        else:
            selected.append(path)
    if missing:
        raise FileNotFoundError("Missing Canada extracts: " + ", ".join(missing))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Canada OSM administrative relation levels.")
    parser.add_argument(
        "--osm",
        required=True,
        type=Path,
        help="Canada full PBF or directory containing province/territory *-latest.osm.pbf extracts.",
    )
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = _selected_paths(args.osm)
    totals: Counter[int] = Counter()
    by_file = {}
    samples: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        probe = AdminLevelProbe(sample_limit=args.sample_limit)
        print(f"Scanning {path.name}", flush=True)
        probe.apply_file(str(path), locations=False)
        counts = {str(level): count for level, count in sorted(probe.counts.items())}
        by_file[path.name] = counts
        totals.update(probe.counts)
        for level, level_samples in probe.samples.items():
            remaining = max(0, args.sample_limit - len(samples[level]))
            samples[level].extend(level_samples[:remaining])

    payload = {
        "osm": str(args.osm),
        "files": [str(path) for path in paths],
        "relation_counts_by_level": {str(level): count for level, count in sorted(totals.items())},
        "relation_counts_by_file": by_file,
        "samples_by_level": {
            str(level): rows for level, rows in sorted(samples.items())
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
