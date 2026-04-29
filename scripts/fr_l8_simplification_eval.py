#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from shapely.geometry import Point, mapping, shape
from shapely.ops import nearest_points

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ffsf import export_cadis_to_ffsf


LEVEL_KEYS = ("4", "6", "8")


@dataclass(frozen=True)
class SamplePoint:
    sample_id: str
    lat: float
    lon: float
    category: str
    feature_id: str | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class JsonStream:
    def __init__(self, f: Any, *, chunk_size: int = 1024 * 1024):
        self.f = f
        self.chunk_size = chunk_size
        self.buf = ""

    def fill(self) -> bool:
        chunk = self.f.read(self.chunk_size)
        if not chunk:
            return False
        self.buf += chunk
        return True

    def read_until(self, needle: str) -> None:
        while True:
            pos = self.buf.find(needle)
            if pos >= 0:
                self.buf = self.buf[pos + len(needle) :]
                return
            keep = max(0, len(needle) - 1)
            if len(self.buf) > keep:
                self.buf = self.buf[-keep:]
            if not self.fill():
                raise EOFError(f"Could not find marker: {needle}")

    def trim_value_prefix(self) -> bool:
        while True:
            self.buf = self.buf.lstrip()
            if not self.buf and not self.fill():
                return False
            if self.buf.startswith(","):
                self.buf = self.buf[1:]
                continue
            return True


def iter_array_objects(stream: JsonStream) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    while True:
        if not stream.trim_value_prefix():
            return
        if stream.buf[0] == "]":
            stream.buf = stream.buf[1:]
            return

        while True:
            try:
                obj, idx = decoder.raw_decode(stream.buf)
                stream.buf = stream.buf[idx:]
                if not isinstance(obj, dict):
                    raise ValueError("Expected object in admin_by_level array.")
                yield obj
                break
            except json.JSONDecodeError:
                if not stream.fill():
                    raise


def simplify_feature(
    feature: dict[str, Any],
    *,
    tolerance: float,
    stats: dict[str, Any],
) -> dict[str, Any]:
    geom = shape(feature["geometry"])
    before_area = float(geom.area)
    before_parts = len(getattr(geom, "geoms", [geom]))
    simplified = geom.simplify(tolerance, preserve_topology=True)
    if not simplified.is_valid:
        simplified = simplified.buffer(0)

    after_area = float(simplified.area)
    after_parts = len(getattr(simplified, "geoms", [simplified]))
    stats["features"] += 1
    stats["before_area"] += before_area
    stats["after_area"] += after_area
    stats["before_parts"] += before_parts
    stats["after_parts"] += after_parts
    if simplified.is_empty:
        stats["empty_features"].append(feature.get("id"))
    if not simplified.is_valid:
        stats["invalid_features"].append(feature.get("id"))
    if before_parts > 0 and after_parts < before_parts:
        stats["part_loss_features"].append(feature.get("id"))

    out = dict(feature)
    out["geometry"] = mapping(simplified)
    out["bbox"] = list(map(float, simplified.bounds)) if not simplified.is_empty else feature.get("bbox")
    return out


def write_simplified_admin_json(
    *,
    source_admin_json: Path,
    output_admin_json: Path,
    tolerance: float,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "tolerance": tolerance,
        "features": 0,
        "before_area": 0.0,
        "after_area": 0.0,
        "before_parts": 0,
        "after_parts": 0,
        "empty_features": [],
        "invalid_features": [],
        "part_loss_features": [],
    }
    output_admin_json.parent.mkdir(parents=True, exist_ok=True)
    with source_admin_json.open("r", encoding="utf-8") as src, output_admin_json.open("w", encoding="utf-8") as out:
        stream = JsonStream(src)
        out.write(
            json.dumps(
                {
                    "meta": {
                        "country": "FR",
                        "country_name": "France",
                        "country_geometry_filter_applied": True,
                        "levels": [4, 6, 8],
                        "source": "OpenStreetMap",
                        "evaluation_note": "level_8_simplification_variant",
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )[:-1]
        )
        out.write(',"admin_by_level":{')
        stream.read_until('"admin_by_level": {')
        for level_idx, level_key in enumerate(LEVEL_KEYS):
            stream.read_until(f'"{level_key}": [')
            if level_idx:
                out.write(",")
            out.write(json.dumps(level_key))
            out.write(":[")
            first = True
            for feature in iter_array_objects(stream):
                if level_key == "8":
                    feature = simplify_feature(feature, tolerance=tolerance, stats=stats)
                if not first:
                    out.write(",")
                json.dump(feature, out, ensure_ascii=False, separators=(",", ":"))
                first = False
            out.write("]")
        out.write("}}\n")
    stats["area_ratio"] = stats["after_area"] / stats["before_area"] if stats["before_area"] else None
    return stats


def write_release_manifest(*, variant_dir: Path, baseline_manifest: Path) -> None:
    payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
    payload["dataset_version"] = f"{payload.get('dataset_version', 'v1.0.0')}-l8-simplification-eval"
    payload["generated_at"] = "evaluation-local"
    files = {}
    for name in ("geometry.ffsf", "geometry_meta.json", "hierarchy.json", "runtime_policy.json"):
        p = variant_dir / name
        files[name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    payload["checksums"] = {"files": files}
    (variant_dir / "dataset_release_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def package_variant(variant_dir: Path) -> Path:
    package_path = variant_dir / "dataset_package.tar.gz"
    with tarfile.open(package_path, "w:gz") as tar:
        for name in ("dataset_release_manifest.json", "geometry.ffsf", "geometry_meta.json", "hierarchy.json", "runtime_policy.json"):
            tar.add(variant_dir / name, arcname=name)
    (variant_dir / "dataset_package.tar.gz.sha256").write_text(
        f"{sha256_file(package_path)}  dataset_package.tar.gz\n",
        encoding="utf-8",
    )
    return package_path


def build_variant(args: argparse.Namespace) -> None:
    variant_dir = args.output_dir / f"tol_{args.tolerance:g}"
    admin_json = variant_dir / "france_admin_simplified.json"
    stats = write_simplified_admin_json(
        source_admin_json=args.source_admin_json,
        output_admin_json=admin_json,
        tolerance=args.tolerance,
    )
    export_cadis_to_ffsf(
        input_path=admin_json,
        output_path=variant_dir / "geometry.ffsf",
        version=3,
        country_geometry_path=args.country_geometry,
    )
    shutil.copy2(args.baseline_dir / "hierarchy.json", variant_dir / "hierarchy.json")
    shutil.copy2(args.baseline_dir / "runtime_policy.json", variant_dir / "runtime_policy.json")
    # Exporter writes <COUNTRY>_feature_meta_by_index.json next to output.
    meta_path = variant_dir / "FR_feature_meta_by_index.json"
    if meta_path.exists():
        shutil.copy2(meta_path, variant_dir / "geometry_meta.json")
    write_release_manifest(variant_dir=variant_dir, baseline_manifest=args.baseline_manifest)
    package_path = package_variant(variant_dir)
    stats["sizes"] = {
        name: (variant_dir / name).stat().st_size
        for name in ("geometry.ffsf", "geometry_meta.json", "hierarchy.json", "runtime_policy.json", "dataset_package.tar.gz")
    }
    stats["package_path"] = str(package_path)
    (variant_dir / "simplification_build_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def load_runtime(dataset_dir: Path) -> Any:
    cadis_path = REPO_ROOT.parent / "cadis"
    if cadis_path.exists():
        sys.path.insert(0, str(cadis_path))
    else:
        sys.path.insert(0, "/Users/isempty/Projects/my_cadis/cadis")
    from cadis.runtime import CadisRuntime

    return CadisRuntime(dataset_dir=dataset_dir)


def hierarchy_signature(body: dict[str, Any]) -> tuple[tuple[int, str | None, str | None, str | None], ...]:
    rows = (((body.get("result") or {}).get("admin_hierarchy")) or [])
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append((row.get("level"), row.get("osm_id"), row.get("name"), row.get("source")))
    return tuple(out)


def levels(sig: tuple[tuple[int, str | None, str | None, str | None], ...]) -> tuple[int, ...]:
    return tuple(int(row[0]) for row in sig if isinstance(row[0], int))


def level_id(sig: tuple[tuple[int, str | None, str | None, str | None], ...], level: int) -> str | None:
    for lvl, osm_id, _name, _source in sig:
        if lvl == level:
            return osm_id
    return None


def make_samples(admin_json: Path, geometry_meta: Path, *, limit: int) -> list[SamplePoint]:
    meta = json.loads(geometry_meta.read_text(encoding="utf-8"))
    samples: list[SamplePoint] = []
    level8 = [row for row in meta if row.get("level") == 8 and isinstance(row.get("representative_point_exact"), list)]
    for idx, row in enumerate(level8[:limit]):
        lon, lat = row["representative_point_exact"]
        samples.append(SamplePoint(f"rep:{row['feature_id']}", float(lat), float(lon), "representative", row["feature_id"]))

    fragile: list[tuple[float, str, Point, Any]] = []
    with admin_json.open("r", encoding="utf-8") as src:
        stream = JsonStream(src)
        stream.read_until('"admin_by_level": {')
        stream.read_until('"4": [')
        for _ in iter_array_objects(stream):
            pass
        stream.read_until('"6": [')
        for _ in iter_array_objects(stream):
            pass
        stream.read_until('"8": [')
        for feature in iter_array_objects(stream):
            geom = shape(feature["geometry"])
            if geom.is_empty:
                continue
            fragile.append((float(geom.area), feature["id"], geom.representative_point(), geom))
    fragile.sort(key=lambda item: (item[0], item[1]))
    for area, feature_id, rep, geom in fragile[: min(500, len(fragile))]:
        samples.append(SamplePoint(f"fragile-rep:{feature_id}", rep.y, rep.x, "fragile_representative", feature_id))
        boundary = geom.boundary
        if not boundary.is_empty:
            try:
                _a, b = nearest_points(rep, boundary)
                for frac in (0.90, 0.97):
                    x = rep.x + (b.x - rep.x) * frac
                    y = rep.y + (b.y - rep.y) * frac
                    p = Point(x, y)
                    if geom.covers(p):
                        samples.append(SamplePoint(f"fragile-boundary:{feature_id}:{frac}", y, x, "fragile_boundary", feature_id))
            except Exception:
                pass
    seen = set()
    out = []
    for sample in samples:
        if sample.sample_id in seen:
            continue
        seen.add(sample.sample_id)
        out.append(sample)
    return out


def compare_once(samples: list[SamplePoint], baseline_dir: Path, variant_dir: Path) -> dict[str, Any]:
    baseline = load_runtime(baseline_dir)
    variant = load_runtime(variant_dir)
    counts = Counter()
    switch_rows = []
    loss_rows = []
    source_transitions = Counter()
    shape_transitions = Counter()
    for sample in samples:
        b_body = baseline.lookup(sample.lat, sample.lon)
        v_body = variant.lookup(sample.lat, sample.lon)
        b_sig = hierarchy_signature(b_body)
        v_sig = hierarchy_signature(v_body)
        b_levels = levels(b_sig)
        v_levels = levels(v_sig)
        counts["total"] += 1
        if b_sig == v_sig:
            counts["exact_match"] += 1
        else:
            counts["mismatch"] += 1
        shape_transitions[(str(b_levels), str(v_levels))] += 1
        b_l4, v_l4 = level_id(b_sig, 4), level_id(v_sig, 4)
        b_l6, v_l6 = level_id(b_sig, 6), level_id(v_sig, 6)
        b_l8, v_l8 = level_id(b_sig, 8), level_id(v_sig, 8)
        if b_l4 != v_l4:
            counts["level4_mismatch"] += 1
        if b_l6 != v_l6:
            counts["level6_mismatch"] += 1
        if b_l8 and v_l8 and b_l8 != v_l8:
            counts["commune_switch"] += 1
            if len(switch_rows) < 200:
                switch_rows.append(sample.__dict__ | {"baseline_l8": b_l8, "variant_l8": v_l8})
        if b_l8 and not v_l8:
            counts["level8_loss"] += 1
            if len(loss_rows) < 200:
                loss_rows.append(sample.__dict__ | {"baseline_l8": b_l8})
        b_source = (b_sig[-1][3] if b_sig else None) or "__none__"
        v_source = (v_sig[-1][3] if v_sig else None) or "__none__"
        source_transitions[(b_source, v_source)] += 1
    total = counts["total"] or 1
    return {
        "counts": dict(counts),
        "rates": {
            "exact_match": counts["exact_match"] / total,
            "mismatch": counts["mismatch"] / total,
            "commune_switch": counts["commune_switch"] / total,
            "level8_loss": counts["level8_loss"] / total,
            "level4_mismatch": counts["level4_mismatch"] / total,
            "level6_mismatch": counts["level6_mismatch"] / total,
        },
        "shape_transitions": {f"{k[0]} -> {k[1]}": v for k, v in shape_transitions.most_common()},
        "source_transitions": {f"{k[0]} -> {k[1]}": v for k, v in source_transitions.most_common()},
        "commune_switch_samples": switch_rows,
        "level8_loss_samples": loss_rows,
    }


def evaluate(args: argparse.Namespace) -> None:
    samples = make_samples(args.source_admin_json, args.baseline_dir / "geometry_meta.json", limit=args.sample_limit)
    runs = []
    for _ in range(args.runs):
        runs.append(compare_once(samples, args.baseline_dir, args.variant_dir))
    stable_keys = ("commune_switch_samples", "level8_loss_samples", "shape_transitions", "source_transitions", "counts")
    deterministic = all(
        json.dumps(runs[0].get(k), sort_keys=True, ensure_ascii=False) == json.dumps(run.get(k), sort_keys=True, ensure_ascii=False)
        for run in runs[1:]
        for k in stable_keys
    )
    first = runs[0]
    decision = "safe"
    if not deterministic or first["counts"].get("level4_mismatch") or first["counts"].get("level6_mismatch") or first["counts"].get("commune_switch"):
        decision = "reject"
    elif first["counts"].get("level8_loss"):
        decision = "tuning_needed"
    report = {
        "variant_dir": str(args.variant_dir),
        "baseline_dir": str(args.baseline_dir),
        "sample_count": len(samples),
        "runs": args.runs,
        "deterministic": deterministic,
        "decision": decision,
        "run_reports": runs,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FR level-8 geometry simplification variants.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-variant")
    build.add_argument("--source-admin-json", type=Path, required=True)
    build.add_argument("--baseline-dir", type=Path, required=True)
    build.add_argument("--baseline-manifest", type=Path, required=True)
    build.add_argument("--country-geometry", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--tolerance", type=float, required=True)
    build.set_defaults(func=build_variant)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--source-admin-json", type=Path, required=True)
    ev.add_argument("--baseline-dir", type=Path, required=True)
    ev.add_argument("--variant-dir", type=Path, required=True)
    ev.add_argument("--sample-limit", type=int, default=5000)
    ev.add_argument("--runs", type=int, default=3)
    ev.add_argument("--report", type=Path, required=True)
    ev.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
