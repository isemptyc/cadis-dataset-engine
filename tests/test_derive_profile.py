"""Tests for derive_profile.py on a synthetic three-level dataset.

Run: python3 -m unittest tests/test_derive_profile.py (from the repo root).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import derive_profile  # noqa: E402


def _square(minx: float, miny: float, maxx: float, maxy: float) -> list[tuple[float, float]]:
    return [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]


def _ffsf(features: list[list[tuple[float, float]]]) -> bytes:
    """FFSF v2 with one single-ring part per feature."""
    index, bboxes, geoms, rings, geometry = bytearray(), bytearray(), bytearray(), bytearray(), bytearray()
    for part, ring in enumerate(features):
        xs, ys = [x for x, _ in ring], [y for _, y in ring]
        minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
        index += struct.pack("<4I", 0, 0, part, 1)
        bboxes += struct.pack("<4f", minx, miny, maxx, maxy)
        offset = len(geometry)
        for x, y in ring:
            geometry += struct.pack(
                "<2H",
                round((x - minx) / (maxx - minx) * 65535),
                round((y - miny) / (maxy - miny) * 65535),
            )
        geoms += struct.pack("<4I", offset, len(geometry) - offset, part, 1)
        rings += struct.pack("<I", len(ring))
    header = b"FFSF" + struct.pack("<III", 2, len(features), len(features))
    return header + bytes(index + bboxes + geoms + rings + geometry)


class DeriveProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source = root / "source"
        self.output = root / "profile"
        self.source.mkdir()
        # Region R (level 4) covers x 0..10; county A (level 6) inside it;
        # districts: D inside R (dropped) and G outside every level-4
        # polygon (a gap filler, repaired to the polygon-less region Q).
        self.features = [
            ("xx_r1", 4, "R", None, _square(0, 0, 10, 10), True),
            ("xx_a1", 6, "A", "xx_r1", _square(1, 1, 5, 5), True),
            ("xx_d1", 8, "D", "xx_a1", _square(2, 2, 3, 3), True),
            ("xx_g1", 8, "G", "xx_x1", _square(20, 20, 22, 22), True),
            # A parentless level-6 unit inside R (common in real releases).
            ("xx_o1", 6, "O", None, _square(6, 6, 8, 8), True),
        ]
        self.nodes = [
            {"id": "r1", "level": 4, "name": "R", "names": {}, "parent_id": None},
            {"id": "q1", "level": 4, "name": "Q", "names": {}, "parent_id": None},
            {"id": "a1", "level": 6, "name": "A", "names": {}, "parent_id": "r1"},
            {"id": "x1", "level": 6, "name": "X", "names": {}, "parent_id": "q1"},
            {"id": "d1", "level": 8, "name": "D", "names": {}, "parent_id": "a1"},
            {"id": "g1", "level": 8, "name": "G", "names": {}, "parent_id": "x1"},
            {"id": "o1", "level": 6, "name": "O", "names": {}, "parent_id": None},
        ]
        self._write_geography()
        policy = {
            "allowed_levels": [4, 6, 8],
            "allowed_shapes": [[4], [4, 6], [4, 6, 8], [4, 8], [8]],
            "shape_status": [
                {"levels": [4], "status": "partial"},
                {"levels": [4, 6], "status": "ok"},
                {"levels": [4, 6, 8], "status": "ok"},
                {"levels": [4, 8], "status": "partial"},
                {"levels": [8], "status": "partial"},
            ],
            "layers": {"hierarchy_required": True, "repair_required": False},
            "hierarchy_repair_rules": {"parent_level": 4, "child_levels": [6, 8]},
        }
        (self.source / "runtime_policy.json").write_text(json.dumps(policy))
        self.manifest = json.dumps({
            "dataset_id": "xx.admin", "country_iso": "XX", "dataset_version": "v1.0.0",
            "checksums": {"files": {}},
        }).encode()
        (self.source / "dataset_release_manifest.json").write_bytes(self.manifest)

    def _write_geography(self) -> None:
        (self.source / "geometry.ffsf").write_bytes(_ffsf([f[4] for f in self.features]))
        meta = [
            {
                "feature_id": fid, "level": level, "name": name, "names": {"en": name},
                "parent_id": parent, "representative_point_exact": [ring[0][0] + 0.5, ring[0][1] + 0.5],
                "country_scope_flag": flag,
            }
            for fid, level, name, parent, ring, flag in self.features
        ]
        (self.source / "geometry_meta.json").write_text(json.dumps(meta))
        (self.source / "hierarchy.json").write_text(json.dumps({"nodes": self.nodes}))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def derive(self, levels: set[int]) -> dict:
        return derive_profile.derive_profile(
            self.source, self.output, levels=levels, profile="photolens", revision=1
        )

    def test_retains_levels_scope_and_gap_fillers_only(self) -> None:
        summary = self.derive({4})
        meta = json.loads((self.output / "geometry_meta.json").read_text())
        self.assertEqual([m["feature_id"] for m in meta], ["xx_r1", "xx_g1"])
        self.assertEqual(summary["retained_levels"], [4])
        self.assertEqual(summary["gap_filler_features"], 1)
        ffsf = derive_profile.read_ffsf((self.output / "geometry.ffsf").read_bytes())
        self.assertEqual(len(ffsf["features"]), 2)
        source = derive_profile.read_ffsf((self.source / "geometry.ffsf").read_bytes())
        # Geometry bytes are copied verbatim, never re-quantized.
        g_source = source["geoms"][3]
        g_profile = ffsf["geoms"][1]
        self.assertEqual(
            source["geometry"][g_source[0]: g_source[0] + g_source[1]],
            ffsf["geometry"][g_profile[0]: g_profile[0] + g_profile[1]],
        )

    def test_hierarchy_keeps_polygonless_retained_nodes_and_relinks(self) -> None:
        self.derive({4})
        nodes = {n["id"]: n for n in json.loads((self.output / "hierarchy.json").read_text())["nodes"]}
        # Q has no polygon but is retained, so G's repair still reaches it.
        self.assertEqual(set(nodes), {"r1", "q1", "g1"})
        self.assertEqual(nodes["g1"]["parent_id"], "q1")

    def test_policy_projects_shapes_and_keeps_repair(self) -> None:
        self.derive({4})
        policy = json.loads((self.output / "runtime_policy.json").read_text())
        self.assertEqual(policy["allowed_levels"], [4, 8])
        self.assertEqual(policy["allowed_shapes"], [[4], [4, 8], [8]])
        status = {tuple(e["levels"]): e["status"] for e in policy["shape_status"]}
        self.assertEqual(status[(4,)], "ok")          # [4, 6] was ok and projects here
        self.assertEqual(status[(4, 8)], "ok")        # [4, 6, 8] was ok
        self.assertEqual(policy["hierarchy_repair_rules"], {"parent_level": 4, "child_levels": [8]})
        self.assertTrue(policy["layers"]["hierarchy_required"])

    def test_repair_is_disabled_when_its_parent_level_is_dropped(self) -> None:
        policy = json.loads((self.source / "runtime_policy.json").read_text())
        policy["hierarchy_repair_rules"]["parent_level"] = 6
        (self.source / "runtime_policy.json").write_text(json.dumps(policy))
        self.derive({8})
        derived = json.loads((self.output / "runtime_policy.json").read_text())
        self.assertNotIn("hierarchy_repair_rules", derived)
        self.assertFalse(derived["layers"]["hierarchy_required"])

    def test_manifest_names_profile_source_and_checksums(self) -> None:
        summary = self.derive({4})
        manifest_bytes = (self.output / "dataset_release_manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertEqual(manifest["dataset_version"], "v1.0.0-photolens.1")
        self.assertEqual(summary["manifest_sha256"], hashlib.sha256(manifest_bytes).hexdigest())
        profile = manifest["derived_profile"]
        self.assertEqual(profile["requested_levels"], [4])
        self.assertEqual(profile["inferred_parents"]["source"], "inferred_containment")
        # The manifest names the derivation code, not the commit, so an engine
        # commit that leaves the derivation unchanged keeps profile bytes.
        self.assertEqual(profile["derivation_code_sha256"], derive_profile.derivation_code_sha256())
        self.assertEqual(summary["derivation_code_sha256"], profile["derivation_code_sha256"])
        self.assertNotIn("commit", profile)
        self.assertEqual(profile["source"]["manifest_sha256"], hashlib.sha256(self.manifest).hexdigest())
        for name, entry in manifest["checksums"]["files"].items():
            data = (self.output / name).read_bytes()
            self.assertEqual(entry, {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)})

    def test_parentless_retained_unit_gets_its_containing_parent(self) -> None:
        summary = self.derive({4, 6})
        meta = {m["feature_id"]: m for m in json.loads((self.output / "geometry_meta.json").read_text())}
        nodes = {n["id"]: n for n in json.loads((self.output / "hierarchy.json").read_text())["nodes"]}
        # O had no parent; its representative point lies in R. The inferred
        # parent is marked so it stays distinct from a recorded one.
        self.assertEqual(meta["xx_o1"]["parent_id"], "xx_r1")
        self.assertEqual(nodes["o1"]["parent_id"], "r1")
        self.assertEqual(meta["xx_o1"]["parent_source"], "inferred_containment")
        self.assertEqual(nodes["o1"]["parent_source"], "inferred_containment")
        # A recorded parent is never replaced and carries no inference mark.
        self.assertEqual(meta["xx_a1"]["parent_id"], "xx_r1")
        self.assertEqual(nodes["a1"]["parent_id"], "r1")
        self.assertNotIn("parent_source", meta["xx_a1"])
        self.assertNotIn("parent_source", nodes["a1"])
        self.assertEqual(summary["spatial_parents"], 1)
        self.assertEqual(summary["inferred_parents"], {"assigned": 1, "ambiguous": 0, "no_container": 0})

    def test_ambiguous_or_missing_container_leaves_ancestry_unknown(self) -> None:
        # S overlaps R at x/y 8..10; M's representative point (9, 9) lies in
        # both, and N lies outside every level-4 unit.
        self.features += [
            ("xx_s1", 4, "S", None, _square(8, 8, 14, 14), True),
            ("xx_m1", 6, "M", None, _square(8.5, 8.5, 9.5, 9.5), True),
            ("xx_n1", 6, "N", None, _square(30, 30, 32, 32), True),
        ]
        self.nodes += [
            {"id": "s1", "level": 4, "name": "S", "names": {}, "parent_id": None},
            {"id": "m1", "level": 6, "name": "M", "names": {}, "parent_id": None},
            {"id": "n1", "level": 6, "name": "N", "names": {}, "parent_id": None},
        ]
        self._write_geography()
        summary = self.derive({4, 6})
        meta = {m["feature_id"]: m for m in json.loads((self.output / "geometry_meta.json").read_text())}
        nodes = {n["id"]: n for n in json.loads((self.output / "hierarchy.json").read_text())["nodes"]}
        for fid, nid in (("xx_m1", "m1"), ("xx_n1", "n1")):
            self.assertIsNone(meta[fid]["parent_id"])
            self.assertIsNone(nodes[nid]["parent_id"])
            self.assertNotIn("parent_source", meta[fid])
        self.assertEqual(summary["inferred_parents"], {"assigned": 1, "ambiguous": 1, "no_container": 1})

    def test_engine_identity_names_commit_and_code(self) -> None:
        identity = derive_profile.engine_identity()
        self.assertEqual(identity["derivation_code_sha256"], derive_profile.derivation_code_sha256())
        self.assertIn("commit", identity)
        self.assertIn("dirty", identity)

    def test_derivation_is_deterministic(self) -> None:
        first = self.derive({4})["manifest_sha256"]
        second = self.derive({4})["manifest_sha256"]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
