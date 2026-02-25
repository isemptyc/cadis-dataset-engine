from abc import ABC
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re


class DatasetBuildEngineBase(ABC):
    """
    Base class for country- or region-specific deterministic dataset build engines.

    Responsibilities:
    - Build engine-owned datasets in an external, writable work directory
    - Define release packaging boundaries and write build manifests
    - Emit runtime policy metadata consumed by downstream runtime repositories

    Non-responsibilities:
    - Runtime lookup pipeline behavior
    - Country-specific logic
    """

    # -------- identity --------
    ENGINE: str
    VERSION: str

    # -------- dataset scope --------
    LEVELS: list[int]
    COUNTRY_ISO: str
    COUNTRY_NAME: str

    # -------- internal state (populated at init) --------
    BUILD_MANIFEST_SCHEMA_VERSION = 2
    BUILD_MANIFEST_PROFILE = "cadis.dataset.build"
    BUILD_MANIFEST_FILENAME = "dataset_build_manifest.json"
    RUNTIME_POLICY_FILENAME = "runtime_policy.json"

    # ==================================================
    # lifecycle
    # ==================================================

    def __init__(self, *, work_dir: Path):
        self._work_dir = Path(work_dir)

    @classmethod
    def prepare_datasets(
        cls,
        *,
        osm_pbf_path: str | Path,
        work_dir: Path | None = None,
    ) -> Path:
        """
        One-time dataset initialization entrypoint.
        """
        engine = cls(
            osm_pbf_path=osm_pbf_path,
            work_dir=work_dir,
        )
        engine._write_dataset_build_manifest()
        return Path(engine._work_dir)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _release_dataset_paths(self) -> list[Path]:
        """
        Define distributable dataset boundary for release packaging.
        Engines may override this method for custom boundaries.
        """
        ordered_attrs = [
            "_country_path",
            "_admin_hierarchy_path",
            "_ffsf_dataset_path",
            "_ffsf_meta_path",
            "_semantic_dataset_path",
            "_l9_exception_dataset_path",
        ]
        paths: list[Path] = []
        seen: set[str] = set()
        for attr in ordered_attrs:
            value = getattr(self, attr, None)
            if not isinstance(value, Path):
                continue
            norm = str(value.resolve())
            if norm in seen:
                continue
            seen.add(norm)
            paths.append(value)
        runtime_policy = self._runtime_policy_path()
        if runtime_policy.exists():
            norm = str(runtime_policy.resolve())
            if norm not in seen:
                paths.append(runtime_policy)
        return paths

    def _runtime_policy_path(self) -> Path:
        return self._work_dir / self.RUNTIME_POLICY_FILENAME

    def _runtime_policy_payload(self) -> dict | None:
        """
        Engine-owned runtime policy payload.
        Override in country engines as needed.
        """
        levels = sorted({int(level) for level in getattr(self, "LEVELS", [])})
        if not levels:
            return None

        allowed_shapes_attr = getattr(self, "ALLOWED_SHAPES", None)
        if isinstance(allowed_shapes_attr, (set, list, tuple)) and allowed_shapes_attr:
            allowed_shapes = sorted(
                {
                    tuple(sorted({int(v) for v in shape}))
                    for shape in allowed_shapes_attr
                    if isinstance(shape, (set, list, tuple)) and shape
                }
            )
        else:
            allowed_shapes = [tuple(levels)]
        if not allowed_shapes:
            allowed_shapes = [tuple(levels)]

        full_shape = tuple(levels)
        shape_status = []
        for shape in allowed_shapes:
            status = "ok" if shape == full_shape else "partial"
            shape_status.append({"levels": list(shape), "status": status})

        parent_level = levels[0]
        child_levels = [lvl for lvl in levels if lvl != parent_level]
        hierarchy_path = getattr(self, "_admin_hierarchy_path", None)
        repair_path = getattr(self, "_repair_layer_path", None)

        return {
            "runtime_policy_version": "1.0",
            "allowed_levels": levels,
            "allowed_shapes": [list(shape) for shape in allowed_shapes],
            "shape_status": shape_status,
            "layers": {
                "hierarchy_required": isinstance(hierarchy_path, Path) and hierarchy_path.exists(),
                "repair_required": isinstance(repair_path, Path) and repair_path.exists(),
            },
            "hierarchy_repair_rules": {
                "parent_level": parent_level,
                "child_levels": child_levels,
            },
            "repair_rules": {
                "parent_level": parent_level,
                "child_levels": child_levels,
            },
        }

    def _write_runtime_policy(self) -> Path | None:
        payload = self._runtime_policy_payload()
        if payload is None:
            return None
        out = self._runtime_policy_path()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    def _dataset_build_manifest_path(self) -> Path:
        return self._work_dir / self.BUILD_MANIFEST_FILENAME

    def _dataset_build_manifest_dataset_id(self) -> str:
        return f"engine.{self.COUNTRY_ISO.lower()}_admin"

    def _write_dataset_build_manifest(self) -> Path:
        self._write_runtime_policy()
        release_paths = self._release_dataset_paths()
        if not release_paths:
            raise ValueError("No release dataset paths discovered for build manifest.")

        release_files: list[str] = []
        files_meta: dict[str, dict] = {}
        for path in release_paths:
            if not path.exists():
                raise FileNotFoundError(
                    f"Release dataset file missing when writing build manifest: {path}"
                )
            if not path.is_file():
                raise ValueError(
                    f"Release dataset entry must be a file when writing build manifest: {path}"
                )
            rel = path.relative_to(self._work_dir).as_posix()
            release_files.append(rel)
            files_meta[rel] = {
                "sha256": self._sha256_file(path),
                "size": path.stat().st_size,
            }

        payload = {
            "schema_version": self.BUILD_MANIFEST_SCHEMA_VERSION,
            "profile": self.BUILD_MANIFEST_PROFILE,
            "dataset_id": self._dataset_build_manifest_dataset_id(),
            "dataset_type": "engine",
            "engine": self.ENGINE,
            "engine_iso2": self.COUNTRY_ISO,
            "country_name": self.COUNTRY_NAME,
            "engine_logic_version": self.VERSION,
            "work_dir": str(self._work_dir.resolve()),
            "release_files": release_files,
            "files": files_meta,
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        out = self._dataset_build_manifest_path()
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _assert_required_build_datasets(self, required_paths: list[Path]) -> None:
        missing = [p for p in required_paths if not p.exists()]
        if missing:
            missing_str = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                "Required build datasets are missing. Run "
                f"{self.__class__.__name__}.prepare_datasets(...) first. "
                f"Missing: {missing_str}"
            )

    # ==================================================
    # hierarchy loader
    # ==================================================

    def _load_admin_hierarchy(self, path: Path):
        """
        Load admin hierarchy from rendered admin_tree.txt.

        Parsing rule:
        - Indentation defines hierarchy.
        - Stack always represents the active ancestor chain.
        - Parent MUST be determined *after* pruning deeper-or-equal nodes.
        """

        line_re = re.compile(
            r"^(?P<indent>\s*)- (?P<name>.+?) "
            r"\[level=(?P<level>\d+)\] "
            r"\[id=(?P<id>[^\]]+)\]"
        )

        nodes = []
        stack = []

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                m = line_re.match(raw)
                if not m:
                    continue

                indent = len(m.group("indent"))

                # CRITICAL: prune stack first
                # Remove nodes that are not ancestors of this indentation level
                while stack and stack[-1]["indent"] >= indent:
                    stack.pop()

                node = {
                    "id": m.group("id"),
                    "name": m.group("name"),
                    "level": int(m.group("level")),
                    # After pruning, stack[-1] (if exists) is the true semantic parent
                    "parent_id": stack[-1]["id"] if stack else None,
                    "indent": indent,
                }

                stack.append(node)
                nodes.append(node)

        return nodes
