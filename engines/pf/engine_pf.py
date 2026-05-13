from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

PF_LEVELS = (2, 4, 6, 8, 10)


class FrenchPolynesiaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "pf_admin"
    LEVELS = list(PF_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(PF_LEVELS)
    COUNTRY_ISO = "PF"
    COUNTRY_NAME = "French Polynesia"
    WORK_DIR_NAME = "frenchpolynesia"
    FILE_STEM = "frenchpolynesia"
    LEVEL_LABELS = default_level_labels(PF_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:fr", "name:en", "name", "official_name"),
        levels=PF_LEVELS,
        languages=("fr", "en"),
    )
