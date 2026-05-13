from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

SB_LEVELS = (2, 4, 6, 8, 10)


class SolomonIslandsAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "sb_admin"
    LEVELS = list(SB_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(SB_LEVELS)
    COUNTRY_ISO = "SB"
    COUNTRY_NAME = "Solomon Islands"
    WORK_DIR_NAME = "solomonislands"
    FILE_STEM = "solomonislands"
    LEVEL_LABELS = default_level_labels(SB_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=SB_LEVELS,
        languages=("en",),
    )
