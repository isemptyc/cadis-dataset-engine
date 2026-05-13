from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

TV_LEVELS = (2, 4, 6, 8, 10)


class TuvaluAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "tv_admin"
    LEVELS = list(TV_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(TV_LEVELS)
    COUNTRY_ISO = "TV"
    COUNTRY_NAME = "Tuvalu"
    WORK_DIR_NAME = "tuvalu"
    FILE_STEM = "tuvalu"
    LEVEL_LABELS = default_level_labels(TV_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=TV_LEVELS,
        languages=("en",),
    )
