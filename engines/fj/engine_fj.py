from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

FJ_LEVELS = (2, 4, 6, 8, 10)


class FijiAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "fj_admin"
    LEVELS = list(FJ_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(FJ_LEVELS)
    COUNTRY_ISO = "FJ"
    COUNTRY_NAME = "Fiji"
    WORK_DIR_NAME = "fiji"
    FILE_STEM = "fiji"
    LEVEL_LABELS = default_level_labels(FJ_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=FJ_LEVELS,
        languages=("en",),
    )
