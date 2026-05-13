from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

TO_LEVELS = (2, 4, 6, 8, 10)


class TongaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "to_admin"
    LEVELS = list(TO_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(TO_LEVELS)
    COUNTRY_ISO = "TO"
    COUNTRY_NAME = "Tonga"
    WORK_DIR_NAME = "tonga"
    FILE_STEM = "tonga"
    LEVEL_LABELS = default_level_labels(TO_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=TO_LEVELS,
        languages=("en",),
    )
