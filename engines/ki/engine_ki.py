from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

KI_LEVELS = (2, 4, 6, 8, 10)


class KiribatiAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "ki_admin"
    LEVELS = list(KI_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(KI_LEVELS)
    COUNTRY_ISO = "KI"
    COUNTRY_NAME = "Kiribati"
    WORK_DIR_NAME = "kiribati"
    FILE_STEM = "kiribati"
    LEVEL_LABELS = default_level_labels(KI_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=KI_LEVELS,
        languages=("en",),
    )
