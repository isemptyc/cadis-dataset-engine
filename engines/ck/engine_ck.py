from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

CK_LEVELS = (2, 4, 6, 8, 10)


class CookIslandsAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "ck_admin"
    LEVELS = list(CK_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(CK_LEVELS)
    COUNTRY_ISO = "CK"
    COUNTRY_NAME = "Cook Islands"
    WORK_DIR_NAME = "cookislands"
    FILE_STEM = "cookislands"
    LEVEL_LABELS = default_level_labels(CK_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=CK_LEVELS,
        languages=("en",),
    )
