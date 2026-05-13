from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

TK_LEVELS = (2, 4, 6, 8, 10)


class TokelauAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "tk_admin"
    LEVELS = list(TK_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(TK_LEVELS)
    COUNTRY_ISO = "TK"
    COUNTRY_NAME = "Tokelau"
    WORK_DIR_NAME = "tokelau"
    FILE_STEM = "tokelau"
    LEVEL_LABELS = default_level_labels(TK_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=TK_LEVELS,
        languages=("en",),
    )
