from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

WS_LEVELS = (2, 4, 6, 8, 10)


class SamoaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "ws_admin"
    LEVELS = list(WS_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(WS_LEVELS)
    COUNTRY_ISO = "WS"
    COUNTRY_NAME = "Samoa"
    WORK_DIR_NAME = "samoa"
    FILE_STEM = "samoa"
    LEVEL_LABELS = default_level_labels(WS_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=WS_LEVELS,
        languages=("en",),
    )
