from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

MH_LEVELS = (2, 4, 6, 8, 10)


class MarshallIslandsAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "mh_admin"
    LEVELS = list(MH_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(MH_LEVELS)
    COUNTRY_ISO = "MH"
    COUNTRY_NAME = "Marshall Islands"
    WORK_DIR_NAME = "marshallislands"
    FILE_STEM = "marshallislands"
    LEVEL_LABELS = default_level_labels(MH_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=MH_LEVELS,
        languages=("en",),
    )
