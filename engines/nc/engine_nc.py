from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

NC_LEVELS = (2, 4, 6, 8, 10)


class NewCaledoniaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "nc_admin"
    LEVELS = list(NC_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(NC_LEVELS)
    COUNTRY_ISO = "NC"
    COUNTRY_NAME = "New Caledonia"
    WORK_DIR_NAME = "newcaledonia"
    FILE_STEM = "newcaledonia"
    LEVEL_LABELS = default_level_labels(NC_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:fr", "name:en", "name", "official_name"),
        levels=NC_LEVELS,
        languages=("fr", "en"),
    )
