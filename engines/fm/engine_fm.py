from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

FM_LEVELS = (2, 4, 6, 8, 10)


class MicronesiaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "fm_admin"
    LEVELS = list(FM_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(FM_LEVELS)
    COUNTRY_ISO = "FM"
    COUNTRY_NAME = "Micronesia"
    WORK_DIR_NAME = "micronesia"
    FILE_STEM = "micronesia"
    LEVEL_LABELS = default_level_labels(FM_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=FM_LEVELS,
        languages=("en",),
    )
