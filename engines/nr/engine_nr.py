from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

NR_LEVELS = (2, 4, 6, 8, 10)


class NauruAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "nr_admin"
    LEVELS = list(NR_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(NR_LEVELS)
    COUNTRY_ISO = "NR"
    COUNTRY_NAME = "Nauru"
    WORK_DIR_NAME = "nauru"
    FILE_STEM = "nauru"
    LEVEL_LABELS = default_level_labels(NR_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=NR_LEVELS,
        languages=("en",),
    )
