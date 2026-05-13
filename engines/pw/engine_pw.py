from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

PW_LEVELS = (2, 4, 6, 8, 10)


class PalauAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "pw_admin"
    LEVELS = list(PW_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(PW_LEVELS)
    COUNTRY_ISO = "PW"
    COUNTRY_NAME = "Palau"
    WORK_DIR_NAME = "palau"
    FILE_STEM = "palau"
    LEVEL_LABELS = default_level_labels(PW_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=PW_LEVELS,
        languages=("en",),
    )
