from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

NU_LEVELS = (2, 4, 6, 8, 10)


class NiueAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "nu_admin"
    LEVELS = list(NU_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(NU_LEVELS)
    COUNTRY_ISO = "NU"
    COUNTRY_NAME = "Niue"
    WORK_DIR_NAME = "niue"
    FILE_STEM = "niue"
    LEVEL_LABELS = default_level_labels(NU_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=NU_LEVELS,
        languages=("en",),
    )
