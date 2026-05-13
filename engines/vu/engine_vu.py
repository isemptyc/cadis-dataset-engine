from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

VU_LEVELS = (2, 4, 6, 8, 10)


class VanuatuAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "vu_admin"
    LEVELS = list(VU_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(VU_LEVELS)
    COUNTRY_ISO = "VU"
    COUNTRY_NAME = "Vanuatu"
    WORK_DIR_NAME = "vanuatu"
    FILE_STEM = "vanuatu"
    LEVEL_LABELS = default_level_labels(VU_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name", "name:fr"),
        levels=VU_LEVELS,
        languages=("en", "fr"),
    )
