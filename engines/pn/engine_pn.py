from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

PN_LEVELS = (2, 4, 6, 8, 10)


class PitcairnIslandsAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "pn_admin"
    LEVELS = list(PN_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(PN_LEVELS)
    COUNTRY_ISO = "PN"
    COUNTRY_NAME = "Pitcairn Islands"
    WORK_DIR_NAME = "pitcairnislands"
    FILE_STEM = "pitcairnislands"
    LEVEL_LABELS = default_level_labels(PN_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=PN_LEVELS,
        languages=("en",),
    )
