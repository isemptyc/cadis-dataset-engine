from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

WF_LEVELS = (2, 4, 6, 8, 10)


class WallisAndFutunaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "wf_admin"
    LEVELS = list(WF_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(WF_LEVELS)
    COUNTRY_ISO = "WF"
    COUNTRY_NAME = "Wallis and Futuna"
    WORK_DIR_NAME = "wallisandfutuna"
    FILE_STEM = "wallisandfutuna"
    LEVEL_LABELS = default_level_labels(WF_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:fr", "name:en", "name", "official_name"),
        levels=WF_LEVELS,
        languages=("fr", "en"),
    )
