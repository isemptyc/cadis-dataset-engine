from engines.oceania_base import (
    OceaniaAdminEngineBase,
    all_nonempty_level_shapes,
    build_oceania_profile,
    default_level_labels,
)

PG_LEVELS = (2, 4, 6, 8, 10)


class PapuaNewGuineaAdminEngine(OceaniaAdminEngineBase):
    ENGINE = "pg_admin"
    LEVELS = list(PG_LEVELS)
    ALLOWED_SHAPES = all_nonempty_level_shapes(PG_LEVELS)
    COUNTRY_ISO = "PG"
    COUNTRY_NAME = "Papua New Guinea"
    WORK_DIR_NAME = "papuanewguinea"
    FILE_STEM = "papuanewguinea"
    LEVEL_LABELS = default_level_labels(PG_LEVELS)
    PROFILE = build_oceania_profile(
        name_keys=("name:en", "name", "official_name"),
        levels=PG_LEVELS,
        languages=("en",),
    )
