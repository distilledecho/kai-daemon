"""Shared enum types used across multiple state stores."""

from __future__ import annotations

from enum import StrEnum


class EpistemicOrigin(StrEnum):
    """Where an item originated. Set at write time; immutable thereafter."""

    INTERNAL = "internal"
    EXTERNAL_SEARCH = "external_search"
    INNER_LIFE_PIPELINE = "inner_life_pipeline"
