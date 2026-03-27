"""Aesthetic log — rolling log of daemon reactions (§4e).

Feeds into ``DAEMON_SELF.aesthetic_sensibilities`` over distillation cycles.
Written by ``daemon_integration`` when a thought is classified as
``aesthetic_reaction``.  Never deleted.

File: ``data/daemon_state/aesthetic_log.yaml``
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._paths import daemon_state_dir
from ._utils import _utcnow

_FILENAME = "aesthetic_log.yaml"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AestheticReaction(BaseModel):
    """A single aesthetic reaction entry."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_utcnow)
    thought: str
    reaction: str

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AestheticLog:
    """File-backed rolling log of aesthetic reactions.

    Append-only; entries are never deleted.
    Inject ``path`` in tests to use a temporary file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / _FILENAME)
        self._entries: list[AestheticReaction] = []
        self._load()

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: list[dict[str, Any]] = yaml.safe_load(self._path.read_text()) or []
        except yaml.YAMLError as exc:
            warnings.warn(
                f"Aesthetic log file {self._path} could not be parsed — "
                f"starting with empty log. Error: {exc}",
                stacklevel=2,
            )
            return
        self._entries = [AestheticReaction.model_validate(r) for r in raw]

    def _save(self) -> None:
        data = [e.model_dump(mode="json") for e in self._entries]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, thought: str, reaction: str) -> AestheticReaction:
        """Append a new reaction entry and persist.

        Returns the created ``AestheticReaction``.
        """
        entry = AestheticReaction(thought=thought, reaction=reaction)
        self._entries.append(entry)
        self._save()
        return entry

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all_entries(self) -> list[AestheticReaction]:
        """Return all entries in the order they were appended."""
        return list(self._entries)
