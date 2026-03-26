"""Holding store — things not ready for conversation (§4d).

Invariants enforced in code:

- ``type: reasoned_disagreement`` requires a non-null ``contradiction_id``.
  Enforced in ``HoldingStore.write()``.
- Urgency-based forced surface thresholds:
  - ``high``   → forced after 7 days
  - ``medium`` → forced after 21 days
  - ``low``    → never forced
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._paths import daemon_state_dir

_FORCED_SURFACE_DAYS: dict[str, int | None] = {
    "high": 7,
    "medium": 21,
    "low": None,
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HoldingType(StrEnum):
    OBSERVATION = "observation"
    CONNECTION = "connection"
    CHALLENGE = "challenge"
    DAEMON_CURIOSITY = "daemon_curiosity"
    OPEN_LOOP_FOLLOW_UP = "open_loop_follow_up"
    REASONED_DISAGREEMENT = "reasoned_disagreement"


class RegisterNeeded(StrEnum):
    EXPLORATORY = "exploratory"
    REFLECTIVE = "reflective"
    CASUAL = "casual"
    ANY = "any"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EpistemicOrigin(StrEnum):
    INTERNAL = "internal"
    EXTERNAL_SEARCH = "external_search"
    INNER_LIFE_PIPELINE = "inner_life_pipeline"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HoldingItem(BaseModel):
    """A single item in the holding store."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    type: HoldingType
    relevance_trigger: str
    register_needed: RegisterNeeded
    urgency: Urgency
    created: str = Field(default_factory=_utcnow)
    expires: str | None = None
    surfaced: str | None = None
    discharge_notes: str | None = None
    source_workflow: str
    epistemic_origin: EpistemicOrigin
    thread_ids: list[str] = Field(default_factory=list)
    contradiction_id: str | None = None

    @model_validator(mode="after")
    def _validate_reasoned_disagreement(self) -> HoldingItem:
        if (
            self.type == HoldingType.REASONED_DISAGREEMENT
            and self.contradiction_id is None
        ):
            raise ValueError(
                "type 'reasoned_disagreement' requires a non-null contradiction_id"
            )
        return self

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class HoldingStore:
    """File-backed holding store.

    ``write()`` enforces the validation rule for ``reasoned_disagreement``.
    ``forced_surface()`` returns items that have exceeded their urgency threshold.

    Not thread-safe; assumes single-writer.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / "holding.yaml")
        self._items: dict[str, HoldingItem] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            raw: list[dict[str, Any]] = yaml.safe_load(self._path.read_text()) or []
            for item in raw:
                holding = HoldingItem.model_validate(item)
                self._items[holding.id] = holding

    def _save(self) -> None:
        data = [item.model_dump(mode="json") for item in self._items.values()]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write / read
    # ------------------------------------------------------------------

    def write(self, item: HoldingItem) -> HoldingItem:
        """Persist a new holding item.

        Raises ``ValueError`` if ``type: reasoned_disagreement`` and
        ``contradiction_id`` is ``None`` (enforced by ``HoldingItem`` validator).
        """
        if item.id in self._items:
            raise ValueError(
                f"Holding item {item.id!r} already exists. "
                "Use discharge() to mark it surfaced."
            )
        self._items[item.id] = item
        self._save()
        return item

    def read(self, item_id: str) -> HoldingItem:
        """Return the item with the given ID."""
        if item_id not in self._items:
            raise KeyError(f"Holding item {item_id!r} not found")
        return self._items[item_id]

    def list_all(self) -> list[HoldingItem]:
        """Return all items (surfaced and unsurfaced)."""
        return list(self._items.values())

    def list_unsurfaced(self) -> list[HoldingItem]:
        """Return items that have not yet been discharged."""
        return [i for i in self._items.values() if i.surfaced is None]

    # ------------------------------------------------------------------
    # Discharge
    # ------------------------------------------------------------------

    def discharge(
        self, item_id: str, discharge_notes: str | None = None
    ) -> HoldingItem:
        """Mark an item as surfaced (discharged).

        Sets ``surfaced`` to the current UTC time.
        Optionally records ``discharge_notes``.
        """
        item = self.read(item_id)
        if item.surfaced is not None:
            raise ValueError(f"Holding item {item_id!r} is already discharged")
        updated = item.model_copy(
            update={
                "surfaced": _utcnow(),
                "discharge_notes": discharge_notes,
            }
        )
        self._items[item_id] = updated
        self._save()
        return updated

    # ------------------------------------------------------------------
    # Forced surface (urgency thresholds)
    # ------------------------------------------------------------------

    def forced_surface(self, now: datetime | None = None) -> list[HoldingItem]:
        """Return unsurfaced items that have exceeded their urgency threshold.

        - ``high``   → forced after 7 days
        - ``medium`` → forced after 21 days
        - ``low``    → never forced
        """
        if now is None:
            now = datetime.now(UTC)
        result: list[HoldingItem] = []
        for item in self.list_unsurfaced():
            days = _FORCED_SURFACE_DAYS[item.urgency]
            if days is None:
                continue
            created = datetime.fromisoformat(item.created)
            if now - created >= timedelta(days=days):
                result.append(item)
        return result
