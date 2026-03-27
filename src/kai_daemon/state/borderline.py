"""BORDERLINE pool state (§7b, §2A acceptance criteria).

Stores inner-thought outputs that received a BORDERLINE filter verdict,
pending human review via the kai-devtools panel.

Invariants
----------
- **Append-only at write time** — ``append()`` only adds; it never modifies
  or removes existing items.
- **Promote / discard** — ``promote(id)`` and ``discard(id)`` mark items as
  ``promoted`` or ``discarded`` respectively; they do not delete rows.
- **Auto-expiry** — ``expire_old(now)`` discards all ``pending`` items whose
  ``created`` timestamp is more than ``BORDERLINE_EXPIRY_DAYS`` days old.
  This runs nightly as part of ``transcript_pruning``.
- **Review surface is kai-devtools only** — nothing here surfaces through
  the conversation layer.

File format
-----------
YAML list of ``BorderlineItem`` dicts at the path returned by
``_paths.daemon_state_dir() / "borderline_pool.yaml"``.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._paths import daemon_state_dir
from ._utils import _utcnow

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BORDERLINE_EXPIRY_DAYS: int = 30

_FILENAME = "borderline_pool.yaml"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BorderlineStatus(StrEnum):
    """Lifecycle state of a BORDERLINE pool item."""

    PENDING = "pending"
    PROMOTED = "promoted"
    DISCARDED = "discarded"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BorderlineItem(BaseModel):
    """A single item in the BORDERLINE review pool."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_output: str
    created: str = Field(default_factory=_utcnow)
    status: BorderlineStatus = BorderlineStatus.PENDING
    promoted_at: str | None = None
    discarded_at: str | None = None

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class BorderlinePool:
    """File-backed BORDERLINE review pool.

    Thread-safety: single-writer assumed (matches rest of daemon state layer).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / _FILENAME)
        self._items: dict[str, BorderlineItem] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                raw: list[dict[str, Any]] = yaml.safe_load(self._path.read_text()) or []
            except yaml.YAMLError as exc:
                warnings.warn(
                    f"BORDERLINE pool file {self._path} is corrupt and could not "
                    f"be parsed — starting with an empty pool. Error: {exc}",
                    stacklevel=2,
                )
                return
            for row in raw:
                item = BorderlineItem.model_validate(row)
                self._items[item.id] = item

    def _save(self) -> None:
        data = [item.model_dump(mode="json") for item in self._items.values()]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write (append-only)
    # ------------------------------------------------------------------

    def append(self, raw_output: str) -> BorderlineItem:
        """Append a new BORDERLINE item and persist.

        Returns the created ``BorderlineItem``.
        """
        item = BorderlineItem(raw_output=raw_output)
        self._items[item.id] = item
        self._save()
        return item

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, item_id: str) -> BorderlineItem:
        """Return the item with *item_id*.

        Raises ``KeyError`` if not found.
        """
        if item_id not in self._items:
            raise KeyError(f"BORDERLINE item {item_id!r} not found")
        return self._items[item_id]

    def list_pending(self) -> list[BorderlineItem]:
        """Return all items with status ``pending``."""
        return [i for i in self._items.values() if i.status == BorderlineStatus.PENDING]

    def list_all(self) -> list[BorderlineItem]:
        """Return all items regardless of status."""
        return list(self._items.values())

    # ------------------------------------------------------------------
    # Actions (promote / discard)
    # ------------------------------------------------------------------

    def promote(self, item_id: str, *, now: datetime | None = None) -> BorderlineItem:
        """Mark item as ``promoted`` (enters integration routing).

        Raises ``KeyError`` if not found.
        Raises ``ValueError`` if already promoted or discarded.
        """
        item = self.get(item_id)
        if item.status != BorderlineStatus.PENDING:
            raise ValueError(
                f"Cannot promote item {item_id!r} — "
                f"current status is {item.status!r} (expected 'pending')"
            )
        _now = now or datetime.now(UTC)
        updated = item.model_copy(
            update={
                "status": BorderlineStatus.PROMOTED,
                "promoted_at": _now.isoformat(),
            }
        )
        self._items[item_id] = updated
        self._save()
        return updated

    def discard(self, item_id: str, *, now: datetime | None = None) -> BorderlineItem:
        """Mark item as ``discarded``.

        Raises ``KeyError`` if not found.
        Raises ``ValueError`` if already promoted or discarded.
        """
        item = self.get(item_id)
        if item.status != BorderlineStatus.PENDING:
            raise ValueError(
                f"Cannot discard item {item_id!r} — "
                f"current status is {item.status!r} (expected 'pending')"
            )
        _now = now or datetime.now(UTC)
        updated = item.model_copy(
            update={
                "status": BorderlineStatus.DISCARDED,
                "discarded_at": _now.isoformat(),
            }
        )
        self._items[item_id] = updated
        self._save()
        return updated

    # ------------------------------------------------------------------
    # Auto-expiry
    # ------------------------------------------------------------------

    def expire_old(self, *, now: datetime | None = None) -> int:
        """Discard all ``pending`` items older than ``BORDERLINE_EXPIRY_DAYS`` days.

        Returns the count of items expired.  Runs nightly as part of
        ``transcript_pruning``.
        """
        _now = now or datetime.now(UTC)
        cutoff = _now - timedelta(days=BORDERLINE_EXPIRY_DAYS)
        expired = 0
        for item in list(self._items.values()):
            if item.status != BorderlineStatus.PENDING:
                continue
            try:
                created_dt = datetime.fromisoformat(item.created)
            except ValueError:
                warnings.warn(
                    f"BORDERLINE item {item.id!r} has unparseable 'created' "
                    f"field {item.created!r} — skipping expiry check",
                    stacklevel=2,
                )
                continue
            if created_dt <= cutoff:
                updated = item.model_copy(
                    update={
                        "status": BorderlineStatus.DISCARDED,
                        "discarded_at": _now.isoformat(),
                    }
                )
                self._items[item.id] = updated
                expired += 1
        if expired:
            self._save()
        return expired
