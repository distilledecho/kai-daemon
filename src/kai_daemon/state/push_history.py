"""Push history — 7-day ceiling enforcement (§4g).

Append-only log of out-of-session push events.

``within_ceiling()`` returns ``True`` when the last push is within the
configured ceiling window (default 7 days).  The ``inner_life_push_evaluation``
workflow calls this before running its inference prompt — the ceiling is
enforced in code, not by prompt.

File: ``data/daemon_state/push_history.yaml``
"""

from __future__ import annotations

import uuid
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._paths import daemon_state_dir
from ._utils import _utcnow

PUSH_CEILING_DAYS: int = 7
_FILENAME = "push_history.yaml"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PushRecord(BaseModel):
    """A single out-of-session push event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=_utcnow)
    content_summary: str

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PushHistoryStore:
    """Append-only file-backed push history.

    Inject ``path`` in tests to use a temporary file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / _FILENAME)
        self._records: list[PushRecord] = []
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
                f"Push history file {self._path} could not be parsed — "
                f"starting with empty history. Error: {exc}",
                stacklevel=2,
            )
            return
        self._records = [PushRecord.model_validate(r) for r in raw]

    def _save(self) -> None:
        data = [r.model_dump(mode="json") for r in self._records]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_push(self, content_summary: str) -> PushRecord:
        """Append a new push event and persist.

        Returns the created ``PushRecord``.
        """
        record = PushRecord(content_summary=content_summary)
        self._records.append(record)
        self._save()
        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def last_push_timestamp(self) -> datetime | None:
        """Return the most recent push timestamp, or ``None`` if no push occurred."""
        if not self._records:
            return None
        dt = datetime.fromisoformat(self._records[-1].timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt

    def within_ceiling(
        self,
        *,
        days: int = PUSH_CEILING_DAYS,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` if the last push is within *days* days of *now*.

        When ``True`` the push evaluation workflow must skip inference and
        return silence — the 7-day ceiling is enforced in code.
        """
        last = self.last_push_timestamp()
        if last is None:
            return False
        _now = now if now is not None else datetime.now(UTC)
        return (_now - last) < timedelta(days=days)

    def all_records(self) -> list[PushRecord]:
        """Return all push records in the order they were appended."""
        return list(self._records)
