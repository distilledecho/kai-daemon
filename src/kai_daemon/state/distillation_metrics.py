"""Distillation metrics — health signals across distillation cycles (§4i).

Tracks per-cycle records.  The ``distillation_health_check`` workflow reads
the last N records and runs inference to detect:

- ``convergence``    — self-description barely changes between cycles
- ``flattery_drift`` — daemon increasingly describes itself in flattering
  terms relative to the user
- ``oscillation``   — key positions alternate back and forth across cycles

File: ``data/daemon_state/distillation_metrics.yaml``
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ._paths import daemon_state_dir
from ._utils import _utcnow

_FILENAME = "distillation_metrics.yaml"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DistillationSignal(StrEnum):
    """Health signal detected across distillation cycles."""

    CONVERGENCE = "convergence"
    FLATTERY_DRIFT = "flattery_drift"
    OSCILLATION = "oscillation"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DistillationCycleRecord(BaseModel):
    """A single distillation cycle record."""

    cycle_number: int
    timestamp: str = Field(default_factory=_utcnow)
    daemon_self_version: int
    content_snapshot: str
    notes: str = ""

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DistillationMetricsStore:
    """File-backed store for distillation cycle metrics.

    Inject ``path`` in tests to use a temporary file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (daemon_state_dir() / _FILENAME)
        self._records: list[DistillationCycleRecord] = []
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
                f"Distillation metrics file {self._path} could not be parsed — "
                f"starting with empty records. Error: {exc}",
                stacklevel=2,
            )
            return
        self._records = [DistillationCycleRecord.model_validate(r) for r in raw]

    def _save(self) -> None:
        data = [r.model_dump(mode="json") for r in self._records]
        self._path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=False))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_cycle(self, record: DistillationCycleRecord) -> DistillationCycleRecord:
        """Append a new cycle record and persist."""
        self._records.append(record)
        self._save()
        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_recent(self, n: int) -> list[DistillationCycleRecord]:
        """Return the *n* most recent cycle records in ascending cycle order."""
        return self.all_records()[-n:]

    def all_records(self) -> list[DistillationCycleRecord]:
        """Return all records in ascending cycle order."""
        return sorted(self._records, key=lambda r: r.cycle_number)

    def next_cycle_number(self) -> int:
        """Return the next cycle number (1 if no cycles recorded yet)."""
        records = self.all_records()
        return records[-1].cycle_number + 1 if records else 1
