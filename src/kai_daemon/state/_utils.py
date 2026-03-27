"""Shared utilities for daemon-local state modules."""

from __future__ import annotations

from datetime import UTC, datetime


def _utcnow() -> str:  # pyright: ignore[reportUnusedFunction]
    return datetime.now(UTC).isoformat()
