"""Tests for PushHistoryStore (§4g)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai_daemon.state.push_history import (
    PUSH_CEILING_DAYS,
    PushHistoryStore,
    PushRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> PushHistoryStore:
    return PushHistoryStore(path=tmp_path / "push_history.yaml")


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_empty_store_has_no_records(store: PushHistoryStore) -> None:
    assert store.all_records() == []


def test_empty_store_last_push_timestamp_is_none(store: PushHistoryStore) -> None:
    assert store.last_push_timestamp() is None


def test_empty_store_within_ceiling_false(store: PushHistoryStore) -> None:
    assert store.within_ceiling() is False


# ---------------------------------------------------------------------------
# record_push
# ---------------------------------------------------------------------------


def test_record_push_returns_record(store: PushHistoryStore) -> None:
    record = store.record_push("test push")
    assert record.content_summary == "test push"
    assert record.id
    assert record.timestamp


def test_record_push_appears_in_all_records(store: PushHistoryStore) -> None:
    store.record_push("push one")
    store.record_push("push two")
    records = store.all_records()
    assert len(records) == 2
    assert records[0].content_summary == "push one"
    assert records[1].content_summary == "push two"


def test_record_push_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "push_history.yaml"
    s1 = PushHistoryStore(path=path)
    s1.record_push("persisted push")

    s2 = PushHistoryStore(path=path)
    assert len(s2.all_records()) == 1
    assert s2.all_records()[0].content_summary == "persisted push"


# ---------------------------------------------------------------------------
# last_push_timestamp
# ---------------------------------------------------------------------------


def test_last_push_timestamp_returns_most_recent(store: PushHistoryStore) -> None:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 1, 5, tzinfo=UTC)
    r1 = store.record_push("first")
    store._records = [
        PushRecord(id=r1.id, timestamp=base.isoformat(), content_summary="first"),
        PushRecord(timestamp=later.isoformat(), content_summary="second"),
    ]
    store._save()
    ts = store.last_push_timestamp()
    assert ts is not None
    assert ts == later


def test_last_push_timestamp_tz_aware(store: PushHistoryStore) -> None:
    store.record_push("tz test")
    ts = store.last_push_timestamp()
    assert ts is not None
    assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# within_ceiling
# ---------------------------------------------------------------------------


def test_within_ceiling_true_when_push_within_window(store: PushHistoryStore) -> None:
    now = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)
    recent = now - timedelta(days=3)
    store._records = [
        PushRecord(timestamp=recent.isoformat(), content_summary="recent push")
    ]
    store._save()
    assert store.within_ceiling(days=7, now=now) is True


def test_within_ceiling_false_when_push_outside_window(store: PushHistoryStore) -> None:
    now = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)
    old = now - timedelta(days=10)
    store._records = [PushRecord(timestamp=old.isoformat(), content_summary="old push")]
    store._save()
    assert store.within_ceiling(days=7, now=now) is False


def test_within_ceiling_exactly_at_boundary(store: PushHistoryStore) -> None:
    now = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)
    exactly = now - timedelta(days=7)
    store._records = [
        PushRecord(timestamp=exactly.isoformat(), content_summary="boundary push")
    ]
    store._save()
    assert store.within_ceiling(days=7, now=now) is False


def test_within_ceiling_default_days_is_push_ceiling_days() -> None:
    assert PUSH_CEILING_DAYS == 7


# ---------------------------------------------------------------------------
# Corrupt file handling
# ---------------------------------------------------------------------------


def test_corrupt_file_warns_and_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "push_history.yaml"
    path.write_text("not: valid: yaml: [\n")
    with pytest.warns(UserWarning, match="could not be parsed"):
        bad_store = PushHistoryStore(path=path)
    assert bad_store.all_records() == []
