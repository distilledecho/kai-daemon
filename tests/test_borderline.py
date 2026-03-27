"""Tests for BORDERLINE pool state (§7b, §2A)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai_daemon.state.borderline import (
    BORDERLINE_EXPIRY_DAYS,
    BorderlineItem,
    BorderlinePool,
    BorderlineStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _pool(tmp_path: Path) -> BorderlinePool:
    return BorderlinePool(path=tmp_path / "borderline_pool.yaml")


# ---------------------------------------------------------------------------
# Append (write)
# ---------------------------------------------------------------------------


def test_append_creates_item(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("some thought")
    assert item.raw_output == "some thought"
    assert item.status == BorderlineStatus.PENDING
    assert item.id


def test_append_persists_to_disk(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool2 = _pool(tmp_path)
    assert pool2.get(item.id).raw_output == "thought"


def test_append_multiple_items(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    ids = [pool.append(f"thought {i}").id for i in range(5)]
    assert len(pool.list_all()) == 5
    assert len(set(ids)) == 5  # all unique IDs


def test_append_only_does_not_overwrite(tmp_path: Path) -> None:
    """append() never modifies existing items."""
    pool = _pool(tmp_path)
    item = pool.append("original")
    # Re-loading should show the original unchanged
    pool2 = _pool(tmp_path)
    assert pool2.get(item.id).raw_output == "original"
    assert pool2.get(item.id).status == BorderlineStatus.PENDING


# ---------------------------------------------------------------------------
# Read / list
# ---------------------------------------------------------------------------


def test_get_raises_key_error_for_missing_id(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    with pytest.raises(KeyError, match="not found"):
        pool.get("nonexistent-id")


def test_list_pending_returns_only_pending(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item1 = pool.append("pending")
    item2 = pool.append("to discard")
    pool.discard(item2.id)
    pending = pool.list_pending()
    assert len(pending) == 1
    assert pending[0].id == item1.id


def test_list_all_returns_all_statuses(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.append("a")
    i2 = pool.append("b")
    pool.promote(i2.id)
    assert len(pool.list_all()) == 2


# ---------------------------------------------------------------------------
# Promote
# ---------------------------------------------------------------------------


def test_promote_sets_status(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    promoted = pool.promote(item.id, now=_NOW)
    assert promoted.status == BorderlineStatus.PROMOTED
    assert promoted.promoted_at is not None


def test_promote_records_timestamp(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    promoted = pool.promote(item.id, now=_NOW)
    assert promoted.promoted_at == _NOW.isoformat()


def test_promote_persists(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.promote(item.id, now=_NOW)
    pool2 = _pool(tmp_path)
    assert pool2.get(item.id).status == BorderlineStatus.PROMOTED


def test_promote_raises_if_already_promoted(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.promote(item.id)
    with pytest.raises(ValueError, match="Cannot promote"):
        pool.promote(item.id)


def test_promote_raises_if_discarded(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.discard(item.id)
    with pytest.raises(ValueError, match="Cannot promote"):
        pool.promote(item.id)


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


def test_discard_sets_status(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    discarded = pool.discard(item.id, now=_NOW)
    assert discarded.status == BorderlineStatus.DISCARDED
    assert discarded.discarded_at == _NOW.isoformat()


def test_discard_persists(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.discard(item.id)
    pool2 = _pool(tmp_path)
    assert pool2.get(item.id).status == BorderlineStatus.DISCARDED


def test_discard_raises_if_already_discarded(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.discard(item.id)
    with pytest.raises(ValueError, match="Cannot discard"):
        pool.discard(item.id)


def test_discard_raises_if_promoted(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    item = pool.append("thought")
    pool.promote(item.id)
    with pytest.raises(ValueError, match="Cannot discard"):
        pool.discard(item.id)


# ---------------------------------------------------------------------------
# Auto-expiry
# ---------------------------------------------------------------------------


def test_expire_old_discards_pending_items_over_30_days(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    old_ts = (_NOW - timedelta(days=31)).isoformat()
    # Manually construct an old item (bypass the default factory)
    old_item = BorderlineItem(raw_output="old thought", created=old_ts)
    pool._items[old_item.id] = old_item
    pool._save()

    count = pool.expire_old(now=_NOW)
    assert count == 1
    assert pool.get(old_item.id).status == BorderlineStatus.DISCARDED


def test_expire_old_keeps_items_within_30_days(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    recent_ts = (_NOW - timedelta(days=29)).isoformat()
    item = BorderlineItem(raw_output="recent", created=recent_ts)
    pool._items[item.id] = item
    pool._save()

    count = pool.expire_old(now=_NOW)
    assert count == 0
    assert pool.get(item.id).status == BorderlineStatus.PENDING


def test_expire_old_does_not_discard_promoted_items(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    old_ts = (_NOW - timedelta(days=40)).isoformat()
    item = BorderlineItem(
        raw_output="old promoted",
        created=old_ts,
        status=BorderlineStatus.PROMOTED,
        promoted_at=_NOW.isoformat(),
    )
    pool._items[item.id] = item
    pool._save()

    count = pool.expire_old(now=_NOW)
    assert count == 0
    assert pool.get(item.id).status == BorderlineStatus.PROMOTED


def test_expire_old_boundary_exactly_30_days(tmp_path: Path) -> None:
    """Item created exactly 30 days ago should be expired (created <= cutoff)."""
    pool = _pool(tmp_path)
    exact_ts = (_NOW - timedelta(days=BORDERLINE_EXPIRY_DAYS)).isoformat()
    item = BorderlineItem(raw_output="boundary", created=exact_ts)
    pool._items[item.id] = item
    pool._save()

    count = pool.expire_old(now=_NOW)
    assert count == 1


def test_expire_old_persists(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    old_ts = (_NOW - timedelta(days=35)).isoformat()
    item = BorderlineItem(raw_output="old", created=old_ts)
    pool._items[item.id] = item
    pool._save()

    pool.expire_old(now=_NOW)
    pool2 = _pool(tmp_path)
    assert pool2.get(item.id).status == BorderlineStatus.DISCARDED


def test_expire_old_returns_zero_when_nothing_to_expire(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.append("fresh thought")
    count = pool.expire_old(now=_NOW)
    assert count == 0


def test_expiry_constant_is_30() -> None:
    assert BORDERLINE_EXPIRY_DAYS == 30


# ---------------------------------------------------------------------------
# Empty pool
# ---------------------------------------------------------------------------


def test_empty_pool_list_all(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    assert pool.list_all() == []


def test_empty_pool_list_pending(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    assert pool.list_pending() == []
