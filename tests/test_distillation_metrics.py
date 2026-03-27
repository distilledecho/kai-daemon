"""Tests for DistillationMetricsStore (§4i)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai_daemon.state.distillation_metrics import (
    DistillationCycleRecord,
    DistillationMetricsStore,
    DistillationSignal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> DistillationMetricsStore:
    return DistillationMetricsStore(path=tmp_path / "distillation_metrics.yaml")


def _make_record(
    cycle: int, snapshot: str = "snapshot", version: int = 1
) -> DistillationCycleRecord:
    return DistillationCycleRecord(
        cycle_number=cycle,
        daemon_self_version=version,
        content_snapshot=snapshot,
    )


# ---------------------------------------------------------------------------
# DistillationSignal enum
# ---------------------------------------------------------------------------


def test_signal_values() -> None:
    assert DistillationSignal.CONVERGENCE == "convergence"
    assert DistillationSignal.FLATTERY_DRIFT == "flattery_drift"
    assert DistillationSignal.OSCILLATION == "oscillation"


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_empty_store_has_no_records(store: DistillationMetricsStore) -> None:
    assert store.all_records() == []


def test_empty_store_next_cycle_number_is_one(store: DistillationMetricsStore) -> None:
    assert store.next_cycle_number() == 1


# ---------------------------------------------------------------------------
# record_cycle
# ---------------------------------------------------------------------------


def test_record_cycle_returns_record(store: DistillationMetricsStore) -> None:
    rec = _make_record(1, "first snapshot")
    result = store.record_cycle(rec)
    assert result.cycle_number == 1
    assert result.content_snapshot == "first snapshot"


def test_record_cycle_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "dm.yaml"
    s1 = DistillationMetricsStore(path=path)
    s1.record_cycle(_make_record(1, "snap1"))

    s2 = DistillationMetricsStore(path=path)
    assert len(s2.all_records()) == 1
    assert s2.all_records()[0].cycle_number == 1


# ---------------------------------------------------------------------------
# all_records
# ---------------------------------------------------------------------------


def test_all_records_sorted_ascending(store: DistillationMetricsStore) -> None:
    store.record_cycle(_make_record(3, "third"))
    store.record_cycle(_make_record(1, "first"))
    store.record_cycle(_make_record(2, "second"))
    records = store.all_records()
    assert [r.cycle_number for r in records] == [1, 2, 3]


# ---------------------------------------------------------------------------
# load_recent
# ---------------------------------------------------------------------------


def test_load_recent_returns_last_n(store: DistillationMetricsStore) -> None:
    for i in range(1, 6):
        store.record_cycle(_make_record(i, f"snap{i}"))
    recent = store.load_recent(3)
    assert len(recent) == 3
    assert [r.cycle_number for r in recent] == [3, 4, 5]


def test_load_recent_fewer_than_n_returns_all(store: DistillationMetricsStore) -> None:
    store.record_cycle(_make_record(1))
    store.record_cycle(_make_record(2))
    recent = store.load_recent(5)
    assert len(recent) == 2


def test_load_recent_empty_store(store: DistillationMetricsStore) -> None:
    assert store.load_recent(3) == []


# ---------------------------------------------------------------------------
# next_cycle_number
# ---------------------------------------------------------------------------


def test_next_cycle_number_increments(store: DistillationMetricsStore) -> None:
    store.record_cycle(_make_record(1))
    store.record_cycle(_make_record(2))
    assert store.next_cycle_number() == 3


def test_next_cycle_number_after_gap(store: DistillationMetricsStore) -> None:
    store.record_cycle(_make_record(5))
    assert store.next_cycle_number() == 6


# ---------------------------------------------------------------------------
# Corrupt file
# ---------------------------------------------------------------------------


def test_corrupt_file_warns_and_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "dm.yaml"
    path.write_text("not: valid: yaml: [\n")
    with pytest.warns(UserWarning, match="could not be parsed"):
        bad_store = DistillationMetricsStore(path=path)
    assert bad_store.all_records() == []
