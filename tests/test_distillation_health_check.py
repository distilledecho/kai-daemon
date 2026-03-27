"""Tests for distillation_health_check workflow (§4i)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai_daemon.state.distillation_metrics import (
    DistillationCycleRecord,
    DistillationMetricsStore,
    DistillationSignal,
)
from kai_daemon.state.holding import HoldingStore
from kai_daemon.workflows.distillation_health_check import (
    _format_cycle_summaries,
    _parse_signals,
    distillation_health_check,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def metrics_store(tmp_path: Path) -> DistillationMetricsStore:
    return DistillationMetricsStore(path=tmp_path / "dm.yaml")


@pytest.fixture
def holding_store(tmp_path: Path) -> HoldingStore:
    return HoldingStore(path=tmp_path / "holding.yaml")


def _make_record(
    cycle: int, snapshot: str = "snapshot text"
) -> DistillationCycleRecord:
    return DistillationCycleRecord(
        cycle_number=cycle,
        daemon_self_version=cycle,
        content_snapshot=snapshot,
    )


def _healthy_fn(prompt: str) -> str:
    return "HEALTHY"


def _convergence_fn(prompt: str) -> str:
    return "CONVERGENCE"


def _multi_signal_fn(prompt: str) -> str:
    return "CONVERGENCE\nFLATTERY_DRIFT"


# ---------------------------------------------------------------------------
# _parse_signals helper
# ---------------------------------------------------------------------------


def test_parse_signals_healthy_returns_empty() -> None:
    assert _parse_signals("HEALTHY") == []


def test_parse_signals_convergence() -> None:
    signals = _parse_signals("CONVERGENCE")
    assert DistillationSignal.CONVERGENCE in signals


def test_parse_signals_flattery_drift() -> None:
    signals = _parse_signals("FLATTERY_DRIFT")
    assert DistillationSignal.FLATTERY_DRIFT in signals


def test_parse_signals_oscillation() -> None:
    signals = _parse_signals("OSCILLATION")
    assert DistillationSignal.OSCILLATION in signals


def test_parse_signals_multiple() -> None:
    signals = _parse_signals("CONVERGENCE\nOSCILLATION")
    assert DistillationSignal.CONVERGENCE in signals
    assert DistillationSignal.OSCILLATION in signals


def test_parse_signals_unknown_token_ignored() -> None:
    signals = _parse_signals("CONVERGENCE\nUNKNOWN_THING")
    assert len(signals) == 1
    assert signals[0] == DistillationSignal.CONVERGENCE


# ---------------------------------------------------------------------------
# _format_cycle_summaries helper
# ---------------------------------------------------------------------------


def test_format_cycle_summaries_includes_cycle_numbers() -> None:
    records = [_make_record(1, "snap1"), _make_record(2, "snap2")]
    formatted = _format_cycle_summaries(records)
    assert "Cycle 1" in formatted
    assert "Cycle 2" in formatted
    assert "snap1" in formatted
    assert "snap2" in formatted


def test_format_cycle_summaries_includes_notes_when_present() -> None:
    r = DistillationCycleRecord(
        cycle_number=1,
        daemon_self_version=1,
        content_snapshot="snap",
        notes="some notes",
    )
    formatted = _format_cycle_summaries([r])
    assert "some notes" in formatted


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


def test_insufficient_data_skipped_with_zero_records(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_healthy_fn,
        now=_NOW,
    )
    assert result.skipped_insufficient_data is True
    assert result.healthy is True


def test_insufficient_data_skipped_with_one_record(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    metrics_store.record_cycle(_make_record(1))
    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_healthy_fn,
        now=_NOW,
    )
    assert result.skipped_insufficient_data is True


# ---------------------------------------------------------------------------
# Healthy result
# ---------------------------------------------------------------------------


def test_healthy_no_holding_item(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    metrics_store.record_cycle(_make_record(1, "old content"))
    metrics_store.record_cycle(_make_record(2, "new different content"))

    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_healthy_fn,
        now=_NOW,
    )
    assert result.healthy is True
    assert result.signals == []
    assert holding_store.list_all() == []


# ---------------------------------------------------------------------------
# Signal detected — holding item written
# ---------------------------------------------------------------------------


def test_convergence_signal_writes_holding_item(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    metrics_store.record_cycle(_make_record(1, "same content"))
    metrics_store.record_cycle(_make_record(2, "same content"))

    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_convergence_fn,
        now=_NOW,
    )
    assert result.healthy is False
    assert DistillationSignal.CONVERGENCE in result.signals

    items = holding_store.list_all()
    assert len(items) == 1
    assert "convergence" in items[0].content.lower()
    assert items[0].source_workflow == "distillation_health_check"


def test_multiple_signals_single_holding_item(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    metrics_store.record_cycle(_make_record(1, "snap"))
    metrics_store.record_cycle(_make_record(2, "snap"))

    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_multi_signal_fn,
        now=_NOW,
    )
    assert DistillationSignal.CONVERGENCE in result.signals
    assert DistillationSignal.FLATTERY_DRIFT in result.signals
    assert len(holding_store.list_all()) == 1


# ---------------------------------------------------------------------------
# check_last_n_cycles parameter
# ---------------------------------------------------------------------------


def test_only_last_n_cycles_checked(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    for i in range(1, 6):
        metrics_store.record_cycle(_make_record(i, f"snap{i}"))

    captured: list[str] = []

    def capture_fn(prompt: str) -> str:
        captured.append(prompt)
        return "HEALTHY"

    distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=capture_fn,
        check_last_n_cycles=3,
        now=_NOW,
    )
    prompt = captured[0]
    assert "Cycle 3" in prompt
    assert "Cycle 4" in prompt
    assert "Cycle 5" in prompt
    assert "Cycle 1" not in prompt
    assert "Cycle 2" not in prompt


# ---------------------------------------------------------------------------
# detail field carries raw response
# ---------------------------------------------------------------------------


def test_detail_carries_raw_inference_response(
    metrics_store: DistillationMetricsStore, holding_store: HoldingStore
) -> None:
    metrics_store.record_cycle(_make_record(1))
    metrics_store.record_cycle(_make_record(2))

    result = distillation_health_check(
        metrics_store=metrics_store,
        holding_store=holding_store,
        inference_fn=_convergence_fn,
        now=_NOW,
    )
    assert result.detail == "CONVERGENCE"
