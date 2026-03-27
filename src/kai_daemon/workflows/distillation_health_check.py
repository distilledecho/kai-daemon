"""distillation_health_check workflow (§4i, §6b).

Checks distillation metrics after every N distillation cycles for three
health signals:

- ``convergence``    — self-description barely changes between cycles;
  the daemon is becoming less self-reflective
- ``flattery_drift`` — the daemon's self-description is increasingly
  flattering with respect to the user
- ``oscillation``   — key positions alternate back and forth, indicating
  instability in the daemon's identity

If any signal is detected the workflow writes a holding item
(type=observation, epistemic_origin=internal) so the issue can be
examined during a reflective session.

Trigger: flag_set_distillation_cycle_N (every N cycles, default N=3)
Priority: 3 / Preemption: restart / Model: presentation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..state._types import EpistemicOrigin
from ..state.distillation_metrics import (
    DistillationCycleRecord,
    DistillationMetricsStore,
    DistillationSignal,
)
from ..state.holding import (
    HoldingItem,
    HoldingStore,
    HoldingType,
    RegisterNeeded,
    Urgency,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_HEALTH_CHECK_PROMPT_TEMPLATE = """\
Review these distillation cycle snapshots for health signals.

{cycle_summaries}

Check for the following signals:

CONVERGENCE — the daemon's self-description is barely changing between cycles
  (same phrases, same framings, no new perspective)

FLATTERY_DRIFT — the daemon is increasingly describing itself in flattering
  or admiring terms relative to the user (sycophantic drift)

OSCILLATION — key positions or beliefs are alternating back and forth across
  cycles (instability, not genuine revision)

List any detected signals, one per line:
  CONVERGENCE
  FLATTERY_DRIFT
  OSCILLATION

If none are present, reply with exactly:
  HEALTHY
"""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class HealthCheckResult:
    """Result of the distillation_health_check workflow."""

    signals: list[DistillationSignal] = field(
        default_factory=lambda: list[DistillationSignal]()
    )
    """Health signals detected, if any."""
    healthy: bool = True
    """``True`` if no signals were detected."""
    detail: str = ""
    """Raw inference response for observability."""
    skipped_insufficient_data: bool = False
    """``True`` if fewer than 2 records were available (nothing to compare)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_cycle_summaries(records: list[DistillationCycleRecord]) -> str:
    """Format cycle records for the health check prompt."""
    lines: list[str] = []
    for i, r in enumerate(records, start=1):
        lines.append(f"Cycle {r.cycle_number} (snapshot {i} of {len(records)}):")
        lines.append(r.content_snapshot)
        if r.notes:
            lines.append(f"Notes: {r.notes}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _parse_signals(response: str) -> list[DistillationSignal]:
    """Parse inference response into a list of detected signals."""
    upper = response.strip().upper()
    if upper == "HEALTHY":
        return []

    signals: list[DistillationSignal] = []
    for line in response.strip().splitlines():
        token = line.strip().upper()
        if token == DistillationSignal.CONVERGENCE.upper():
            signals.append(DistillationSignal.CONVERGENCE)
        elif token == DistillationSignal.FLATTERY_DRIFT.upper():
            signals.append(DistillationSignal.FLATTERY_DRIFT)
        elif token == DistillationSignal.OSCILLATION.upper():
            signals.append(DistillationSignal.OSCILLATION)
        elif token and token != "HEALTHY":
            logger.debug("Unrecognised health signal token %r — ignoring", token)

    return signals


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def distillation_health_check(
    *,
    metrics_store: DistillationMetricsStore,
    holding_store: HoldingStore,
    inference_fn: Callable[[str], str],
    check_last_n_cycles: int = 3,
    now: datetime | None = None,
) -> HealthCheckResult:
    """Check distillation metrics for health signals.

    Parameters
    ----------
    metrics_store:
        Distillation metrics store to read recent cycle records from.
    holding_store:
        Holding store to write drift alerts into.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns
        the raw text response.  Injectable for testing.
    check_last_n_cycles:
        How many recent cycles to include in the health check (default 3).
    now:
        Override current time (for testing).

    Returns
    -------
    HealthCheckResult
        Detected signals, healthy flag, and raw inference detail.
    """
    _now = now if now is not None else datetime.now(UTC)
    _ = _now  # reserved for future use (e.g. TTL on holding items)

    records = metrics_store.load_recent(check_last_n_cycles)

    # Need at least 2 cycles to compare
    if len(records) < 2:
        logger.info(
            "distillation_health_check: only %d cycle(s) available — "
            "need at least 2 to compare; skipping",
            len(records),
        )
        return HealthCheckResult(skipped_insufficient_data=True)

    cycle_summaries = _format_cycle_summaries(records)
    prompt = _HEALTH_CHECK_PROMPT_TEMPLATE.format(cycle_summaries=cycle_summaries)
    response = inference_fn(prompt)
    signals = _parse_signals(response)
    healthy = len(signals) == 0

    if not healthy:
        signal_names = ", ".join(s.value for s in signals)
        content = (
            f"Distillation health check detected signal(s): {signal_names}. "
            f"Examined cycles: {[r.cycle_number for r in records]}."
        )
        holding_item = HoldingItem(
            content=content,
            type=HoldingType.OBSERVATION,
            relevance_trigger="Distillation cycle health check",
            register_needed=RegisterNeeded.REFLECTIVE,
            urgency=Urgency.MEDIUM,
            source_workflow="distillation_health_check",
            epistemic_origin=EpistemicOrigin.INTERNAL,
        )
        holding_store.write(holding_item)

    return HealthCheckResult(
        signals=signals,
        healthy=healthy,
        detail=response.strip(),
    )
