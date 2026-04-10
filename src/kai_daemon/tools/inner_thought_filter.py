"""daemon_inner_thought_filter tool (§7b).

Filters a raw inner thought, producing a KEEP / BORDERLINE / DISCARD verdict.

PRIVACY INVARIANT
-----------------
This module must never import ``daemon_relational`` or any module that
carries user context.  It receives **only** ``raw_output: str`` — the
plain text of the generated thought.  This invariant is verified by
``tests/test_inner_life_privacy.py`` — do not relax it.

Bypass valve
------------
A configurable fraction of outputs (default 12 %) skip the inference
filter entirely and receive verdict ``KEEP``.  This prevents the filter
from systematically suppressing an entire class of thought.  The bypass
probability is read from ``user.yaml`` at the call site and passed in;
the default is ``DEFAULT_BYPASS_PROBABILITY``.

Verdicts
--------
``KEEP``        — enter integration routing (``daemon_integration``).
``BORDERLINE``  — added to the BORDERLINE review pool (kai-devtools only).
``DISCARD``     — dropped; no further processing.

The ``inference_fn`` receives a prompt that contains only the raw thought
text — no user data.  Its return value is expected to be one of the three
verdict strings (case-insensitive).  If the return value is unrecognised
the filter falls back to ``BORDERLINE`` and logs a warning.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from enum import StrEnum

from ..sdk import sdk_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BYPASS_PROBABILITY: float = 0.12

_FILTER_PROMPT_TEMPLATE = """\
Read the following thought. Decide whether it is worth keeping.

Thought:
{raw_output}

Reply with exactly one word: KEEP, BORDERLINE, or DISCARD.
- KEEP      — genuinely interesting, novel, or worth developing further.
- BORDERLINE — uncertain; needs human review.
- DISCARD   — incoherent, empty, or clearly not worth keeping.
"""


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------


class FilterVerdict(StrEnum):
    """Three-way classification produced by the filter."""

    KEEP = "keep"
    BORDERLINE = "borderline"
    DISCARD = "discard"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_verdict(raw: str) -> FilterVerdict:
    """Parse a model response into a ``FilterVerdict``.

    Case-insensitive.  Falls back to ``BORDERLINE`` on unrecognised input
    so that ambiguous model output is never silently discarded.
    """
    normalised = raw.strip().lower()
    try:
        return FilterVerdict(normalised)
    except ValueError:
        logger.warning(
            "Unrecognised filter response %r — defaulting to BORDERLINE", raw
        )
        return FilterVerdict.BORDERLINE


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------


@sdk_tool("daemon_inner_thought_filter")
def daemon_inner_thought_filter(
    raw_output: str,
    *,
    inference_fn: Callable[[str], str] | None = None,
    bypass_probability: float = DEFAULT_BYPASS_PROBABILITY,
    rng: random.Random | None = None,
) -> FilterVerdict:
    """Filter a raw inner thought and return a verdict.

    Parameters
    ----------
    raw_output:
        Plain text of the generated thought.  **No user context.**
    inference_fn:
        Callable that sends the filter prompt to the foundation model and
        returns the raw response string.  Required unless the bypass valve
        fires.  Injectable for testing.
    bypass_probability:
        Fraction of calls that skip the filter (verdict = KEEP).
        Default: 0.12 (12 %).  Configurable from ``user.yaml``.
    rng:
        Override random source for deterministic tests.

    Returns
    -------
    FilterVerdict
        ``KEEP``, ``BORDERLINE``, or ``DISCARD``.

    Raises
    ------
    ValueError
        If ``inference_fn`` is ``None`` and the bypass valve does not fire.
    """
    _rng = rng if rng is not None else random.Random()

    if _rng.random() < bypass_probability:
        return FilterVerdict.KEEP

    if inference_fn is None:
        raise ValueError(
            "inference_fn is required when bypass valve does not fire. "
            "Pass an injectable callable or increase bypass_probability in tests."
        )

    prompt = _FILTER_PROMPT_TEMPLATE.format(raw_output=raw_output)
    response = inference_fn(prompt)
    return _parse_verdict(response)
