"""Register inference — §8e / §4G.

Infers the conversational register from a user message.

Output: one of ``exploratory | reflective | casual | urgent``

Signals used:
    - Message content (keyword heuristics)
    - Composition time (if available)
    - Accumulated correction history for this user

The correction pathway (§4G) fires when the user signals a misread:

    1. Writes to register correction log via ``RegisterInferenceLogger``
    2. Updates the within-session relational shadow
    3. Returns an acknowledgment message — prior response is never replaced

Philosophy:
    The corrections accumulate over time. Early in the relationship the
    correction log is the primary learning mechanism. As the relationship
    deepens, accumulated relational knowledge carries more weight and
    corrections become less frequent. [philosophy:18]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .observability import RegisterCorrectionEntry, RegisterInferenceLogger

# ---------------------------------------------------------------------------
# Session relational shadow (§8c)
# ---------------------------------------------------------------------------


@dataclass
class SessionRelationalShadow:
    """Within-session accumulation of register corrections (§8c).

    Lightweight in-memory state that tracks how the user's register has
    presented during this session.  Updated whenever the correction pathway
    fires.  The ``relational_update`` workflow at session end uses this to
    write a more accurate DAEMON_RELATIONAL version.

    Attributes:
        corrections_this_session: ``(inferred_register, corrected_register)``
            pairs accumulated during this session, oldest first.

    Example::

        >>> shadow = SessionRelationalShadow()
        >>> shadow.corrections_this_session
        []
        >>> shadow.corrections_this_session.append(("casual", "reflective"))
        >>> shadow.corrections_this_session
        [('casual', 'reflective')]
    """

    corrections_this_session: list[tuple[str, str]] = field(default_factory=list)  # type: ignore[reportUnknownVariableType]
    """(inferred_register, corrected_register) pairs — oldest first."""


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------


@dataclass
class RegisterInference:
    """Result of register inference for a single turn (§8e).

    Attributes:
        register: Inferred conversational register.  One of
            ``"exploratory"``, ``"reflective"``, ``"casual"``, ``"urgent"``.
        confidence: Signal strength, 0.0–1.0.  Higher means more keyword
            and structural evidence for the chosen register.

    Example::

        >>> ri = RegisterInference(register="reflective", confidence=0.8)
        >>> ri.register
        'reflective'
        >>> ri.confidence
        0.8
    """

    register: str
    confidence: float


# ---------------------------------------------------------------------------
# Valid registers
# ---------------------------------------------------------------------------

VALID_REGISTERS: frozenset[str] = frozenset(
    {"exploratory", "reflective", "casual", "urgent"}
)
"""The four valid register values."""

# ---------------------------------------------------------------------------
# Correction history weight
# ---------------------------------------------------------------------------

_HISTORY_WEIGHT: float = 0.15
"""Per-entry penalty/boost applied when adjusting scores from correction history.

Each past correction penalises the inferred register by this amount and
boosts the corrected register by the same amount.  Referenced by both the
implementation and pinned arithmetic tests so a single constant controls both.
"""

# ---------------------------------------------------------------------------
# Keyword signal tables
# ---------------------------------------------------------------------------

_URGENT_SIGNALS: frozenset[str] = frozenset(
    {
        "urgent",
        "help",
        "broken",
        "error",
        "asap",
        "immediately",
        "critical",
        "emergency",
        "crash",
        "fail",
        "failing",
        "failed",
        "problem",
        "issue",
        "fix",
        "stuck",
        "blocked",
        "down",
    }
)

_REFLECTIVE_SIGNALS: frozenset[str] = frozenset(
    {
        "think",
        "thinking",
        "wonder",
        "wondering",  # also in _EXPLORATORY_SIGNALS — dual membership intentional
        "feel",
        "feeling",
        "believe",
        "realize",
        "realise",
        "consider",
        "considering",
        "reflect",
        "reflecting",
        "perhaps",
        "pondering",
        "ponder",
        "honestly",
        "genuinely",
        "actually",
        "struggled",
        "struggling",
        "weighing",
    }
)

_EXPLORATORY_SIGNALS: frozenset[str] = frozenset(
    {
        "what if",
        "could",
        "imagine",
        "curious",
        "curiosity",
        "explore",
        "exploring",
        "maybe",
        "might",
        "suppose",
        "hypothetical",
        "wondering",  # also in _REFLECTIVE_SIGNALS — dual membership intentional
        "brainstorm",
        "brainstorming",
        "idea",
        "ideas",
        "possibility",
        "possibilities",
    }
)

_CASUAL_SIGNALS: frozenset[str] = frozenset(
    {
        "hey",
        "hi",
        "hello",
        "yeah",
        "yep",
        "yup",
        "cool",
        "ok",
        "okay",
        "lol",
        "haha",
        "ha",
        "nice",
        "great",
        "thanks",
        "thx",
        "awesome",
        "oh",
        "ah",
        "hmm",
        "hm",
        "sure",
        "fine",
        "sounds good",
    }
)

# ---------------------------------------------------------------------------
# Signal scoring helpers
# ---------------------------------------------------------------------------


def _score_signals(tokens: set[str], text_lower: str, signals: frozenset[str]) -> float:
    """Count how many signal phrases/words appear in tokens or text_lower.

    Multi-word signals are matched against the full lowercased text.
    Single-word signals are matched against the token set.

    Args:
        tokens: Set of word tokens from the message.
        text_lower: Full message text, lowercased.
        signals: Signal vocabulary to score against.

    Returns:
        Raw signal count (float).

    Example::

        >>> tokens = {"help", "it", "is", "broken"}
        >>> _score_signals(tokens, "help it is broken", frozenset({"help", "broken"}))
        2.0
    """
    score = 0.0
    for sig in signals:
        if " " in sig:
            if sig in text_lower:
                score += 1.0
        elif sig in tokens:
            score += 1.0
    return score


def _apply_correction_history_prior(
    base_scores: dict[str, float],
    correction_history: list[RegisterCorrectionEntry],
) -> dict[str, float]:
    """Adjust base scores using accumulated correction history.

    For each past correction (inferred → corrected), penalises the
    inferred register slightly and boosts the corrected register.
    Only the most recent 20 entries are used.

    Args:
        base_scores: Raw signal scores per register.
        correction_history: All prior correction log entries.

    Returns:
        Adjusted score dict (new object — base_scores is not mutated).

    Example::

        >>> from kai_daemon.state.observability import RegisterCorrectionEntry
        >>> entry = RegisterCorrectionEntry(
        ...     inferred_register="casual",
        ...     corrected_register="reflective",
        ... )
        >>> base = {"casual": 1.0, "reflective": 0.5, "exploratory": 0.0, "urgent": 0.0}
        >>> adjusted = _apply_correction_history_prior(base, [entry])
        >>> adjusted["reflective"] > base["reflective"]
        True
        >>> adjusted["casual"] < base["casual"]
        True
    """
    if not correction_history:
        return dict(base_scores)

    recent = correction_history[-20:]
    scores: dict[str, float] = dict(base_scores)

    for entry in recent:
        inferred = entry.inferred_register
        corrected = entry.corrected_register
        if inferred in scores:
            scores[inferred] -= _HISTORY_WEIGHT
        if corrected in scores:
            scores[corrected] += _HISTORY_WEIGHT

    return scores


# ---------------------------------------------------------------------------
# Primary inference function
# ---------------------------------------------------------------------------


def infer_register(
    message: str,
    composition_seconds: float | None = None,
    correction_history: list[RegisterCorrectionEntry] | None = None,
) -> RegisterInference:
    """Infer the conversational register for a single message (§8e).

    Uses keyword heuristics, structural signals (message length, punctuation),
    composition time (if available), and accumulated correction history to score
    each of the four candidate registers.  The highest-scoring register wins.
    Defaults to ``"casual"`` when no signals are present.

    Correction history adjusts the priors: each past correction where register
    X was inferred but Y was correct slightly penalises X and boosts Y.  Only
    the most recent 20 corrections are applied to keep the heuristic fresh.

    Args:
        message: The user's message text.
        composition_seconds: Seconds between the previous message send and
            this one.  ``None`` if unavailable.
        correction_history: All prior ``RegisterCorrectionEntry`` records for
            this user.  ``None`` or empty list treats history as absent.

    Returns:
        ``RegisterInference`` with the inferred register and a confidence
        score (0.0–1.0).  Confidence reflects the proportion of total signal
        strength accounted for by the winning register.

    Example::

        >>> result = infer_register("help everything is broken")
        >>> result.register
        'urgent'
        >>> result.confidence > 0.0
        True
    """
    text_lower = message.lower()
    tokens: set[str] = set(re.findall(r"\b\w+\b", text_lower))
    word_count = len(message.split())
    has_question = "?" in message

    # --- Keyword signals ---
    urgent_score = _score_signals(tokens, text_lower, _URGENT_SIGNALS)
    reflective_score = _score_signals(tokens, text_lower, _REFLECTIVE_SIGNALS)
    exploratory_score = _score_signals(tokens, text_lower, _EXPLORATORY_SIGNALS)
    casual_score = _score_signals(tokens, text_lower, _CASUAL_SIGNALS)

    # --- Structural signals ---
    if word_count < 5:
        # Very short messages lean casual
        casual_score += 0.5
    if word_count > 50:
        # Long messages lean reflective
        reflective_score += 0.5
    if has_question and word_count < 20:
        # Short question → exploratory
        exploratory_score += 0.3
    if "!" in message:
        # Exclamation marker for urgency or strong feeling
        urgent_score += 0.3

    # --- Composition time signals ---
    if composition_seconds is not None:
        if composition_seconds < 4.0:
            # Very fast reply: lean casual
            casual_score += 0.3
        elif composition_seconds > 90.0:
            # Slow, deliberate message: lean reflective
            reflective_score += 0.5

    # --- Apply correction history prior ---
    base_scores: dict[str, float] = {
        "urgent": urgent_score,
        "reflective": reflective_score,
        "exploratory": exploratory_score,
        "casual": casual_score,
    }
    adjusted = _apply_correction_history_prior(base_scores, correction_history or [])

    # --- Select winner ---
    best_register = max(adjusted, key=lambda r: adjusted[r])
    best_score = adjusted[best_register]

    # Default to casual when all signals are zero or negative
    all_non_positive = all(v <= 0 for v in adjusted.values())
    if all_non_positive:
        best_register = "casual"
        best_score = 0.0

    # Confidence: share of total positive signal claimed by the winner
    total_positive = sum(max(v, 0.0) for v in adjusted.values())
    if total_positive > 0.0:
        confidence = min(best_score / total_positive, 0.95)
    else:
        confidence = 0.1  # minimal confidence for default-casual

    return RegisterInference(register=best_register, confidence=confidence)


# ---------------------------------------------------------------------------
# Acknowledgment messages (§4G / philosophy:18)
# ---------------------------------------------------------------------------

_ACK_MESSAGES: dict[tuple[str, str], str] = {
    (
        "casual",
        "reflective",
    ): "Oh wait — you're being serious. Let me come at this differently.",
    (
        "casual",
        "exploratory",
    ): (
        "I read that too lightly. You're actually digging into something"
        " — let me engage properly."
    ),
    (
        "casual",
        "urgent",
    ): "Oh, this is urgent. I misread the tone — what do you need?",
    (
        "exploratory",
        "reflective",
    ): "This is weighing on you more than I clocked. I hear it.",
    (
        "exploratory",
        "urgent",
    ): "This is more pressing than I caught. What's the immediate thing?",
    (
        "exploratory",
        "casual",
    ): "Lighter than I thought — got it.",
    (
        "reflective",
        "casual",
    ): "Lighter than I thought — got it.",
    (
        "reflective",
        "exploratory",
    ): "More of a live question than a settled one, I see.",
    (
        "reflective",
        "urgent",
    ): "This is urgent. I misread the weight of it — what do you need?",
    (
        "urgent",
        "reflective",
    ): "You're not in crisis mode — more thinking out loud. I'll adjust.",
    (
        "urgent",
        "casual",
    ): "Lighter than I read it. Got it.",
    (
        "urgent",
        "exploratory",
    ): "More curious than pressured. I can work with that.",
}

_ACK_DEFAULT = "I misread the tone there. Let me recalibrate."


def _acknowledgment_message(inferred_register: str, corrected_register: str) -> str:
    """Return the acknowledgment text for this correction pair.

    Falls back to a generic message if the specific pair is not in the table.

    Args:
        inferred_register: The register that was incorrectly inferred.
        corrected_register: The register the user signalled.

    Returns:
        Acknowledgment message text.

    Example::

        >>> _acknowledgment_message("casual", "reflective")
        "Oh wait — you're being serious. Let me come at this differently."
        >>> _acknowledgment_message("casual", "casual")
        'I misread the tone there. Let me recalibrate.'
    """
    return _ACK_MESSAGES.get((inferred_register, corrected_register), _ACK_DEFAULT)


# ---------------------------------------------------------------------------
# Correction pathway (§4G)
# ---------------------------------------------------------------------------


def apply_correction(
    inferred_register: str,
    corrected_register: str,
    session_shadow: SessionRelationalShadow,
    correction_logger: RegisterInferenceLogger,
    thread_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Fire the register correction pathway (§4G).

    Must be called when the user signals that the daemon misread their register.
    This function performs three actions:

    1. Appends a ``RegisterCorrectionEntry`` to the correction log
       (``data/logs/register_inference.jsonl``).  Append-only — never modifies
       prior entries.
    2. Records the correction in the within-session relational shadow so that
       subsequent turns in this session benefit immediately.
    3. Returns an acknowledgment message string.

    **The prior response is preserved.**  The caller must emit the returned
    string as a *new* message — this function does not touch, replace, or
    regenerate prior responses.

    Args:
        inferred_register: The register the daemon incorrectly inferred.
        corrected_register: The register the user signalled via correction.
        session_shadow: Within-session relational shadow to update in-place.
        correction_logger: ``RegisterInferenceLogger`` instance to write to.
        thread_id: Active thread ID at correction time, if any.
        metadata: Additional metadata to attach to the log entry.

    Raises:
        ValueError: If either register value is not in ``VALID_REGISTERS``.

    Returns:
        Acknowledgment message text — caller emits as a new message.

    Example::

        >>> import tempfile
        >>> from pathlib import Path
        >>> from kai_daemon.state.observability import RegisterInferenceLogger
        >>> with tempfile.TemporaryDirectory() as d:
        ...     logger = RegisterInferenceLogger(log_path=Path(d) / "reg.jsonl")
        ...     shadow = SessionRelationalShadow()
        ...     msg = apply_correction("casual", "reflective", shadow, logger)
        ...     print(msg)
        ...     print(shadow.corrections_this_session)
        Oh wait — you're being serious. Let me come at this differently.
        [('casual', 'reflective')]
    """
    if inferred_register not in VALID_REGISTERS:
        raise ValueError(
            f"inferred_register {inferred_register!r} is not a valid register; "
            f"expected one of {sorted(VALID_REGISTERS)}"
        )
    if corrected_register not in VALID_REGISTERS:
        raise ValueError(
            f"corrected_register {corrected_register!r} is not a valid register; "
            f"expected one of {sorted(VALID_REGISTERS)}"
        )
    entry = RegisterCorrectionEntry(
        inferred_register=inferred_register,
        corrected_register=corrected_register,
        thread_id=thread_id,
        metadata=metadata or {},
    )
    correction_logger.append(entry)
    session_shadow.corrections_this_session.append(
        (inferred_register, corrected_register)
    )
    return _acknowledgment_message(inferred_register, corrected_register)
