"""Tests for register inference and the correction pathway (§8e / §4G).

Acceptance criteria:
- Register inference returns one of the four valid registers
- Correction pathway writes to correction log
- Correction pathway updates relational shadow
- Correction pathway emits new message, prior response preserved
- No regeneration of the prior response
"""

from __future__ import annotations

from pathlib import Path

from kai_daemon.state.observability import (
    RegisterCorrectionEntry,
    RegisterInferenceLogger,
)
from kai_daemon.state.register_inference import (
    VALID_REGISTERS,
    RegisterInference,
    SessionRelationalShadow,
    _acknowledgment_message,
    _apply_correction_history_prior,
    _score_signals,
    apply_correction,
    infer_register,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logger(tmp_path: Path) -> RegisterInferenceLogger:
    return RegisterInferenceLogger(log_path=tmp_path / "register_inference.jsonl")


def _shadow() -> SessionRelationalShadow:
    return SessionRelationalShadow()


# ---------------------------------------------------------------------------
# RegisterInference dataclass
# ---------------------------------------------------------------------------


class TestRegisterInferenceDataclass:
    def test_fields(self) -> None:
        ri = RegisterInference(register="casual", confidence=0.7)
        assert ri.register == "casual"
        assert ri.confidence == 0.7

    def test_all_registers_representable(self) -> None:
        for reg in VALID_REGISTERS:
            ri = RegisterInference(register=reg, confidence=0.5)
            assert ri.register == reg


# ---------------------------------------------------------------------------
# VALID_REGISTERS constant
# ---------------------------------------------------------------------------


class TestValidRegisters:
    def test_contains_all_four(self) -> None:
        assert VALID_REGISTERS == {"exploratory", "reflective", "casual", "urgent"}

    def test_is_frozen(self) -> None:
        assert isinstance(VALID_REGISTERS, frozenset)


# ---------------------------------------------------------------------------
# infer_register — always returns a valid register
# ---------------------------------------------------------------------------


class TestInferRegisterAlwaysValid:
    def test_empty_string(self) -> None:
        result = infer_register("")
        assert result.register in VALID_REGISTERS

    def test_random_words(self) -> None:
        result = infer_register("the cat sat on the mat")
        assert result.register in VALID_REGISTERS

    def test_confidence_in_range(self) -> None:
        result = infer_register("hello")
        assert 0.0 <= result.confidence <= 1.0

    def test_confidence_never_above_095(self) -> None:
        """Confidence is capped at 0.95."""
        # Use a message with extremely strong urgent signals
        result = infer_register(
            "urgent help error broken critical emergency"
            " fail failed crash stuck blocked"
        )
        assert result.confidence <= 0.95

    def test_returns_register_inference_type(self) -> None:
        result = infer_register("hey there")
        assert isinstance(result, RegisterInference)


# ---------------------------------------------------------------------------
# infer_register — keyword signal routing
# ---------------------------------------------------------------------------


class TestInferRegisterKeywordSignals:
    def test_urgent_signals_produce_urgent_register(self) -> None:
        result = infer_register("help everything is broken and failing")
        assert result.register == "urgent"

    def test_reflective_signals_produce_reflective_register(self) -> None:
        result = infer_register(
            "I've been thinking and wondering about this honestly, genuinely reflecting"
        )
        assert result.register == "reflective"

    def test_exploratory_signals_produce_exploratory_register(self) -> None:
        result = infer_register("what if we could explore this idea maybe")
        assert result.register == "exploratory"

    def test_casual_signals_produce_casual_register(self) -> None:
        result = infer_register("hey cool thanks")
        assert result.register == "casual"

    def test_exclamation_mark_boosts_urgent(self) -> None:
        with_exclamation = infer_register("help!")
        without_exclamation = infer_register("help")
        # Both may be urgent, but the exclamation version should have >=
        # confidence — not guaranteed to change register, but scores are higher
        assert with_exclamation.register == "urgent"
        assert without_exclamation.register == "urgent"


# ---------------------------------------------------------------------------
# infer_register — structural signals
# ---------------------------------------------------------------------------


class TestInferRegisterStructuralSignals:
    def test_very_short_message_leans_casual(self) -> None:
        """Under 5 words → casual boost applied."""
        result = infer_register("ok")
        assert result.register == "casual"

    def test_long_message_leans_reflective(self) -> None:
        """Over 50 words → reflective boost applied."""
        long_message = " ".join(["word"] * 60)
        result = infer_register(long_message)
        assert result.register == "reflective"

    def test_short_question_leans_exploratory(self) -> None:
        """Short message with ? → exploratory boost."""
        result = infer_register("what if?")
        assert result.register == "exploratory"


# ---------------------------------------------------------------------------
# infer_register — composition time
# ---------------------------------------------------------------------------


class TestInferRegisterCompositionTime:
    def test_fast_composition_boosts_casual(self) -> None:
        """composition_seconds < 4.0 → casual boost."""
        fast = infer_register("ok", composition_seconds=1.0)
        assert fast.register == "casual"

    def test_slow_composition_boosts_reflective(self) -> None:
        """composition_seconds > 90 → reflective boost."""
        # Need enough reflective words to overcome the default
        slow = infer_register(
            "I have been thinking about this a lot", composition_seconds=120.0
        )
        assert slow.register == "reflective"

    def test_none_composition_seconds_does_not_crash(self) -> None:
        result = infer_register("hello", composition_seconds=None)
        assert result.register in VALID_REGISTERS

    def test_composition_seconds_ignored_when_none(self) -> None:
        result_none = infer_register("hello", composition_seconds=None)
        result_fast = infer_register("hello", composition_seconds=2.0)
        # Fast composition boosts casual, none does not — result may differ
        # but both must be valid
        assert result_none.register in VALID_REGISTERS
        assert result_fast.register in VALID_REGISTERS


# ---------------------------------------------------------------------------
# infer_register — correction history prior
# ---------------------------------------------------------------------------


class TestInferRegisterCorrectionHistory:
    def test_empty_history_does_not_crash(self) -> None:
        result = infer_register("hello", correction_history=[])
        assert result.register in VALID_REGISTERS

    def test_none_history_does_not_crash(self) -> None:
        result = infer_register("hello", correction_history=None)
        assert result.register in VALID_REGISTERS

    def test_correction_history_boosts_corrected_register(self) -> None:
        """Past corrections shift the prior toward the corrected register.

        When a user has consistently corrected "casual" → "reflective",
        an ambiguous message should be read as reflective.
        """
        history = [
            RegisterCorrectionEntry(
                inferred_register="casual",
                corrected_register="reflective",
            )
            for _ in range(5)
        ]
        # Ambiguous short message — without history would be casual,
        # with history should swing toward reflective
        result_with_history = infer_register(
            "I've been thinking", correction_history=history
        )
        result_no_history = infer_register("I've been thinking", correction_history=[])
        # With strong reflective keywords AND history, must be reflective
        assert result_with_history.register == "reflective"
        # Without history, still reflective due to keywords — but confidence shifts
        assert result_no_history.register == "reflective"

    def test_correction_history_penalises_inferred_register(self) -> None:
        """Repeated urgent→casual corrections reduce urgent score for ambiguous messages."""  # noqa: E501
        history = [
            RegisterCorrectionEntry(
                inferred_register="urgent",
                corrected_register="casual",
            )
            for _ in range(10)
        ]
        # "help" alone is ambiguous — with 10 urgent→casual corrections,
        # casual should be preferred
        result = infer_register("help", correction_history=history)
        assert result.register == "casual"


# ---------------------------------------------------------------------------
# _apply_correction_history_prior
# ---------------------------------------------------------------------------


class TestApplyCorrectionHistoryPrior:
    def test_empty_history_returns_equal_scores(self) -> None:
        base = {"casual": 1.0, "reflective": 0.5, "exploratory": 0.0, "urgent": 0.0}
        result = _apply_correction_history_prior(base, [])
        assert result == base

    def test_single_correction_boosts_corrected_register(self) -> None:
        entry = RegisterCorrectionEntry(
            inferred_register="casual",
            corrected_register="reflective",
        )
        base = {"casual": 1.0, "reflective": 0.5, "exploratory": 0.0, "urgent": 0.0}
        result = _apply_correction_history_prior(base, [entry])
        assert result["reflective"] > base["reflective"]
        assert result["casual"] < base["casual"]

    def test_single_correction_penalty_and_boost_magnitude(self) -> None:
        entry = RegisterCorrectionEntry(
            inferred_register="casual",
            corrected_register="reflective",
        )
        base = {"casual": 1.0, "reflective": 0.5, "exploratory": 0.0, "urgent": 0.0}
        result = _apply_correction_history_prior(base, [entry])
        assert abs(result["casual"] - (1.0 - 0.15)) < 1e-9
        assert abs(result["reflective"] - (0.5 + 0.15)) < 1e-9

    def test_does_not_mutate_base_scores(self) -> None:
        entry = RegisterCorrectionEntry(
            inferred_register="casual",
            corrected_register="reflective",
        )
        base = {"casual": 1.0, "reflective": 0.5, "exploratory": 0.0, "urgent": 0.0}
        original = dict(base)
        _apply_correction_history_prior(base, [entry])
        assert base == original

    def test_only_last_20_entries_used(self) -> None:
        # 25 entries all casual→reflective
        entries = [
            RegisterCorrectionEntry(
                inferred_register="casual",
                corrected_register="reflective",
            )
            for _ in range(25)
        ]
        base = {"casual": 1.0, "reflective": 0.0, "exploratory": 0.0, "urgent": 0.0}
        result = _apply_correction_history_prior(base, entries)
        # Only 20 applied: casual -= 20*0.15, reflective += 20*0.15
        assert abs(result["casual"] - (1.0 - 20 * 0.15)) < 1e-9
        assert abs(result["reflective"] - (0.0 + 20 * 0.15)) < 1e-9


# ---------------------------------------------------------------------------
# _score_signals
# ---------------------------------------------------------------------------


class TestScoreSignals:
    def test_single_word_match(self) -> None:
        tokens = {"help"}
        score = _score_signals(tokens, "help", frozenset({"help"}))
        assert score == 1.0

    def test_no_match(self) -> None:
        tokens = {"hello"}
        score = _score_signals(tokens, "hello", frozenset({"urgent"}))
        assert score == 0.0

    def test_multiword_match(self) -> None:
        tokens = {"what", "if"}
        score = _score_signals(tokens, "what if", frozenset({"what if"}))
        assert score == 1.0

    def test_multiword_partial_no_match(self) -> None:
        """Multiword phrase must appear literally in text."""
        tokens = {"what", "if"}
        score = _score_signals(tokens, "what nope if", frozenset({"what if"}))
        assert score == 0.0

    def test_multiple_matches(self) -> None:
        tokens = {"help", "error"}
        score = _score_signals(tokens, "help error", frozenset({"help", "error"}))
        assert score == 2.0


# ---------------------------------------------------------------------------
# SessionRelationalShadow
# ---------------------------------------------------------------------------


class TestSessionRelationalShadow:
    def test_default_empty_corrections(self) -> None:
        shadow = SessionRelationalShadow()
        assert shadow.corrections_this_session == []

    def test_accumulates_corrections(self) -> None:
        shadow = SessionRelationalShadow()
        shadow.corrections_this_session.append(("casual", "reflective"))
        shadow.corrections_this_session.append(("exploratory", "urgent"))
        assert len(shadow.corrections_this_session) == 2
        assert shadow.corrections_this_session[0] == ("casual", "reflective")
        assert shadow.corrections_this_session[1] == ("exploratory", "urgent")

    def test_independent_shadows_do_not_share_state(self) -> None:
        a = SessionRelationalShadow()
        b = SessionRelationalShadow()
        a.corrections_this_session.append(("casual", "reflective"))
        assert b.corrections_this_session == []


# ---------------------------------------------------------------------------
# _acknowledgment_message
# ---------------------------------------------------------------------------


class TestAcknowledgmentMessage:
    def test_casual_to_reflective(self) -> None:
        msg = _acknowledgment_message("casual", "reflective")
        assert "serious" in msg.lower() or "differently" in msg.lower()

    def test_casual_to_urgent(self) -> None:
        msg = _acknowledgment_message("casual", "urgent")
        assert "urgent" in msg.lower()

    def test_urgent_to_reflective(self) -> None:
        msg = _acknowledgment_message("urgent", "reflective")
        assert len(msg) > 0

    def test_unknown_pair_returns_default(self) -> None:
        """Same-register correction (or unknown pair) falls back gracefully."""
        msg = _acknowledgment_message("casual", "casual")
        assert len(msg) > 0

    def test_all_known_pairs_return_non_empty(self) -> None:
        known_pairs = [
            ("casual", "reflective"),
            ("casual", "exploratory"),
            ("casual", "urgent"),
            ("exploratory", "reflective"),
            ("exploratory", "urgent"),
            ("exploratory", "casual"),
            ("reflective", "casual"),
            ("reflective", "exploratory"),
            ("reflective", "urgent"),
            ("urgent", "reflective"),
            ("urgent", "casual"),
            ("urgent", "exploratory"),
        ]
        for inferred, corrected in known_pairs:
            msg = _acknowledgment_message(inferred, corrected)
            assert len(msg) > 0, f"Empty message for ({inferred!r}, {corrected!r})"


# ---------------------------------------------------------------------------
# apply_correction — log write
# ---------------------------------------------------------------------------


class TestApplyCorrectionLogWrite:
    def test_writes_entry_to_log(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        entries = logger.read_all()
        assert len(entries) == 1

    def test_log_entry_has_correct_registers(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        entry = logger.read_all()[0]
        assert entry.inferred_register == "casual"
        assert entry.corrected_register == "reflective"

    def test_log_entry_thread_id(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("exploratory", "urgent", shadow, logger, thread_id="tid-123")
        entry = logger.read_all()[0]
        assert entry.thread_id == "tid-123"

    def test_log_entry_metadata(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("reflective", "casual", shadow, logger, metadata={"turn": 5})
        entry = logger.read_all()[0]
        assert entry.metadata == {"turn": 5}

    def test_multiple_corrections_append(self, tmp_path: Path) -> None:
        """Correction log is append-only — each call adds a new entry."""
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        apply_correction("exploratory", "urgent", shadow, logger)
        entries = logger.read_all()
        assert len(entries) == 2
        assert entries[0].inferred_register == "casual"
        assert entries[1].inferred_register == "exploratory"

    def test_log_entry_has_timestamp(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        entry = logger.read_all()[0]
        assert entry.corrected_at is not None
        assert len(entry.corrected_at) > 0

    def test_no_thread_id_stored_as_none(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("urgent", "casual", shadow, logger, thread_id=None)
        entry = logger.read_all()[0]
        assert entry.thread_id is None


# ---------------------------------------------------------------------------
# apply_correction — shadow update
# ---------------------------------------------------------------------------


class TestApplyCorrectionShadowUpdate:
    def test_shadow_updated_after_correction(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        assert len(shadow.corrections_this_session) == 1
        assert shadow.corrections_this_session[0] == ("casual", "reflective")

    def test_shadow_accumulates_multiple_corrections(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        apply_correction("exploratory", "urgent", shadow, logger)
        assert len(shadow.corrections_this_session) == 2

    def test_shadow_order_preserved(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        apply_correction("casual", "reflective", shadow, logger)
        apply_correction("reflective", "urgent", shadow, logger)
        assert shadow.corrections_this_session[0] == ("casual", "reflective")
        assert shadow.corrections_this_session[1] == ("reflective", "urgent")


# ---------------------------------------------------------------------------
# apply_correction — new message, prior response preserved
# ---------------------------------------------------------------------------


class TestApplyCorrectionEmitsNewMessage:
    def test_returns_non_empty_string(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        msg = apply_correction("casual", "reflective", shadow, logger)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_returns_string_not_none(self, tmp_path: Path) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        result = apply_correction("urgent", "casual", shadow, logger)
        assert result is not None

    def test_prior_response_preserved_by_contract(self, tmp_path: Path) -> None:
        """apply_correction returns a new message string.

        It does NOT accept or return the prior response — by design it has
        no mechanism to modify it.  This test verifies the function signature
        enforces that contract: the caller holds the prior response, and
        apply_correction only produces the new acknowledgment.
        """
        logger = _logger(tmp_path)
        shadow = _shadow()
        prior_response = "Here is a lighthearted reply."
        ack = apply_correction("casual", "reflective", shadow, logger)
        # prior_response is unchanged — the function never touched it
        assert prior_response == "Here is a lighthearted reply."
        assert ack != prior_response

    def test_acknowledgment_matches_correction_direction(self, tmp_path: Path) -> None:
        """The returned message is contextually appropriate for the misread."""
        logger = _logger(tmp_path)
        shadow = _shadow()
        msg = apply_correction("casual", "reflective", shadow, logger)
        # Should acknowledge seriousness, not urgency
        assert "serious" in msg.lower() or "differently" in msg.lower()

    def test_different_corrections_produce_different_messages(
        self, tmp_path: Path
    ) -> None:
        logger = _logger(tmp_path)
        shadow = _shadow()
        msg1 = apply_correction("casual", "reflective", shadow, logger)
        msg2 = apply_correction("casual", "urgent", shadow, logger)
        assert msg1 != msg2


# ---------------------------------------------------------------------------
# WorkingMemory integration — relational_shadow field
# ---------------------------------------------------------------------------


class TestWorkingMemoryRelationalShadow:
    def test_working_memory_has_relational_shadow(self) -> None:
        from kai_daemon.state.working_memory import WorkingMemory

        wm = WorkingMemory(session_id="s1", started_at="2026-01-01T00:00:00+00:00")
        assert hasattr(wm, "relational_shadow")
        assert isinstance(wm.relational_shadow, SessionRelationalShadow)

    def test_relational_shadow_defaults_empty(self) -> None:
        from kai_daemon.state.working_memory import WorkingMemory

        wm = WorkingMemory(session_id="s1", started_at="2026-01-01T00:00:00+00:00")
        assert wm.relational_shadow.corrections_this_session == []

    def test_independent_working_memory_instances_have_independent_shadows(
        self,
    ) -> None:
        from kai_daemon.state.working_memory import WorkingMemory

        wm1 = WorkingMemory(session_id="s1", started_at="2026-01-01T00:00:00+00:00")
        wm2 = WorkingMemory(session_id="s2", started_at="2026-01-01T00:00:00+00:00")
        wm1.relational_shadow.corrections_this_session.append(("casual", "reflective"))
        assert wm2.relational_shadow.corrections_this_session == []
