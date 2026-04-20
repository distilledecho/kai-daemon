"""personal_assistant workflow (§8, §8a–§8g).

Priority 1. Preemption: suspend. Trigger: message_received.

Per-turn sequence
-----------------
1. Infer register from the incoming message.
2. Run conversational_retrieval with current thread stack.
3. Respond — presence first (philosophy §12); retrieval serves the response.
4. Write a turn note to working memory (after response — user never waits).
5. Update thread stack salience.
6. Check for a discharge candidate; surface naturally if found.
7. If register correction is signalled, run the correction pathway.

Session start
-------------
Load DAEMON_SELF, DAEMON_RELATIONAL into working memory context.
Initialise ``SessionRelationalShadow``.

Session end
-----------
Call ``run_session_end``.  Clear working memory only when
``flush_succeeded`` is ``True``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from ..state.daemon_relational import DaemonRelational, DaemonRelationalStore
from ..state.daemon_self import DaemonSelf, DaemonSelfStore
from ..state.discharge import (
    ContradictionClientProtocol,
    ContradictionRecord,
    hydrate_contradiction,
    select_discharge_candidate,
)
from ..state.holding import HoldingItem, HoldingStore
from ..state.observability import RegisterCorrectionEntry, RegisterInferenceLogger
from ..state.register_inference import (
    RegisterInference,
    SessionRelationalShadow,
    apply_correction,
    infer_register,
)
from ..state.retrieval import (
    MemoryClientProtocol,
    RetrievalContext,
    conversational_retrieval,
)
from ..state.thread_stack import (
    SalienceConfig,
    ThreadStackEntry,
    update_stack,
)
from ..state.threads import ThreadStore
from ..state.working_memory import TurnNote, WorkingMemory
from .session_end import SessionEndResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUBTEXT_PRIMING = (
    "Before responding, consider: is this the thing they are actually"
    " thinking about, or is it adjacent to it? You know this person."
    " If there is a gap between what was said and what might be going"
    " on underneath it, you can hold both — respond to what was said,"
    " but from a position of awareness rather than literalism."
)

# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------

InferenceFn = Callable[[str], str]
"""``(prompt: str) → str`` — calls the local model."""

ScoreDischargeItemsFn = Callable[[str, list[HoldingItem]], dict[str, float]]
"""``(message, items) → {item.id: similarity_score}``."""

SessionEndFn = Callable[[WorkingMemory, datetime], SessionEndResult]
"""``(working_memory, ended_at) → SessionEndResult``."""

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """Result of a single conversation turn.

    The caller emits messages in this order:
    1. ``response`` — the primary response.
    2. ``discharge_message`` — if not ``None``, a second message surfacing
       a held item.
    3. ``correction_message`` — if not ``None``, an acknowledgment of a
       register misread (emitted only when ``correction_triggered`` is
       ``True``).

    Attributes:
        response: Primary response text.
        register: Inferred register for this turn.
        register_confidence: Confidence of the inference (0.0–1.0).
        discharge_surfaced: True if a holding item was discharged.
        discharge_message: Surface text for the discharged item, or ``None``.
        correction_triggered: True if the register correction pathway fired.
        correction_message: Acknowledgment text, or ``None``.

    Example::

        >>> r = TurnResult(
        ...     response="Sure, let's explore that.",
        ...     register="exploratory",
        ...     register_confidence=0.8,
        ...     discharge_surfaced=False,
        ...     discharge_message=None,
        ...     correction_triggered=False,
        ...     correction_message=None,
        ... )
        >>> r.discharge_surfaced
        False
    """

    response: str
    register: str
    register_confidence: float
    discharge_surfaced: bool
    discharge_message: str | None
    correction_triggered: bool
    correction_message: str | None


# ---------------------------------------------------------------------------
# Spec §4B — stance movement detection
# ---------------------------------------------------------------------------


def detect_stance_movements(
    pre_turn_stances: dict[str, str],
    thread_stack: list[ThreadStackEntry],
    thread_store: ThreadStore,
) -> list[str]:
    """Return thread IDs whose epistemic status changed during this turn.

    Compares pre-turn snapshot of ``epistemic_status`` against current
    thread store state.  Returns IDs that shifted.

    Args:
        pre_turn_stances: Mapping of thread_id → epistemic_status captured
            before the turn.
        thread_stack: Non-floating stack entries at turn write time.
        thread_store: Current thread store.

    Returns:
        List of thread IDs whose stance changed.

    Example::

        >>> detect_stance_movements({}, [], ThreadStore.__new__(ThreadStore))
        []
    """
    movements: list[str] = []
    for entry in thread_stack:
        try:
            thread = thread_store.load(entry.thread_id)
        except KeyError:
            continue
        prior = pre_turn_stances.get(entry.thread_id)
        if prior is not None and thread.stance.epistemic_status != prior:
            movements.append(entry.thread_id)
    return movements


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "been",
        "being",
        "between",
        "could",
        "every",
        "going",
        "have",
        "here",
        "just",
        "know",
        "like",
        "more",
        "much",
        "only",
        "other",
        "really",
        "some",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "think",
        "this",
        "those",
        "thought",
        "through",
        "very",
        "want",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
        "from",
        "does",
        "into",
        "will",
        "were",
        "well",
    }
)


def _extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract 2–5 significant topic words from *text*.

    Uses word frequency heuristics — words longer than 3 characters that
    are not stop words.  Returns up to *max_topics* most common terms.

    Args:
        text: Source text (message or combined message + response).
        max_topics: Maximum number of topics to return.

    Returns:
        List of topic words, most frequent first.

    Example::

        >>> topics = _extract_topics("thinking about memory systems and retrieval")
        >>> "memory" in topics
        True
        >>> len(topics) <= 5
        True
    """
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    counts: Counter[str] = Counter(w for w in words if w not in _STOP_WORDS)
    return [word for word, _ in counts.most_common(max_topics)]


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(
    daemon_self: DaemonSelf | None,
    daemon_relational: DaemonRelational | None,
    thread_stack: list[ThreadStackEntry],
    thread_store: ThreadStore,
    retrieval_ctx: RetrievalContext,
) -> str:
    """Build the per-turn system prompt.

    Includes daemon identity, relational context, thread stack state,
    retrieval results, and the fixed subtext awareness priming paragraph.

    The subtext priming paragraph is included at all register levels.

    Args:
        daemon_self: Loaded DAEMON_SELF, or ``None`` if unavailable.
        daemon_relational: Loaded DAEMON_RELATIONAL, or ``None``.
        thread_stack: Non-floating stack entries for this turn.
        thread_store: For loading thread titles and questions.
        retrieval_ctx: Retrieval results for this turn.

    Returns:
        System prompt string.

    Example::

        >>> from kai_daemon.state.daemon_self import DaemonSelf
        >>> ts = ThreadStore.__new__(ThreadStore)
        >>> prompt = _build_system_prompt(None, None, [], ts, RetrievalContext())
        >>> _SUBTEXT_PRIMING in prompt
        True
    """
    parts: list[str] = []

    # Daemon identity
    if daemon_self and daemon_self.who_daemon_is:
        parts.append(f"Who you are:\n{daemon_self.who_daemon_is}")

    # Relational context
    if daemon_relational:
        rel_parts: list[str] = []
        if daemon_relational.how_user_thinks:
            rel_parts.append(f"How they think: {daemon_relational.how_user_thinks}")
        if daemon_relational.what_user_is_working_on:
            rel_parts.append(
                f"What they're working on: {daemon_relational.what_user_is_working_on}"
            )
        if daemon_relational.users_current_register:
            rel_parts.append(
                f"Their register lately: {daemon_relational.users_current_register}"
            )
        if rel_parts:
            parts.append("Relational context:\n" + "\n".join(rel_parts))

    # Thread stack context
    if thread_stack:
        thread_lines: list[str] = []
        for entry in thread_stack:
            try:
                thread = thread_store.load(entry.thread_id)
                thread_lines.append(
                    f"- [{entry.state}] {thread.title}: {thread.central_question}"
                )
            except KeyError:
                thread_lines.append(f"- [{entry.state}] thread {entry.thread_id}")
        parts.append("Active threads:\n" + "\n".join(thread_lines))

    # Retrieval context
    if retrieval_ctx.semantic:
        snippets: list[str] = []
        for result in retrieval_ctx.semantic[:5]:
            snippet = result.text[:300].replace("\n", " ")
            snippets.append(f"- [{result.space}] {snippet}")
        parts.append("Relevant context:\n" + "\n".join(snippets))

    # Pending artifacts — acknowledge naturally
    if retrieval_ctx.has_pending:
        titles = [
            r.metadata.get("title", r.document_id)
            for r in retrieval_ctx.pending_artifacts
        ]
        plural = "These artifacts are" if len(titles) > 1 else "This artifact is"
        titles_str = ", ".join(str(t) for t in titles)
        parts.append(
            f"Note: {plural} still being processed — you can acknowledge"
            f" naturally ('still reading through it') if referenced:"
            f" {titles_str}"
        )

    # Fixed subtext awareness priming (all register levels)
    parts.append(_SUBTEXT_PRIMING)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Turn note helpers
# ---------------------------------------------------------------------------


def _is_notable(
    stance_movements: list[str],
    discharge_surfaced: bool,
    correction_triggered: bool,
    message: str,
    response: str,
) -> bool:
    """Heuristic judgment — is this turn worth preserving in episodic memory?

    Most turns are not notable.  The bar is meaningful, not generous.
    """
    if stance_movements:
        return True
    if discharge_surfaced:
        return True
    if correction_triggered:
        return True
    # Long substantive exchange
    if len(message.split()) + len(response.split()) > 150:
        return True
    return False


def _notable_note(
    inference_fn: InferenceFn,
    message: str,
    response: str,
    reasons: list[str],
) -> str:
    """Generate a short prose note for a notable turn via inference.

    Args:
        inference_fn: Inference callable.
        message: User message (truncated).
        response: Daemon response (truncated).
        reasons: Human-readable reasons why this turn is notable.

    Returns:
        1–2 sentence prose note.
    """
    reasons_text = "; ".join(reasons) if reasons else "substantive exchange"
    prompt = (
        "Summarise this conversation turn in 1-2 sentences for future memory.\n"
        f"Notable because: {reasons_text}\n\n"
        f"User: {message[:400]}\n\n"
        f"Response: {response[:300]}\n\n"
        "Summary (1-2 sentences only):"
    )
    return inference_fn(prompt).strip()


# ---------------------------------------------------------------------------
# PersonalAssistant
# ---------------------------------------------------------------------------


class PersonalAssistant:
    """Per-session conversation handler wiring all Stage 4 components (§8).

    Parameters
    ----------
    inference_fn:
        Callable ``(prompt: str) → str``.  Used for response generation
        and notable turn note synthesis.
    memory_client:
        Async client for conversational retrieval.  ``None`` → retrieval
        returns empty context gracefully (daemon proceeds from local state).
    holding_store:
        Holding store for discharge candidates.
    thread_store:
        Thread store for stack context and stance movement detection.
    daemon_self_store:
        Store for loading DAEMON_SELF at session start.
    daemon_relational_store:
        Store for loading DAEMON_RELATIONAL at session start.
    register_inference_logger:
        Append-only log for register corrections.
    salience_config:
        Salience computation constants from ``user.yaml``.
    discharge_threshold:
        Similarity gate for discharge candidates (default 0.72).
    correction_history:
        Prior register correction entries loaded at session start.
    score_discharge_items_fn:
        ``(message, items) → {item.id: score}`` — computes similarity
        between the current message and each item's ``relevance_trigger``.
    session_end_fn:
        ``(working_memory, ended_at) → SessionEndResult`` — executes the
        session end sequence (§4I).

    Example::

        >>> pa = PersonalAssistant.__new__(PersonalAssistant)
        >>> pa._working_memory = None
        >>> pa._working_memory is None
        True
    """

    def __init__(
        self,
        inference_fn: InferenceFn,
        memory_client: MemoryClientProtocol | None,
        holding_store: HoldingStore,
        thread_store: ThreadStore,
        daemon_self_store: DaemonSelfStore,
        daemon_relational_store: DaemonRelationalStore,
        register_inference_logger: RegisterInferenceLogger,
        salience_config: SalienceConfig,
        discharge_threshold: float,
        correction_history: list[RegisterCorrectionEntry],
        score_discharge_items_fn: ScoreDischargeItemsFn,
        session_end_fn: SessionEndFn,
    ) -> None:
        self._inference_fn = inference_fn
        self._memory_client = memory_client
        self._holding_store = holding_store
        self._thread_store = thread_store
        self._daemon_self_store = daemon_self_store
        self._daemon_relational_store = daemon_relational_store
        self._register_inference_logger = register_inference_logger
        self._salience_config = salience_config
        self._discharge_threshold = discharge_threshold
        self._correction_history = correction_history
        self._score_discharge_items_fn = score_discharge_items_fn
        self._session_end_fn = session_end_fn

        # State initialised by begin_session
        self._working_memory: WorkingMemory | None = None
        self._daemon_self: DaemonSelf | None = None
        self._daemon_relational: DaemonRelational | None = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def begin_session(self) -> WorkingMemory:
        """Initialise a new session (session start sequence).

        Loads DAEMON_SELF and DAEMON_RELATIONAL into session context.
        Initialises a fresh ``WorkingMemory`` with a new session ID.

        Must be called before :meth:`handle_turn`.

        Returns:
            The initialised ``WorkingMemory`` for this session.

        Example::

            >>> import tempfile
            >>> from pathlib import Path
            >>> from kai_daemon.state.daemon_self import DaemonSelfStore
            >>> from kai_daemon.state.daemon_relational import DaemonRelationalStore
            >>> from kai_daemon.state.holding import HoldingStore
            >>> from kai_daemon.state.threads import ThreadStore
            >>> from kai_daemon.state.observability import RegisterInferenceLogger
            >>> from kai_daemon.state.thread_stack import SalienceConfig
            >>> def _se_ok(wm, dt):
            ...     return SessionEndResult(
            ...         session_id=wm.session_id, flush_succeeded=True
            ...     )
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dp = Path(d)
            ...     hs = HoldingStore(dp / "h.yaml")
            ...     ts = ThreadStore(dp / "t", dp / "p")
            ...     ds = DaemonSelfStore(dp, dp / "dsh")
            ...     dr = DaemonRelationalStore(dp, dp / "drh")
            ...     rl = RegisterInferenceLogger(dp / "r.jsonl")
            ...     pa = PersonalAssistant(
            ...         inference_fn=lambda p: "ok",
            ...         memory_client=None,
            ...         holding_store=hs,
            ...         thread_store=ts,
            ...         daemon_self_store=ds,
            ...         daemon_relational_store=dr,
            ...         register_inference_logger=rl,
            ...         salience_config=SalienceConfig(),
            ...         discharge_threshold=0.72,
            ...         correction_history=[],
            ...         score_discharge_items_fn=lambda m, i: {},
            ...         session_end_fn=_se_ok,
            ...     )
            ...     wm = pa.begin_session()
            ...     wm.turn_count
            0
        """
        session_id = str(uuid.uuid4())
        started_at = datetime.now(UTC).isoformat()

        self._daemon_self = self._daemon_self_store.load()
        self._daemon_relational = self._daemon_relational_store.load()

        self._working_memory = WorkingMemory(
            session_id=session_id,
            started_at=started_at,
            relational_shadow=SessionRelationalShadow(),
        )

        logger.info("personal_assistant: session started session=%s", session_id)
        return self._working_memory

    def end_session(self) -> SessionEndResult:
        """Run the session end sequence (§4I).

        Calls the injected ``session_end_fn`` with the current working
        memory.  Clears working memory only when ``flush_succeeded`` is
        ``True`` — if the flush fails, working memory is retained for
        retry on reconnection.

        Returns:
            ``SessionEndResult`` — the caller must inspect
            ``flush_succeeded`` before assuming memory was cleared.

        Raises:
            RuntimeError: If ``begin_session`` was not called first.

        Example::

            >>> import tempfile
            >>> from pathlib import Path
            >>> from kai_daemon.state.daemon_self import DaemonSelfStore
            >>> from kai_daemon.state.daemon_relational import DaemonRelationalStore
            >>> from kai_daemon.state.holding import HoldingStore
            >>> from kai_daemon.state.threads import ThreadStore
            >>> from kai_daemon.state.observability import RegisterInferenceLogger
            >>> from kai_daemon.state.thread_stack import SalienceConfig
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dp = Path(d)
            ...     pa = PersonalAssistant(
            ...         inference_fn=lambda p: "ok",
            ...         memory_client=None,
            ...         holding_store=HoldingStore(dp / "holding.yaml"),
            ...         thread_store=ThreadStore(
            ...             threads_path=dp / "threads",
            ...             pickup_notes_path=dp / "pn",
            ...         ),
            ...         daemon_self_store=DaemonSelfStore(dp),
            ...         daemon_relational_store=DaemonRelationalStore(dp),
            ...         register_inference_logger=RegisterInferenceLogger(
            ...             dp / "reg.jsonl"
            ...         ),
            ...         salience_config=SalienceConfig(),
            ...         discharge_threshold=0.72,
            ...         correction_history=[],
            ...         score_discharge_items_fn=lambda m, items: {},
            ...         session_end_fn=lambda wm, dt: SessionEndResult(
            ...             session_id=wm.session_id, flush_succeeded=True
            ...         ),
            ...     )
            ...     _ = pa.begin_session()
            ...     result = pa.end_session()
            ...     result.flush_succeeded
            True
        """
        if self._working_memory is None:
            raise RuntimeError("begin_session() must be called before end_session()")

        session_id = self._working_memory.session_id
        ended_at = datetime.now(UTC)

        logger.info(
            "personal_assistant: session ending session=%s turns=%d",
            session_id,
            self._working_memory.turn_count,
        )

        result = self._session_end_fn(self._working_memory, ended_at)

        if result.flush_succeeded:
            self._working_memory = None
            logger.info(
                "personal_assistant: working memory cleared session=%s", session_id
            )
        else:
            logger.warning(
                "personal_assistant: flush did not confirm — "
                "working memory retained session=%s",
                session_id,
            )

        return result

    # ------------------------------------------------------------------
    # Per-turn handler
    # ------------------------------------------------------------------

    async def handle_turn(
        self,
        message: str,
        composition_seconds: float | None = None,
        correction_signal: tuple[str, str] | None = None,
    ) -> TurnResult:
        """Process a single conversation turn (§8, per-turn sequence).

        Steps (executed in order):

        1. Infer register from *message*.
        2. Run ``conversational_retrieval`` for context.
        3. Generate response (presence first; retrieval serves response).
        4. Write ``TurnNote`` to working memory.
        5. Update thread stack salience.
        6. Check for discharge candidate; surface if found.
        7. Fire correction pathway if *correction_signal* provided.

        Args:
            message: The user's incoming message.
            composition_seconds: Time since previous message in seconds,
                used to refine register inference.  ``None`` if unavailable.
            correction_signal: ``(inferred_register, corrected_register)``
                pair signalling that the previous turn's register was
                misread.  ``None`` if no correction.

        Returns:
            ``TurnResult`` with response text and secondary messages.

        Raises:
            RuntimeError: If ``begin_session`` was not called first.

        Example::

            >>> import asyncio
            >>> import tempfile
            >>> from pathlib import Path
            >>> from kai_daemon.state.daemon_self import DaemonSelfStore
            >>> from kai_daemon.state.daemon_relational import DaemonRelationalStore
            >>> from kai_daemon.state.holding import HoldingStore
            >>> from kai_daemon.state.threads import ThreadStore
            >>> from kai_daemon.state.observability import RegisterInferenceLogger
            >>> from kai_daemon.state.thread_stack import SalienceConfig
            >>> with tempfile.TemporaryDirectory() as d:
            ...     dp = Path(d)
            ...     pa = PersonalAssistant(
            ...         inference_fn=lambda p: "hello there",
            ...         memory_client=None,
            ...         holding_store=HoldingStore(dp / "holding.yaml"),
            ...         thread_store=ThreadStore(
            ...             threads_path=dp / "threads",
            ...             pickup_notes_path=dp / "pn",
            ...         ),
            ...         daemon_self_store=DaemonSelfStore(dp),
            ...         daemon_relational_store=DaemonRelationalStore(dp),
            ...         register_inference_logger=RegisterInferenceLogger(
            ...             dp / "reg.jsonl"
            ...         ),
            ...         salience_config=SalienceConfig(),
            ...         discharge_threshold=0.72,
            ...         correction_history=[],
            ...         score_discharge_items_fn=lambda m, items: {},
            ...         session_end_fn=lambda wm, dt: SessionEndResult(
            ...             session_id=wm.session_id, flush_succeeded=True
            ...         ),
            ...     )
            ...     _ = pa.begin_session()
            ...     result = asyncio.run(pa.handle_turn("hey there"))
            ...     result.response
            'hello there'
        """
        if self._working_memory is None:
            raise RuntimeError("begin_session() must be called before handle_turn()")

        wm = self._working_memory
        wm.turn_count += 1
        turn_number = wm.turn_count

        # ------------------------------------------------------------------
        # Step 1: Infer register
        # ------------------------------------------------------------------
        inferred: RegisterInference = infer_register(
            message,
            composition_seconds=composition_seconds,
            correction_history=self._correction_history,
        )
        register = inferred.register
        logger.debug(
            "personal_assistant: register=%s confidence=%.2f turn=%d",
            register,
            inferred.confidence,
            turn_number,
        )

        # ------------------------------------------------------------------
        # Step 2: Conversational retrieval (async context loading, no inference)
        # ------------------------------------------------------------------
        retrieval_ctx: RetrievalContext
        if self._memory_client is not None:
            retrieval_ctx = await conversational_retrieval(
                message=message,
                thread_stack=wm.thread_stack,
                memory_client=self._memory_client,
                thread_store=self._thread_store,
            )
        else:
            retrieval_ctx = RetrievalContext()

        # ------------------------------------------------------------------
        # Step 3: Respond (presence first — retrieval serves the response)
        # ------------------------------------------------------------------
        system_prompt = _build_system_prompt(
            self._daemon_self,
            self._daemon_relational,
            wm.thread_stack,
            self._thread_store,
            retrieval_ctx,
        )
        full_prompt = f"{system_prompt}\n\nUser: {message}\n\nResponse:"
        response = self._inference_fn(full_prompt)

        # ------------------------------------------------------------------
        # Step 4: Write turn note (after response — user never waits)
        # ------------------------------------------------------------------

        # Capture pre-turn stances for stance movement detection
        pre_turn_stances: dict[str, str] = {}
        for entry in wm.thread_stack:
            try:
                thread = self._thread_store.load(entry.thread_id)
                pre_turn_stances[entry.thread_id] = str(thread.stance.epistemic_status)
            except KeyError:
                pass

        stance_movements = detect_stance_movements(
            pre_turn_stances,
            wm.thread_stack,
            self._thread_store,
        )

        topics = _extract_topics(message + " " + response)
        artifact_refs: list[str] = [
            str(r.metadata["artifact_id"])
            for r in retrieval_ctx.semantic
            if r.metadata.get("artifact_id")
        ]

        # Record pending artifacts in session artifacts list
        for pending in retrieval_ctx.pending_artifacts:
            artifact_id = pending.metadata.get("artifact_id")
            if artifact_id and str(artifact_id) not in wm.artifacts_this_session:
                wm.artifacts_this_session.append(str(artifact_id))

        notable_reasons: list[str] = []
        if stance_movements:
            notable_reasons.append(f"stance shifted on {stance_movements}")
        # discharge not yet determined at this point; will be set after step 6

        note_text: str | None = None

        turn_note = TurnNote(
            turn_id=f"{wm.session_id}:{turn_number}",
            session_id=wm.session_id,
            turn_number=turn_number,
            timestamp=datetime.now(UTC).isoformat(),
            thread_ids_active=[e.thread_id for e in wm.thread_stack],
            register=register,
            register_corrected=False,  # updated in step 7 if correction fires
            topics_touched=topics,
            stance_movements=stance_movements,
            artifacts_referenced=artifact_refs,
            notable=False,  # finalised after step 6
            note=None,
        )

        # ------------------------------------------------------------------
        # Step 5: Update thread stack salience
        # ------------------------------------------------------------------
        referenced_ids: set[str] = set()
        # Treat any thread whose central_question appears in retrieval results
        # as referenced; threads directly mentioned in message/response are
        # also candidates.  For the initial implementation, we infer references
        # from retrieval result metadata.
        for result in retrieval_ctx.semantic:
            tid = result.metadata.get("thread_id")
            if tid:
                referenced_ids.add(str(tid))

        wm.thread_stack, wm.floating_threads = update_stack(
            stack=wm.thread_stack,
            floating_threads=wm.floating_threads,
            current_turn=turn_number,
            referenced_thread_ids=referenced_ids,
            stance_movement_ids=set(stance_movements),
            config=self._salience_config,
        )

        # ------------------------------------------------------------------
        # Step 6: Check for discharge candidate; surface naturally if found
        # ------------------------------------------------------------------
        discharge_surfaced = False
        discharge_message: str | None = None

        unsurfaced_items = self._holding_store.list_unsurfaced()
        if unsurfaced_items:
            scores = self._score_discharge_items_fn(message, unsurfaced_items)
            candidate = select_discharge_candidate(
                unsurfaced_items,
                register,
                scores,
                threshold=self._discharge_threshold,
            )
            if candidate is not None:
                # Hydrate contradiction record if needed before surfacing
                contradiction_record: ContradictionRecord | None = None
                if (
                    candidate.contradiction_id is not None
                    and self._memory_client is not None
                    and isinstance(self._memory_client, ContradictionClientProtocol)
                ):
                    contradiction_record = await hydrate_contradiction(
                        candidate, self._memory_client
                    )

                discharge_message = _format_discharge(candidate, contradiction_record)
                discharge_surfaced = True

                # Mark as surfaced in the holding store
                self._holding_store.discharge(candidate.id)

                # Record in working memory
                if candidate.contradiction_id:
                    wm.contradictions_surfaced.append(candidate.id)

                notable_reasons.append("holding item discharged")
                logger.info(
                    "personal_assistant: discharged item=%s register=%s turn=%d",
                    candidate.id,
                    register,
                    turn_number,
                )

        # ------------------------------------------------------------------
        # Finalise turn note (now that discharge outcome is known)
        # ------------------------------------------------------------------
        notable = _is_notable(
            stance_movements, discharge_surfaced, False, message, response
        )
        if notable:
            note_text = _notable_note(
                self._inference_fn, message, response, notable_reasons
            )

        # Patch the turn note with final values
        turn_note = TurnNote(
            turn_id=turn_note.turn_id,
            session_id=turn_note.session_id,
            turn_number=turn_note.turn_number,
            timestamp=turn_note.timestamp,
            thread_ids_active=turn_note.thread_ids_active,
            register=turn_note.register,
            register_corrected=False,  # updated in step 7 if correction fires
            topics_touched=turn_note.topics_touched,
            stance_movements=turn_note.stance_movements,
            artifacts_referenced=turn_note.artifacts_referenced,
            notable=notable,
            note=note_text,
        )

        # ------------------------------------------------------------------
        # Step 7: Register correction pathway
        # ------------------------------------------------------------------
        correction_triggered = False
        correction_message: str | None = None

        if correction_signal is not None:
            prev_inferred, corrected = correction_signal
            try:
                correction_message = apply_correction(
                    inferred_register=prev_inferred,
                    corrected_register=corrected,
                    session_shadow=wm.relational_shadow,
                    correction_logger=self._register_inference_logger,
                    thread_id=(
                        wm.thread_stack[0].thread_id if wm.thread_stack else None
                    ),
                )
                correction_triggered = True

                # Refresh correction history from the logger
                self._correction_history = self._register_inference_logger.read_all()

                # Patch turn note to record correction
                turn_note = TurnNote(
                    turn_id=turn_note.turn_id,
                    session_id=turn_note.session_id,
                    turn_number=turn_note.turn_number,
                    timestamp=turn_note.timestamp,
                    thread_ids_active=turn_note.thread_ids_active,
                    register=turn_note.register,
                    register_corrected=True,
                    topics_touched=turn_note.topics_touched,
                    stance_movements=turn_note.stance_movements,
                    artifacts_referenced=turn_note.artifacts_referenced,
                    notable=True,
                    note=turn_note.note
                    or f"Register correction: {prev_inferred} → {corrected}",
                )

                logger.info(
                    "personal_assistant: correction fired %r → %r turn=%d",
                    prev_inferred,
                    corrected,
                    turn_number,
                )
            except ValueError:
                logger.warning(
                    "personal_assistant: invalid correction_signal %r — ignoring",
                    correction_signal,
                    exc_info=True,
                )

        # Commit turn note to working memory
        wm.turn_notes.append(turn_note)

        return TurnResult(
            response=response,
            register=register,
            register_confidence=inferred.confidence,
            discharge_surfaced=discharge_surfaced,
            discharge_message=discharge_message,
            correction_triggered=correction_triggered,
            correction_message=correction_message,
        )


# ---------------------------------------------------------------------------
# Discharge formatting
# ---------------------------------------------------------------------------


def _format_discharge(
    item: HoldingItem,
    contradiction_record: ContradictionRecord | None,
) -> str:
    """Format a holding item as a natural surface message.

    When a contradiction record is available, the conflict summary is
    included.  Otherwise, the item's own content is surfaced.

    Args:
        item: The discharge candidate.
        contradiction_record: Hydrated contradiction record, or ``None``.

    Returns:
        Natural surface text.

    Example::

        >>> from kai_daemon.state.holding import (
        ...     HoldingItem, HoldingType, RegisterNeeded, Urgency
        ... )
        >>> from kai_daemon.state._types import EpistemicOrigin
        >>> item = HoldingItem(
        ...     content="You once said X, but now lean toward Y.",
        ...     type=HoldingType.OBSERVATION,
        ...     relevance_trigger="X vs Y",
        ...     register_needed=RegisterNeeded.ANY,
        ...     urgency=Urgency.LOW,
        ...     source_workflow="test",
        ...     epistemic_origin=EpistemicOrigin.INTERNAL,
        ... )
        >>> msg = _format_discharge(item, None)
        >>> "X" in msg
        True
    """
    if contradiction_record is not None:
        summary = contradiction_record.conflict_summary
        return f"Something I've been sitting with: {summary}\n\n{item.content}"
    return item.content
