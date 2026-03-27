"""DAEMON_RELATIONAL versioned store (§4b).

Versioning invariants:
- Each write produces a new file — the current version is never overwritten
  in place; it is first copied to daemon_relational_history/v{N}.yaml.
- Full history is retained indefinitely.

Token budget:
- Warn at load time if token count > 700 (cl100k_base).
- On write: truncate the ``overflow`` field first to bring under budget;
  warn if still over budget after truncation.

ChromaDB:
- One embedding per version stored in the ``daemon_relational_versions``
  collection for semantic diff queries.
- Connection config read from daemon-memory-server.yaml.
- If the ChromaDB server is unavailable the write still succeeds (with a warning).

Structural separation:
- No combined read path with DAEMON_SELF exists.  ``DaemonRelationalStore``
  has no methods that accept or return ``DaemonSelf`` objects.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from pydantic import BaseModel, Field

from ._chroma import DAEMON_RELATIONAL_COLLECTION
from ._paths import daemon_relational_history_dir, daemon_state_dir
from ._utils import _utcnow

TOKEN_BUDGET = 700
_CURRENT_FILENAME = "daemon_relational.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class OpenLoopType(StrEnum):
    """Semantic category of an open loop."""

    INTENTION = "intention"
    PLAN = "plan"
    UNRESOLVED_CONVERSATION = "unresolved_conversation"
    QUESTION = "question"


class FollowUpStyle(StrEnum):
    """How the daemon should surface this loop when the time comes."""

    CHECK_IN = "check_in"
    CHALLENGE = "challenge"
    CURIOSITY = "curiosity"
    SILENCE = "silence"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class OpenLoop(BaseModel):
    """A single open loop within DAEMON_RELATIONAL."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    type: OpenLoopType
    follow_up_style: FollowUpStyle
    follow_up_after: str
    context_needed: str = ""
    created: str = Field(default_factory=_utcnow)
    last_surfaced: str | None = None
    resolved: bool = False


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class DaemonRelational(BaseModel):
    """DAEMON_RELATIONAL document — how the daemon reads the user.

    ``version`` is managed by ``DaemonRelationalStore.write()`` and should
    not be set by callers directly.
    """

    version: int = 0
    timestamp: str = Field(default_factory=_utcnow)
    how_user_thinks: str = ""
    what_user_is_working_on: str = ""
    users_current_register: str = ""
    where_daemon_reads_user_wrong: str = ""
    open_loops: list[OpenLoop] = []
    overflow: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaml_text(dr: DaemonRelational) -> str:
    return yaml.dump(dr.model_dump(mode="json"), allow_unicode=True, sort_keys=False)


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_overflow(dr: DaemonRelational) -> DaemonRelational:
    """Trim ``overflow`` until token count fits within TOKEN_BUDGET.

    Removes tokens from the end of ``overflow``.  If ``overflow`` alone
    cannot absorb the excess it is cleared entirely.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    text = _yaml_text(dr)
    excess = len(enc.encode(text)) - TOKEN_BUDGET
    if excess <= 0:
        return dr
    overflow_tokens = enc.encode(dr.overflow) if dr.overflow else []
    keep = max(0, len(overflow_tokens) - excess)
    if keep == 0:
        return dr.model_copy(update={"overflow": ""})
    new_overflow = enc.decode(overflow_tokens[:keep])
    return dr.model_copy(update={"overflow": new_overflow})


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DaemonRelationalStore:
    """Versioned DAEMON_RELATIONAL store.

    Each ``write()`` call archives the current file to history and writes
    the new version as the current file.  Files are never overwritten.

    Inject ``chroma_client`` in tests to avoid needing a live ChromaDB server.
    Inject ``state_dir`` / ``history_dir`` to use a temporary directory.
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        history_dir: Path | None = None,
        chroma_client: Any | None = None,
    ) -> None:
        self._state_dir = state_dir or daemon_state_dir()
        self._history_dir = history_dir or daemon_relational_history_dir()
        self._current_path = self._state_dir / _CURRENT_FILENAME
        self._chroma = chroma_client
        self._collection: Any | None = None
        if self._chroma is not None:
            self._collection = self._chroma.get_or_create_collection(
                DAEMON_RELATIONAL_COLLECTION
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> DaemonRelational | None:
        """Load the current DAEMON_RELATIONAL.

        Returns ``None`` if no current version exists yet.
        Emits a ``UserWarning`` if the token count exceeds TOKEN_BUDGET.
        """
        if not self._current_path.exists():
            return None
        raw: dict[str, Any] = yaml.safe_load(self._current_path.read_text()) or {}
        dr = DaemonRelational.model_validate(raw)
        token_count = _count_tokens(_yaml_text(dr))
        if token_count > TOKEN_BUDGET:
            warnings.warn(
                f"DAEMON_RELATIONAL v{dr.version} is {token_count} tokens "
                f"(budget: {TOKEN_BUDGET})",
                stacklevel=2,
            )
        return dr

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, dr: DaemonRelational) -> DaemonRelational:
        """Write a new DAEMON_RELATIONAL version.

        Steps:
        1. Determine next version number (1 if no current, else current + 1).
        2. Stamp version and timestamp onto the new document.
        3. Truncate ``overflow`` if over TOKEN_BUDGET; warn if still over.
        4. Archive current file to history (never overwrite existing history).
        5. Write new version as the current file.
        6. Store embedding in ChromaDB (best-effort; failure only warns).
        """
        current = self.load() if self._current_path.exists() else None
        next_version = 1 if current is None else current.version + 1
        dr = dr.model_copy(update={"version": next_version, "timestamp": _utcnow()})

        # Truncate overflow if over budget
        if _count_tokens(_yaml_text(dr)) > TOKEN_BUDGET:
            dr = _truncate_overflow(dr)
            remaining = _count_tokens(_yaml_text(dr))
            if remaining > TOKEN_BUDGET:
                warnings.warn(
                    f"DAEMON_RELATIONAL v{next_version} is {remaining} tokens after "
                    f"overflow truncation (budget: {TOKEN_BUDGET})",
                    stacklevel=2,
                )

        text = _yaml_text(dr)

        # Archive current to history before replacing
        if self._current_path.exists():
            assert current is not None
            history_path = self._history_dir / f"v{current.version}.yaml"
            if not history_path.exists():
                history_path.write_text(self._current_path.read_text())

        # Write new current
        self._current_path.write_text(text)

        # Store embedding (best-effort)
        if self._collection is not None:
            try:
                self._collection.add(
                    documents=[text],
                    ids=[f"v{dr.version}"],
                    metadatas=[{"version": dr.version, "timestamp": dr.timestamp}],
                )
            except Exception:
                logger.warning(
                    "Failed to store DAEMON_RELATIONAL v%d embedding in ChromaDB",
                    dr.version,
                    exc_info=True,
                )

        return dr

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self) -> list[DaemonRelational]:
        """Return all archived versions in ascending version order."""
        versions: list[DaemonRelational] = []
        for path in self._history_dir.glob("v*.yaml"):
            raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
            versions.append(DaemonRelational.model_validate(raw))
        return sorted(versions, key=lambda d: d.version)
