"""DAEMON_SELF versioned store (§4a).

Versioning invariants:
- Each write produces a new file — the current version is never overwritten
  in place; it is first copied to daemon_self_history/v{N}.yaml.
- Full history is retained indefinitely.

Token budget:
- Warn at load time if token count > 500 (cl100k_base).
- On write: truncate the ``overflow`` field first to bring under budget;
  warn if still over budget after truncation.

ChromaDB:
- One embedding per version stored in the ``daemon_self_versions`` collection
  for semantic diff queries.
- Connection config read from daemon-memory-server.yaml.
- If the ChromaDB server is unavailable the write still succeeds (with a warning).

Structural separation:
- No combined read path with DAEMON_RELATIONAL exists.  ``DaemonSelfStore``
  has no methods that accept or return ``DaemonRelational`` objects.
"""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from pydantic import BaseModel, Field

from ._chroma import DAEMON_SELF_COLLECTION
from ._paths import daemon_self_history_dir, daemon_state_dir

TOKEN_BUDGET = 500
_CURRENT_FILENAME = "daemon_self.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FascinationStatus(StrEnum):
    """Lifecycle state of a fascination topic."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PROMOTED_TO_OPEN_QUESTION = "promoted_to_open_question"


class FascinationOrigin(StrEnum):
    """How the fascination first arose."""

    SEEDING = "seeding"
    INNER_LIFE_PIPELINE = "inner_life_pipeline"
    CONVERSATION = "conversation"
    INTEGRATION = "integration"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Fascination(BaseModel):
    """A single fascination entry within DAEMON_SELF."""

    topic: str
    what_daemon_finds_interesting: str
    connection_to_user: str | None = None
    created: str = Field(default_factory=_utcnow)
    last_updated: str = Field(default_factory=_utcnow)
    last_developed: str | None = None
    development_count: int = 0
    status: FascinationStatus = FascinationStatus.ACTIVE
    origin: FascinationOrigin


class OpenQuestion(BaseModel):
    """An unresolved question the daemon is sitting with."""

    question: str
    why_unresolved: str
    created: str = Field(default_factory=_utcnow)
    user_has_touched_this: bool = False


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class DaemonSelf(BaseModel):
    """DAEMON_SELF document — who the daemon is.

    ``version`` is managed by ``DaemonSelfStore.write()`` and should not be
    set by callers directly.
    """

    version: int = 0
    timestamp: str = Field(default_factory=_utcnow)
    who_daemon_is: str = ""
    current_fascinations: list[Fascination] = []
    aesthetic_sensibilities: str = ""
    open_questions: list[OpenQuestion] = []
    daemon_on_daemon: str = ""
    overflow: str = ""
    distillation_notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaml_text(ds: DaemonSelf) -> str:
    return yaml.dump(ds.model_dump(mode="json"), allow_unicode=True, sort_keys=False)


def _count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_overflow(ds: DaemonSelf) -> DaemonSelf:
    """Trim ``overflow`` until token count fits within TOKEN_BUDGET.

    Removes tokens from the end of ``overflow``.  If ``overflow`` alone
    cannot absorb the excess it is cleared entirely (further truncation of
    other fields is left to the caller).
    """
    enc = tiktoken.get_encoding("cl100k_base")
    text = _yaml_text(ds)
    excess = len(enc.encode(text)) - TOKEN_BUDGET
    if excess <= 0:
        return ds
    overflow_tokens = enc.encode(ds.overflow) if ds.overflow else []
    keep = max(0, len(overflow_tokens) - excess)
    if keep == 0:
        return ds.model_copy(update={"overflow": ""})
    new_overflow = enc.decode(overflow_tokens[:keep])
    return ds.model_copy(update={"overflow": new_overflow})


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DaemonSelfStore:
    """Versioned DAEMON_SELF store.

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
        self._history_dir = history_dir or daemon_self_history_dir()
        self._current_path = self._state_dir / _CURRENT_FILENAME
        self._chroma = chroma_client
        self._collection: Any | None = None
        if self._chroma is not None:
            self._collection = self._chroma.get_or_create_collection(
                DAEMON_SELF_COLLECTION
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self) -> DaemonSelf | None:
        """Load the current DAEMON_SELF.

        Returns ``None`` if no current version exists yet.
        Emits a ``UserWarning`` if the token count exceeds TOKEN_BUDGET.
        """
        if not self._current_path.exists():
            return None
        raw: dict[str, Any] = yaml.safe_load(self._current_path.read_text()) or {}
        ds = DaemonSelf.model_validate(raw)
        token_count = _count_tokens(_yaml_text(ds))
        if token_count > TOKEN_BUDGET:
            warnings.warn(
                f"DAEMON_SELF v{ds.version} is {token_count} tokens "
                f"(budget: {TOKEN_BUDGET})",
                stacklevel=2,
            )
        return ds

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, ds: DaemonSelf) -> DaemonSelf:
        """Write a new DAEMON_SELF version.

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
        ds = ds.model_copy(update={"version": next_version, "timestamp": _utcnow()})

        # Truncate overflow if over budget
        if _count_tokens(_yaml_text(ds)) > TOKEN_BUDGET:
            ds = _truncate_overflow(ds)
            remaining = _count_tokens(_yaml_text(ds))
            if remaining > TOKEN_BUDGET:
                warnings.warn(
                    f"DAEMON_SELF v{next_version} is {remaining} tokens after "
                    f"overflow truncation (budget: {TOKEN_BUDGET})",
                    stacklevel=2,
                )

        text = _yaml_text(ds)

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
                    ids=[f"v{ds.version}"],
                    metadatas=[{"version": ds.version, "timestamp": ds.timestamp}],
                )
            except Exception:
                logger.warning(
                    "Failed to store DAEMON_SELF v%d embedding in ChromaDB",
                    ds.version,
                    exc_info=True,
                )

        return ds

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(self) -> list[DaemonSelf]:
        """Return all archived versions in ascending version order."""
        versions: list[DaemonSelf] = []
        for path in self._history_dir.glob("v*.yaml"):
            raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
            versions.append(DaemonSelf.model_validate(raw))
        return sorted(versions, key=lambda d: d.version)
