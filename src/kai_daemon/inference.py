"""Instrumented mlx-kv-client wrapper — inference call logging (Stage 3.5).

``InstrumentedMlxKvClient`` wraps any mlx-kv-server client object and logs
every primitive call (prefill, generate, checkpoint, rollback, evict) to
``data/logs/inference_calls.jsonl`` on completion.

Design notes
------------
* The wrapper uses duck typing — it accepts ``Any`` as the client type so
  that ``mlx_kv_client`` does not need to be installed in the build/test
  environment (the server runs on the M1 Max outside this devcontainer).
* ``tokens_before`` and ``tokens_after`` are obtained by calling
  ``client.status().cache_used_tokens`` before and after each primitive.
  If the status call fails, 0 is substituted and a warning is logged —
  the primitive call is never blocked by a failed status lookup.
* ``workflow_id`` is read from the SDK's ``ContextVar`` at call time, so it
  correctly reflects whichever workflow is currently executing on the engine's
  worker thread.
* Logging is fire-and-forget: I/O errors are swallowed with a warning;
  they never affect the primitive's return value or exception.
* The ``status()`` method is proxied directly — it is not treated as a
  primitive and is not logged.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .sdk import get_workflow_id
from .state.observability import InferenceCallEntry, InferenceCallLogger

logger = logging.getLogger(__name__)

# The five mlx-kv-server inference primitives.
_PRIMITIVES: frozenset[str] = frozenset(
    {"prefill", "generate", "checkpoint", "rollback", "evict"}
)


class InstrumentedMlxKvClient:
    """mlx-kv-client wrapper that logs every primitive call.

    Parameters
    ----------
    client:
        Any object that exposes the mlx-kv-server interface — the five
        primitives and a ``status()`` method that returns an object with a
        ``cache_used_tokens: int`` attribute.  Duck-typed so that
        ``mlx_kv_client`` does not need to be installed at build time.
    log_path:
        Override the default ``inference_calls.jsonl`` path (for tests).

    Example
    -------
    ::

        from mlx_kv_client import MlxKvClient
        from kai_daemon.inference import InstrumentedMlxKvClient

        raw = MlxKvClient("http://localhost:8080")
        client = InstrumentedMlxKvClient(raw)

        client.checkpoint()   # logged to inference_calls.jsonl
        client.rollback()     # logged to inference_calls.jsonl
        client.status()       # proxied directly — not logged
    """

    def __init__(
        self,
        client: Any,
        log_path: Path | None = None,
    ) -> None:
        self._client = client
        self._logger = InferenceCallLogger(log_path)

    # ------------------------------------------------------------------
    # Status proxy (not a primitive — not logged)
    # ------------------------------------------------------------------

    def status(self) -> Any:
        """Proxy ``client.status()`` without logging."""
        return self._client.status()

    # ------------------------------------------------------------------
    # Primitive methods
    # ------------------------------------------------------------------

    def prefill(self, *args: Any, **kwargs: Any) -> Any:
        """Instrumented ``prefill`` — logged on completion."""
        return self._call_primitive("prefill", *args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Instrumented ``generate`` — logged on completion."""
        return self._call_primitive("generate", *args, **kwargs)

    def checkpoint(self, *args: Any, **kwargs: Any) -> Any:
        """Instrumented ``checkpoint`` — logged on completion."""
        return self._call_primitive("checkpoint", *args, **kwargs)

    def rollback(self, *args: Any, **kwargs: Any) -> Any:
        """Instrumented ``rollback`` — logged on completion."""
        return self._call_primitive("rollback", *args, **kwargs)

    def evict(self, *args: Any, **kwargs: Any) -> Any:
        """Instrumented ``evict`` — logged on completion."""
        return self._call_primitive("evict", *args, **kwargs)

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def _call_primitive(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call *name* on the underlying client and log the outcome."""
        tokens_before = self._safe_token_count()
        started = time.monotonic()
        success = False
        try:
            result = getattr(self._client, name)(*args, **kwargs)
            success = True
            return result
        except Exception:
            raise
        finally:
            duration_ms = int((time.monotonic() - started) * 1000)
            tokens_after = self._safe_token_count()
            self._append_log(name, tokens_before, tokens_after, duration_ms, success)

    def _safe_token_count(self) -> int:
        """Return ``cache_used_tokens`` from ``status()``, or 0 on error."""
        try:
            return int(self._client.status().cache_used_tokens)
        except Exception:
            logger.warning(
                "inference: status() call failed — substituting 0 for token count",
                exc_info=True,
            )
            return 0

    def _append_log(
        self,
        primitive: str,
        tokens_before: int,
        tokens_after: int,
        duration_ms: int,
        success: bool,
    ) -> None:
        """Append one entry to the inference call log.  Swallows all errors."""
        try:
            entry = InferenceCallEntry(
                timestamp=datetime.now(UTC).isoformat(),
                primitive=primitive,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                duration_ms=duration_ms,
                success=success,
                workflow_id=get_workflow_id(),
            )
            self._logger.append(entry)
        except Exception:
            logger.warning(
                "inference: failed to log primitive call %r", primitive, exc_info=True
            )
