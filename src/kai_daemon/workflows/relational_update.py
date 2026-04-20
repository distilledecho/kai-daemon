"""relational_update workflow (§4I).

Updates DAEMON_RELATIONAL at session end based on the working memory
snapshot.  Short, idempotent — restart preemption mode is safe because
the write is deterministic given the same snapshot.

Reads from the snapshot:
- Turn notes (dominant register, topics, thread IDs)
- Within-session relational shadow (register corrections)

Uses inference (presentation model) to update the four prose fields:
  how_user_thinks, what_user_is_working_on, users_current_register,
  where_daemon_reads_user_wrong

Priority: 4 / Preemption: restart / Model: presentation
Trigger: conversation_ended
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ..state.daemon_relational import DaemonRelational, DaemonRelationalStore
from ..state.working_memory import WorkingMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are updating a relational model that captures how the daemon reads this \
user.

Current relational model:
---
how_user_thinks: {how_user_thinks}
what_user_is_working_on: {what_user_is_working_on}
users_current_register: {users_current_register}
where_daemon_reads_user_wrong: {where_daemon_reads_user_wrong}
---

Session data:
- Turn count: {turn_count}
- Topics touched: {topics}
- Dominant register: {dominant_register}
- Register corrections this session: {corrections}
- Active thread IDs: {thread_ids}

Update only what the session evidence supports. Write "unchanged" for any \
field where the session adds nothing new.

Reply with exactly these labeled sections:

HOW_USER_THINKS: <updated prose or "unchanged">
WHAT_USER_IS_WORKING_ON: <updated prose or "unchanged">
USERS_CURRENT_REGISTER: <updated prose or "unchanged">
WHERE_DAEMON_READS_USER_WRONG: <updated prose or "unchanged">
"""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class RelationalUpdateResult:
    """Returned on successful completion."""

    session_id: str
    relational_version: int
    fields_updated: list[str]
    """Names of fields that were updated (not "unchanged")."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_response(
    response: str,
    current: DaemonRelational,
) -> tuple[DaemonRelational, list[str]]:
    """Parse labeled-section response; return updated model and changed field names."""
    fields: dict[str, str | None] = {
        "HOW_USER_THINKS": None,
        "WHAT_USER_IS_WORKING_ON": None,
        "USERS_CURRENT_REGISTER": None,
        "WHERE_DAEMON_READS_USER_WRONG": None,
    }
    pattern = re.compile(
        r"^(HOW_USER_THINKS|WHAT_USER_IS_WORKING_ON|USERS_CURRENT_REGISTER"
        r"|WHERE_DAEMON_READS_USER_WRONG):\s*(.*)$",
        re.MULTILINE,
    )
    for m in pattern.finditer(response):
        key, value = m.group(1), m.group(2).strip()
        fields[key] = value

    updated_fields: list[str] = []
    updates: dict[str, str] = {}

    mapping = {
        "HOW_USER_THINKS": "how_user_thinks",
        "WHAT_USER_IS_WORKING_ON": "what_user_is_working_on",
        "USERS_CURRENT_REGISTER": "users_current_register",
        "WHERE_DAEMON_READS_USER_WRONG": "where_daemon_reads_user_wrong",
    }

    for label, attr in mapping.items():
        value = fields[label]
        if value and value.lower() != "unchanged":
            updates[attr] = value
            updated_fields.append(attr)

    new_dr = current.model_copy(update=updates) if updates else current
    return new_dr, updated_fields


# ---------------------------------------------------------------------------
# Public workflow function
# ---------------------------------------------------------------------------


def relational_update(
    working_memory: WorkingMemory,
    *,
    inference_fn: Callable[[str], str],
    store: DaemonRelationalStore | None = None,
) -> RelationalUpdateResult:
    """Update DAEMON_RELATIONAL from the session snapshot (§4I).

    Idempotent: given the same snapshot and current relational state, the
    same update is written.  Safe for restart preemption.

    Parameters
    ----------
    working_memory:
        Immutable working memory snapshot.  This function never mutates it.
    inference_fn:
        Callable that sends a prompt to the presentation model and returns
        text.
    store:
        Injectable ``DaemonRelationalStore``.  Defaults to the real store.

    Returns
    -------
    RelationalUpdateResult
        Session ID, new relational version, and list of updated field names.
    """
    _store = store or DaemonRelationalStore()
    session_id = working_memory.session_id

    logger.info("relational_update: starting session=%s", session_id)

    current = _store.load() or DaemonRelational()

    # Build session summary for the prompt
    turn_notes = working_memory.turn_notes
    topics = list(
        dict.fromkeys(topic for tn in turn_notes for topic in tn.topics_touched)
    )
    thread_ids = list(
        dict.fromkeys(tid for tn in turn_notes for tid in tn.thread_ids_active)
    )
    dominant_register = ""
    if turn_notes:
        counter: Counter[str] = Counter(tn.register for tn in turn_notes)
        dominant_register = counter.most_common(1)[0][0]

    corrections = working_memory.relational_shadow.corrections_this_session
    corrections_text = (
        ", ".join(f"{inf} → {cor}" for inf, cor in corrections)
        if corrections
        else "none"
    )

    prompt = _PROMPT_TEMPLATE.format(
        how_user_thinks=current.how_user_thinks or "(not yet set)",
        what_user_is_working_on=current.what_user_is_working_on or "(not yet set)",
        users_current_register=current.users_current_register or "(not yet set)",
        where_daemon_reads_user_wrong=current.where_daemon_reads_user_wrong
        or "(not yet set)",
        turn_count=working_memory.turn_count,
        topics=", ".join(topics[:10]) if topics else "none",
        dominant_register=dominant_register or "unknown",
        corrections=corrections_text,
        thread_ids=", ".join(thread_ids[:5]) if thread_ids else "none",
    )

    response = inference_fn(prompt)
    updated_dr, updated_fields = _parse_response(response, current)

    written = _store.write(updated_dr)
    logger.info(
        "relational_update: complete session=%s version=%d fields_updated=%s",
        session_id,
        written.version,
        updated_fields,
    )

    return RelationalUpdateResult(
        session_id=session_id,
        relational_version=written.version,
        fields_updated=updated_fields,
    )
