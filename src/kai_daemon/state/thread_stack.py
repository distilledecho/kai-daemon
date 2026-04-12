"""Thread stack system (§8a).

Stack cap: 2 entries (foreground + peripheral). Floating entries held separately.

State   | Meaning
--------|----------------------------------------------------------
foreground | Primary topic of current exchange
peripheral | Introduced, being tracked, not currently primary
floating   | Introduced, not resolved — needs to be held, not responded to

Floating threads: No eviction. No cap. Persist until explicitly resolved or dismissed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import StrEnum


class ThreadStackState(StrEnum):
    """Computed state for non-floating entries.

    Never set directly on construction — always computed from rank via assign_states.
    """

    foreground = "foreground"
    peripheral = "peripheral"


@dataclass
class ThreadStackEntry:
    """A single entry in the thread stack.

    Attributes:
        thread_id: Unique identifier for the thread.
        state: Computed from rank — never set directly on construction.
        salience: Starts at 0.5, computed from update_salience each turn.
        engagement_depth: Starts at 0.0, decays or grows based on reference/stance
            movement.
        last_touched_turn: Turn number when entry was last referenced.
        entered_turn: Turn number when entry first entered the stack.
        is_floating: True if this is a floating thread (bypasses salience entirely).
    """

    thread_id: str
    state: ThreadStackState
    salience: float
    engagement_depth: float
    last_touched_turn: int
    entered_turn: int
    is_floating: bool = False

    # Floating threads use a separate list and bypass salience entirely


@dataclass
class SalienceConfig:
    """Configuration for salience computation.

    All constants are starting hypotheses from the spec. Loaded from user.yaml
    under thread_stack: key.

    Attributes:
        recency_decay: Decay factor for recency calculation (default 0.85).
        recency_floor: Minimum recency value (default 0.20).
        depth_decay: Decay factor for engagement_depth when unreferenced (default 0.95).
        depth_boost_reference: Depth boost for plain reference (default 0.10).
        depth_boost_stance: Depth boost when stance movement occurred (default 0.20).
        recency_weight: Weight for recency in salience calculation (default 0.60).
        depth_weight: Weight for engagement_depth in salience calculation
            (default 0.40).
        drop_threshold: Salience threshold below which entry is removed (default 0.30).
        floating_depth_decay: Decay factor for floating threads (default 0.99).
        floating_recency_floor: Recency floor for floating threads (default 0.50).
        floating_drop_threshold: Threshold for removing unresolved floating threads
            (default 0.45).
        floating_max_turns_unresolved: Max turns before removing unresolved floating
            thread (default 20).
    """

    recency_decay: float = 0.85
    recency_floor: float = 0.20
    depth_decay: float = 0.95
    depth_boost_reference: float = 0.10
    depth_boost_stance: float = 0.20
    recency_weight: float = 0.60
    depth_weight: float = 0.40
    drop_threshold: float = 0.30
    floating_depth_decay: float = 0.99
    floating_recency_floor: float = 0.50
    floating_drop_threshold: float = 0.45
    floating_max_turns_unresolved: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SalienceConfig:
        """Create config from user.yaml dictionary."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)  # type: ignore[arg-type]


def update_salience(
    entry: ThreadStackEntry,
    current_turn: int,
    was_referenced: bool,
    had_stance_movement: bool,
    config: SalienceConfig,
) -> float:
    """Update salience for an entry based on turn activity.

    Args:
        entry: The thread stack entry to update.
        current_turn: Current turn number.
        was_referenced: True if thread referenced this turn.
        had_stance_movement: True if referencing caused epistemic shift.
        config: Salience configuration from user.yaml.

    Returns:
        Computed salience value (0.0 to 1.0).

    Note:
        engagement_depth updated in-place on entry.
        last_touched_turn only updated if was_referenced is True.
    """
    turns_since_touch = current_turn - entry.last_touched_turn

    if was_referenced:
        depth_boost = (
            config.depth_boost_stance
            if had_stance_movement
            else config.depth_boost_reference
        )
        new_depth = min(1.0, entry.engagement_depth + depth_boost)
    else:
        new_depth = entry.engagement_depth * config.depth_decay

    recency = config.recency_floor + (1.0 - config.recency_floor) * (
        config.recency_decay**turns_since_touch
    )

    salience = (config.recency_weight * recency) + (config.depth_weight * new_depth)

    new_salience = min(1.0, salience)
    entry.engagement_depth = new_depth
    entry.salience = new_salience
    if was_referenced:
        entry.last_touched_turn = current_turn

    return new_salience


def assign_states(stack: list[ThreadStackEntry]) -> None:
    """Assign state (foreground/peripheral) based on salience rank.

    State is computed from rank, never stored independently. Only operates
    on non-floating entries — floating entries bypass salience entirely.

    Args:
        stack: List of ThreadStackEntry to assign states. Modified in-place.

    Note:
        Highest salience gets foreground, second highest gets peripheral.
        Entries below drop_threshold should be removed before calling this function.
    """
    ranked = sorted(stack, key=lambda e: e.salience, reverse=True)
    for i, entry in enumerate(ranked):
        if not entry.is_floating:
            entry.state = (
                ThreadStackState.foreground if i == 0 else ThreadStackState.peripheral
            )


def update_stack(
    stack: list[ThreadStackEntry],
    floating_threads: list[ThreadStackEntry],
    current_turn: int,
    referenced_thread_ids: set[str],
    stance_movement_ids: set[str],
    config: SalienceConfig,
) -> tuple[list[ThreadStackEntry], list[ThreadStackEntry]]:
    """Update the stack and floating threads for a new turn.

    This is the main per-turn function that:
    - Updates salience for each stack entry
    - Removes entries below drop_threshold from the stack
    - Assigns states (foreground/peripheral) to remaining non-floating entries
    - Applies floating thread decay and removes unresolved threads if needed

    Args:
        stack: Current thread stack (non-floating entries).
        floating_threads: List of currently floating threads.
        current_turn: Current turn number.
        referenced_thread_ids: Set of thread IDs that were referenced this turn.
        stance_movement_ids: Set of thread IDs whose epistemic status shifted.
        config: Salience configuration from user.yaml.

    Returns:
        Tuple of (updated_stack, updated_floating_threads).

    Note:
        Stack cap is 2 non-floating entries. If more than 2 remain after updates,
        the lowest-salience entry is dropped. Floating threads have no cap or eviction.
    """
    # Update salience for stack entries and filter out those below threshold
    updated_stack: list[ThreadStackEntry] = []

    for entry in stack:
        was_referenced = entry.thread_id in referenced_thread_ids
        had_stance_movement = entry.thread_id in stance_movement_ids

        new_salience = update_salience(
            entry, current_turn, was_referenced, had_stance_movement, config
        )

        if new_salience >= config.drop_threshold:
            updated_stack.append(entry)

    # Apply stack cap of 2 — keep only top 2 by salience
    if len(updated_stack) > 2:
        ranked = sorted(updated_stack, key=lambda e: e.salience, reverse=True)
        updated_stack = ranked[:2]

    # Assign states to remaining non-floating entries
    assign_states(updated_stack)

    # Update floating threads: apply decay, remove if unresolved too long
    updated_floating: list[ThreadStackEntry] = []

    for entry in floating_threads:
        turns_unresolved = current_turn - entry.last_touched_turn

        # Apply floating decay (no assign_states for floating)
        new_depth = entry.engagement_depth * config.floating_depth_decay

        recency = config.floating_recency_floor + (
            1.0 - config.floating_recency_floor
        ) * (config.recency_decay**turns_unresolved)

        new_salience = (config.recency_weight * recency) + (
            config.depth_weight * new_depth
        )

        entry.engagement_depth = new_depth
        entry.salience = new_salience

        # Remove if unresolved too long and salience below threshold
        if turns_unresolved <= config.floating_max_turns_unresolved:
            if new_salience >= config.floating_drop_threshold:
                updated_floating.append(entry)

    return (updated_stack, updated_floating)


def add_thread_to_stack(
    stack: list[ThreadStackEntry],
    thread_id: str,
    current_turn: int,
) -> list[ThreadStackEntry]:
    """Add a new thread to the stack.

    New threads enter at salience 0.5, engagement_depth 0.0. If adding would
    exceed the stack cap of 2, drop the lowest-salience non-foreground entry first.

    Args:
        stack: Current thread stack.
        thread_id: ID of new thread to add.
        current_turn: Current turn number.

    Returns:
        Updated stack with new entry added (or replacing lowest if cap exceeded).

    Note:
        New entries created with state computed from rank after addition.
    """
    new_entry = ThreadStackEntry(
        thread_id=thread_id,
        state=ThreadStackState.foreground,  # Will be recomputed by assign_states
        salience=0.5,
        engagement_depth=0.0,
        last_touched_turn=current_turn,
        entered_turn=current_turn,
        is_floating=False,
    )

    # If stack would exceed 2, drop lowest-salience entry first
    if len(stack) >= 2:
        ranked = sorted(stack, key=lambda e: e.salience, reverse=True)
        # Keep the highest-salience entry, drop the rest
        stack = [ranked[0]]

    stack.append(new_entry)
    assign_states(stack)

    return stack


def add_floating_thread(
    floating: list[ThreadStackEntry],
    thread_id: str,
    current_turn: int,
) -> list[ThreadStackEntry]:
    """Add a thread to the floating threads list.

    Floating threads have no cap and no eviction (except when max_turns_unresolved
    is exceeded). They bypass salience computation entirely — only depth decay applies.

    Args:
        floating: Current list of floating threads.
        thread_id: ID of new floating thread to add.
        current_turn: Current turn number.

    Returns:
        Updated floating threads list with new entry added.

    Note:
        Floating entries do not go through assign_states and maintain their own
        state tracking separate from main stack.
    """
    new_entry = ThreadStackEntry(
        thread_id=thread_id,
        state=ThreadStackState.foreground,  # not used for ranking in floating
        salience=0.5,
        engagement_depth=0.0,
        last_touched_turn=current_turn,
        entered_turn=current_turn,
        is_floating=True,
    )

    floating.append(new_entry)

    return floating
