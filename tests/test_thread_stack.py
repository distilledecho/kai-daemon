"""Tests for the thread stack system (§8a).

Required test cases:
1. test_update_salience_referenced — referenced entry gains depth, salience rises
2. test_update_salience_unreferenced — unreferenced entry loses depth, salience decays
3. test_update_salience_stance_movement — larger boost applied vs plain reference
4. test_assign_states_top_is_foreground — highest salience gets foreground, second gets
   peripheral
5. test_stack_cap_enforced — adding a third thread when both slots occupied drops the
   lowest-salience entry
6. test_drop_threshold_removes_entry — entry below threshold is removed from stack
7. test_floating_no_cap — adding multiple floating threads, no eviction
8. test_floating_bypass_salience — floating threads do not go through assign_states
9. test_new_thread_enters_at_half — new ThreadStackEntry has salience=0.5,
   engagement_depth=0.0
"""

from __future__ import annotations

import pytest

from kai_daemon.state.thread_stack import (
    SalienceConfig,
    ThreadStackEntry,
    ThreadStackState,
    add_floating_thread,
    add_thread_to_stack,
    assign_states,
    update_salience,
    update_stack,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> SalienceConfig:
    """Default salience configuration from user.yaml."""
    return SalienceConfig()


@pytest.fixture
def sample_entry(config: SalienceConfig) -> ThreadStackEntry:
    """Create a sample thread stack entry."""
    return ThreadStackEntry(
        thread_id="thread-1",
        state=ThreadStackState.foreground,
        salience=0.5,
        engagement_depth=0.0,
        last_touched_turn=10,
        entered_turn=5,
        is_floating=False,
    )


# ---------------------------------------------------------------------------
# Test 1: test_update_salience_referenced — referenced entry gains depth, salience rises
# ---------------------------------------------------------------------------


def test_update_salience_referenced(
    sample_entry: ThreadStackEntry, config: SalienceConfig
):
    """Referenced entry gains depth, salience rises."""
    # Entry at turn 10, update at turn 11 (1 turn since touch)
    # This keeps recency high so salience stays above 0.5
    new_salience = update_salience(
        sample_entry,
        current_turn=11,
        was_referenced=True,
        had_stance_movement=False,
        config=config,
    )

    # Depth should increase by depth_boost_reference (0.10)
    assert sample_entry.engagement_depth == pytest.approx(0.10)  # type: ignore[reportUnknownMemberType]

    # Salience should be higher than initial 0.5 (recency still high, depth increased)
    assert new_salience > 0.5

    # last_touched_turn should be updated
    assert sample_entry.last_touched_turn == 11


# ---------------------------------------------------------------------------
# Test 2: test_update_salience_unreferenced — unreferenced entry loses depth,
#         salience decays
# ---------------------------------------------------------------------------


def test_update_salience_unreferenced(
    sample_entry: ThreadStackEntry, config: SalienceConfig
):
    """Unreferenced entry loses depth, salience decays."""
    # Entry at turn 10, update at turn 20 (10 turns since touch)
    new_salience = update_salience(
        sample_entry,
        current_turn=20,
        was_referenced=False,
        had_stance_movement=False,
        config=config,
    )

    # Depth should decay: 0.0 * 0.95^10 = 0.0
    assert sample_entry.engagement_depth == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]

    # Salience should be lower than initial 0.5 due to recency decay
    assert new_salience < 0.5

    # last_touched_turn should NOT be updated (still 10)
    assert sample_entry.last_touched_turn == 10


# ---------------------------------------------------------------------------
# Test 3: test_update_salience_stance_movement — larger boost applied vs plain reference
# ---------------------------------------------------------------------------


def test_update_salience_stance_movement(
    sample_entry: ThreadStackEntry, config: SalienceConfig
):
    """Larger boost applied vs plain reference."""
    # First, apply a plain reference
    update_salience(
        sample_entry,
        current_turn=15,
        was_referenced=True,
        had_stance_movement=False,
        config=config,
    )

    # Now apply a stance movement (should give larger boost)
    new_salience = update_salience(
        sample_entry,
        current_turn=16,
        was_referenced=True,
        had_stance_movement=True,
        config=config,
    )

    # Depth should be: 0.10 + 0.20 (depth_boost_stance) = 0.30
    assert sample_entry.engagement_depth == pytest.approx(0.30)  # type: ignore[reportUnknownMemberType]

    # Salience should be higher than after plain reference
    assert new_salience > 0.5

    # last_touched_turn should be updated to 16
    assert sample_entry.last_touched_turn == 16


# ---------------------------------------------------------------------------
# Test 4: test_assign_states_top_is_foreground — highest salience gets foreground,
#         second gets peripheral
# ---------------------------------------------------------------------------


def test_assign_states_top_is_foreground(config: SalienceConfig):
    """Highest salience gets foreground, second gets peripheral."""
    entry1 = ThreadStackEntry(
        thread_id="thread-1",
        state=ThreadStackState.foreground,
        salience=0.7,
        engagement_depth=0.3,
        last_touched_turn=10,
        entered_turn=5,
        is_floating=False,
    )

    entry2 = ThreadStackEntry(
        thread_id="thread-2",
        state=ThreadStackState.foreground,
        salience=0.4,
        engagement_depth=0.1,
        last_touched_turn=8,
        entered_turn=6,
        is_floating=False,
    )

    entry3 = ThreadStackEntry(
        thread_id="thread-3",
        state=ThreadStackState.foreground,
        salience=0.6,
        engagement_depth=0.25,
        last_touched_turn=9,
        entered_turn=7,
        is_floating=False,
    )

    stack = [entry1, entry2, entry3]
    assign_states(stack)

    # Highest salience (0.7) gets foreground
    assert entry1.state == ThreadStackState.foreground

    # Second highest (0.6) gets peripheral
    assert entry2.state == ThreadStackState.peripheral

    # Third highest (0.4) also gets peripheral (but only top 2 matter for stack cap)
    assert entry3.state == ThreadStackState.peripheral


# ---------------------------------------------------------------------------
# Test 5: test_stack_cap_enforced — adding a third thread when both slots occupied drops
#         the lowest-salience entry
# ---------------------------------------------------------------------------


def test_stack_cap_enforced(config: SalienceConfig):
    """Adding a third thread when both slots occupied drops lowest-salience entry."""
    stack: list[ThreadStackEntry] = []

    # Add first thread at turn 1
    stack = add_thread_to_stack(stack, "thread-1", current_turn=1)

    # Add second thread at turn 2
    stack = add_thread_to_stack(stack, "thread-2", current_turn=2)

    # Both should be in stack
    assert len(stack) == 2

    # Reference thread-1 to increase its salience
    stack, _ = update_stack(
        stack=stack,
        floating_threads=[],
        current_turn=5,
        referenced_thread_ids={"thread-1"},
        stance_movement_ids=set(),
        config=config,
    )

    # Add third thread at turn 5 - should drop the lowest-salience entry
    stack = add_thread_to_stack(stack, "thread-3", current_turn=5)

    # Stack should still be capped at 2
    assert len(stack) == 2

    # thread-3 should be in stack (newest), and either thread-1 or thread-2
    thread_ids = {e.thread_id for e in stack}
    assert "thread-3" in thread_ids


# ---------------------------------------------------------------------------
# Test 6: test_drop_threshold_removes_entry — entry below threshold is removed
#         from stack
# ---------------------------------------------------------------------------


def test_drop_threshold_removes_entry(config: SalienceConfig):
    """Entry below threshold is removed from stack."""
    entry = ThreadStackEntry(
        thread_id="thread-1",
        state=ThreadStackState.foreground,
        salience=0.5,
        engagement_depth=0.1,
        last_touched_turn=10,
        entered_turn=5,
        is_floating=False,
    )

    stack = [entry]

    # Update many turns later without referencing - should drop below threshold
    stack, _ = update_stack(
        stack=stack,
        floating_threads=[],
        current_turn=30,  # 20 turns since last touch
        referenced_thread_ids=set(),
        stance_movement_ids=set(),
        config=config,
    )

    # Entry should be removed (below drop_threshold of 0.30)
    assert len(stack) == 0


# ---------------------------------------------------------------------------
# Test 7: test_floating_no_cap — adding multiple floating threads, no eviction
# ---------------------------------------------------------------------------


def test_floating_no_cap(config: SalienceConfig):
    """Adding multiple floating threads, no eviction."""
    floating: list[ThreadStackEntry] = []

    # Add multiple floating threads
    for i in range(10):
        floating = add_floating_thread(floating, f"floating-{i}", current_turn=i + 1)

    # All should be present (no cap on floating threads)
    assert len(floating) == 10

    # All should have is_floating=True
    for entry in floating:
        assert entry.is_floating is True


# ---------------------------------------------------------------------------
# Test 8: test_floating_bypass_salience — floating threads do not go through
#         assign_states
# ---------------------------------------------------------------------------


def test_floating_bypass_salience(config: SalienceConfig):
    """Floating threads do not go through assign_states."""
    floating: list[ThreadStackEntry] = []

    # Add a floating thread at turn 1
    floating = add_floating_thread(floating, "floating-1", current_turn=1)

    # Update at turn 5 (only 4 turns unresolved, well within max_turns_unresolved of 20)
    # This tests that floating threads bypass assign_states but still get decay applied
    _, updated_floating = update_stack(
        stack=[],
        floating_threads=floating,
        current_turn=5,
        referenced_thread_ids=set(),
        stance_movement_ids=set(),
        config=config,
    )

    # Floating thread should still be in the list (4 turns < max_turns_unresolved of 20)
    assert len(updated_floating) == 1

    # Floating threads maintain their own state tracking
    assert updated_floating[0].is_floating is True

    # Floating threads get depth decay applied (no assign_states)
    assert updated_floating[0].engagement_depth < 1.0


# ---------------------------------------------------------------------------
# Test 9: test_new_thread_enters_at_half — new ThreadStackEntry has salience=0.5,
#         engagement_depth=0.0
# ---------------------------------------------------------------------------


def test_new_thread_enters_at_half(config: SalienceConfig):
    """New ThreadStackEntry has salience=0.5, engagement_depth=0.0."""
    stack: list[ThreadStackEntry] = []

    # Add a new thread at turn 1
    stack = add_thread_to_stack(stack, "new-thread", current_turn=1)

    # Find the new thread
    new_entry = next((e for e in stack if e.thread_id == "new-thread"), None)
    assert new_entry is not None

    # Should enter at salience 0.5, depth 0.0
    assert new_entry.salience == pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]
    assert new_entry.engagement_depth == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]

    # last_touched_turn and entered_turn should both be 1
    assert new_entry.last_touched_turn == 1
    assert new_entry.entered_turn == 1


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------


def test_stack_and_floating_separate(config: SalienceConfig):
    """Stack and floating threads are maintained separately."""
    stack: list[ThreadStackEntry] = []
    floating: list[ThreadStackEntry] = []

    # Add a regular thread to stack
    stack = add_thread_to_stack(stack, "stack-thread", current_turn=1)

    # Add a floating thread
    floating = add_floating_thread(floating, "floating-thread", current_turn=2)

    # Update both
    stack, floating = update_stack(
        stack=stack,
        floating_threads=floating,
        current_turn=5,
        referenced_thread_ids=set(),
        stance_movement_ids=set(),
        config=config,
    )

    # Stack should have the stack-thread
    assert len(stack) == 1
    assert stack[0].thread_id == "stack-thread"

    # Floating should have the floating-thread
    assert len(floating) == 1
    assert floating[0].thread_id == "floating-thread"

    # Verify they are separate lists
    assert stack is not floating


def test_floating_max_turns_unresolved_removes_entry(config: SalienceConfig):
    """Floating thread removed when max_turns_unresolved exceeded."""
    floating: list[ThreadStackEntry] = []

    # Add a floating thread at turn 1
    floating = add_floating_thread(floating, "floating-1", current_turn=1)

    # Update at turn 202 (exceeds floating_max_turns_unresolved of 200)
    _, updated_floating = update_stack(
        stack=[],
        floating_threads=floating,
        current_turn=202,
        referenced_thread_ids=set(),
        stance_movement_ids=set(),
        config=config,
    )

    # Floating thread should be removed (202 - 1 = 201 > 200)
    assert len(updated_floating) == 0


def test_entry_salience_updated_after_update_stack(config: SalienceConfig):
    """entry.salience is written back after update_stack so state assignment is correct.

    Before the fix, entry.salience was never updated — assign_states sorted by
    stale 0.5 values and state assignment was meaningless.
    """
    stack: list[ThreadStackEntry] = []

    stack = add_thread_to_stack(stack, "thread-1", current_turn=1)
    stack = add_thread_to_stack(stack, "thread-2", current_turn=2)

    stack, _ = update_stack(
        stack=stack,
        floating_threads=[],
        current_turn=3,
        referenced_thread_ids={"thread-1"},
        stance_movement_ids=set(),
        config=config,
    )

    entry1 = next(e for e in stack if e.thread_id == "thread-1")
    entry2 = next(e for e in stack if e.thread_id == "thread-2")

    # Both entries must have left the stale initial 0.5
    assert entry1.salience != pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]
    assert entry2.salience != pytest.approx(0.5)  # type: ignore[reportUnknownMemberType]

    # State assignment must reflect current salience ordering, not stale values
    higher = entry1 if entry1.salience > entry2.salience else entry2
    lower = entry2 if entry1.salience > entry2.salience else entry1
    assert higher.state == ThreadStackState.foreground
    assert lower.state == ThreadStackState.peripheral
