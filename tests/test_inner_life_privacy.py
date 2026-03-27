"""Privacy invariant tests for inner life tools (§7a, Stage 2A).

CRITICAL: This file enforces the load-bearing privacy invariant:

    daemon_inner_thought and daemon_inner_thought_filter must have
    no code path that reaches user data.

Both the module-level import boundary and function-level signature
boundary are verified. These tests must exist and pass before any
inner life implementation is merged.

User-context markers guarded against:
  - DaemonRelational / DaemonRelationalStore  (tracks user state)
  - daemon_relational                          (module import)
  - Any future user-context module added to the guard list below

The guard list is intentionally narrow: it must name concrete types
that carry user data, not hypothetical future types. Add to it when
a new user-context type is introduced.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect

# ---------------------------------------------------------------------------
# User-context identifiers that must never appear in inner life tool modules
# ---------------------------------------------------------------------------

_FORBIDDEN_TYPE_SUBSTRINGS: tuple[str, ...] = (
    "DaemonRelational",
    "DaemonRelationalStore",
)

_FORBIDDEN_MODULE_SUBSTRINGS: tuple[str, ...] = ("daemon_relational",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_path(module_name: str) -> str:
    """Return the filesystem path of an importable module."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None, (
        f"Cannot locate source for {module_name!r}"
    )
    return spec.origin


def _module_imports(module_name: str) -> list[str]:
    """Return all imported names/modules as strings (via AST, not execution).

    Note: only direct imports are checked. Transitive imports through helper
    modules are not covered. If inner_thought.py delegates to a helper that
    imports daemon_relational, this test would not catch it.
    """
    path = _source_path(module_name)
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
            for alias in node.names:
                imported.append(alias.name)
    return imported


def _annotation_strings(fn: object) -> list[str]:
    """Return all annotation string values for a callable (PEP 563-safe)."""
    anns = getattr(fn, "__annotations__", {})
    return [str(v) for v in anns.values()]


# ---------------------------------------------------------------------------
# daemon_inner_thought — module import boundary
# ---------------------------------------------------------------------------


def test_inner_thought_module_does_not_import_user_context() -> None:
    """inner_thought.py must not import daemon_relational or any user-context module."""
    imported = _module_imports("kai_daemon.tools.inner_thought")
    for name in imported:
        for forbidden in _FORBIDDEN_MODULE_SUBSTRINGS:
            assert forbidden not in name, (
                f"kai_daemon.tools.inner_thought imports {name!r} — "
                f"this contains the forbidden substring {forbidden!r}. "
                "The inner life tools must never reach user-context modules."
            )


def test_inner_thought_filter_module_does_not_import_user_context() -> None:
    """inner_thought_filter.py must not import daemon_relational."""
    imported = _module_imports("kai_daemon.tools.inner_thought_filter")
    for name in imported:
        for forbidden in _FORBIDDEN_MODULE_SUBSTRINGS:
            assert forbidden not in name, (
                f"kai_daemon.tools.inner_thought_filter imports {name!r} — "
                f"this contains the forbidden substring {forbidden!r}. "
                "The inner life tools must never reach user-context modules."
            )


# ---------------------------------------------------------------------------
# daemon_inner_thought — function signature boundary
# ---------------------------------------------------------------------------


def test_inner_thought_function_signature_has_no_user_context_types() -> None:
    """daemon_inner_thought parameters must not reference user-context types.

    Under ``from __future__ import annotations`` (PEP 563), annotations are
    stored as strings. We check the string representation directly so that
    complex forms like ``Optional[DaemonRelational]`` are also caught.
    """
    from kai_daemon.tools.inner_thought import daemon_inner_thought

    for annotation_str in _annotation_strings(daemon_inner_thought):
        for forbidden in _FORBIDDEN_TYPE_SUBSTRINGS:
            assert forbidden not in annotation_str, (
                f"daemon_inner_thought annotation {annotation_str!r} references "
                f"{forbidden!r} — this violates the privacy invariant. "
                "Inner life tools must receive no user context."
            )


def test_inner_thought_module_level_functions_have_no_user_context_types() -> None:
    """All callables in inner_thought.py must have no user-context annotations."""
    from kai_daemon.tools import inner_thought as mod

    for name, obj in inspect.getmembers(mod, predicate=inspect.isfunction):
        for annotation_str in _annotation_strings(obj):
            for forbidden in _FORBIDDEN_TYPE_SUBSTRINGS:
                assert forbidden not in annotation_str, (
                    f"kai_daemon.tools.inner_thought.{name} annotation "
                    f"{annotation_str!r} references {forbidden!r} — "
                    "this violates the privacy invariant."
                )


# ---------------------------------------------------------------------------
# daemon_inner_thought_filter — function signature boundary
# ---------------------------------------------------------------------------


def test_inner_thought_filter_function_signature_has_no_user_context_types() -> None:
    """daemon_inner_thought_filter parameters must not reference user-context types."""
    from kai_daemon.tools.inner_thought_filter import daemon_inner_thought_filter

    for annotation_str in _annotation_strings(daemon_inner_thought_filter):
        for forbidden in _FORBIDDEN_TYPE_SUBSTRINGS:
            assert forbidden not in annotation_str, (
                f"daemon_inner_thought_filter annotation {annotation_str!r} "
                f"references {forbidden!r} — this violates the privacy invariant."
            )


def test_inner_thought_filter_module_level_fns_have_no_user_context_types() -> None:
    """All callables in inner_thought_filter.py have no user-context annotations."""
    from kai_daemon.tools import inner_thought_filter as mod

    for name, obj in inspect.getmembers(mod, predicate=inspect.isfunction):
        for annotation_str in _annotation_strings(obj):
            for forbidden in _FORBIDDEN_TYPE_SUBSTRINGS:
                assert forbidden not in annotation_str, (
                    f"kai_daemon.tools.inner_thought_filter.{name} annotation "
                    f"{annotation_str!r} references {forbidden!r} — "
                    "this violates the privacy invariant."
                )


# ---------------------------------------------------------------------------
# daemon_inner_thought_filter — accepts only raw_output (no user payload)
# ---------------------------------------------------------------------------


def test_inner_thought_filter_first_positional_is_raw_output() -> None:
    """daemon_inner_thought_filter's first param must be raw_output: str."""
    from kai_daemon.tools.inner_thought_filter import daemon_inner_thought_filter

    sig = inspect.signature(daemon_inner_thought_filter)
    params = list(sig.parameters.values())
    assert params, "daemon_inner_thought_filter has no parameters"

    first = params[0]
    assert first.name == "raw_output", (
        f"First parameter is {first.name!r}, expected 'raw_output'. "
        "The filter must receive only the raw thought — no user context."
    )
    # Annotation must be 'str' — not a user-context object
    empty = inspect.Parameter.empty
    ann = str(first.annotation) if first.annotation is not empty else "str"
    assert ann == "str", (
        f"raw_output annotation is {ann!r}, expected 'str'. "
        "The filter must receive a plain string — no user-context wrapper."
    )
