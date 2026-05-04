# CLAUDE.md — kai-daemon

## What this repo is

The daemon itself. Persistent memory, background workflow engine, inner life
pipeline, and conversation integration. This is where everything comes together.

Runs on the M1 Max outside a devcontainer.

## Role in the system

- Drives all inference via `mlx-kv-client` (local models on M1 Max)
- Stores and retrieves all persistent memory via `daemon-memory-client`
- Runs 29 background workflows on a priority/preemption model
- Handles conversation through `personal_assistant` (Priority 1)
- Generates embeddings on the M1 Max and writes them to `daemon-memory-server`
- The daemon's name is read from `user.yaml` at runtime — never hardcoded

## What depends on this repo

- `kai-devtools` — reads daemon state and log files for the observability panel

## What this repo depends on

- `mlx-kv-client` — wraps the five mlx-kv-server inference primitives
- `daemon-memory-client` — wraps all daemon-memory-server API routes

During active development of the clients, install as editable from this
devcontainer (siblings are visible at `/workspaces/` via DLS mount):

```bash
uv add --editable /workspaces/mlx-kv-client
uv add --editable /workspaces/daemon-memory-client
```

Once clients have stable releases, switch to published versions.

## Build, test, lint

```bash
uv sync
uv run pytest
uv run tox
uv run pyright
uv run ruff check .
```

Run `uv run tox` before signing off on any session. All four environments
(pre-commit, type-checking, tests, docs) must be green.

After fixing any tox failures, commit immediately once all four environments
pass — do not wait for the user to ask.

## Architecture references

Full spec: `/workspaces/kai-project/docs/kai-technical.md` Stages 1–4
Full architecture: `/workspaces/kai-project/docs/kai-architecture.md`
Tool catalogue, permission matrix, Kai SDK: `/workspaces/kai-project/docs/kai-tools.md`

Key sections by stage:
- Stage 1: §4 (data layer), §4a DAEMON_SELF, §4b DAEMON_RELATIONAL,
  §4c scratch space, §4d holding store, §4f thread store, §13 observability
- Stage 2: §7 inner life pipeline, §5 privacy invariants
- Stage 3: §6 workflow engine, §6b trigger vocabulary, §6d priority model,
  §6e preemption model
- Stage 3.5: `adr-001-mlx-kv-server-status-endpoint.md` — /status and inference logging
- Stage 4: §8 session architecture, §8a thread stack, §8b retrieval,
  §8d discharge, §8e register inference, §8f streaming push, §9 thread lifecycle

## Build sequence

This repo is built across four stages. Do not begin a stage until the previous
one is stable.

```
Stage 1 — State Architecture        (no inference, no workflows — pure data)
Stage 2 — Inner Life Pipeline       (background thought generation)
Stage 3 — Workflow Decomposition    (all 29 workflows, commissioned inquiry)
Stage 3.5 — Inference observability (mlx-kv-server /status, inference logging)
Stage 4 — Conversation Integration  (personal_assistant wired to everything)
```

Stages 2 and 3 can proceed in parallel once Stage 1 is done.

## Critical constraints

**daemon_seeding runs before onboarding.** The daemon produces DAEMON_SELF v1
before the onboarding interview begins. This is a prerequisite relationship in
the workflow engine — `onboarding` has `requires: daemon_seeding`. Seeding uses
the local model only (`reflection` via `mlx-kv-server`). Do not introduce any
OpenRouter dependency into `daemon_seeding` — it must work without outbound
internet access configured.

**Naming convention.** No personal names in any infrastructure identifier.
The daemon's name is `user.yaml` → `daemon_name` at runtime. Schemas, tool
names, workflow IDs, file paths all use generic terms: `daemon`, `user`, etc.

**Privacy invariant — inner life tools receive no user context.** This is the
most load-bearing invariant in the system. `daemon_inner_thought` and
`daemon_inner_thought_filter` must have no code path that reaches user data.
This is enforced by an automated test. If the test does not exist, write it
before implementing anything in Stage 2.

**Three knowledge spaces never collapse.** `user_pkm`, `daemon`, and `shared`
are structurally separated ChromaDB collections in `daemon-memory-server`. The
daemon's research workflow only ever receives the `daemon` collection handle.
This is never relaxed.

**`epistemic_origin` is immutable.** Set at write time on every scratch space
item. Never modifiable after write. Enforced in code.

**Holding store validation.** `type: reasoned_disagreement` requires a non-null
`contradiction_id`. Enforced in `holding_write`. Three test cases are required
— see Stage 1 acceptance criteria.

**Working memory is not cleared until `episodic_flush` confirms success.** If
the memory server is unavailable at session end, working memory stays in memory
until flush succeeds on reconnection.

**Push frequency ceiling is 7 days — enforced in code, not by prompt.** The
`inner_life_push_evaluation` workflow checks push history before the evaluation
prompt runs.

**Preemption mode is per-workflow and must be respected.** `suspend` workflows
call `checkpoint` before pausing and `rollback` on resume. `restart` workflows
are idempotent — their writes can be repeated safely.

**Tool permissions are enforced structurally by the Kai SDK.** Every workflow
declares its permitted tools in `workflows.yaml`. The SDK raises
`ToolPermissionError` at call time if a tool is invoked outside its permitted
context. This cannot be bypassed by prompting. See `kai-tools.md`.

## Local action API (for kai-devtools)

`kai-devtools` is read-only with respect to daemon state files, but surfaces
like contradiction resolution and BORDERLINE pool review require write actions.
These are handled via a small local HTTP API that `kai-daemon` exposes on
localhost — not the same port as any external-facing interface.

Endpoints needed (built during Stage 3 alongside kai-devtools):

```
POST /actions/contradiction/{id}/resolve
POST /actions/contradiction/{id}/dismiss
POST /actions/borderline/{id}/promote
POST /actions/borderline/{id}/discard
GET  /status/kv
     Proxy mlx-kv-server status for kai-devtools. Read-only.
```

This API is localhost-only. It is never exposed on the network.

## File structure

```
data/
  daemon_state/
    daemon_self.yaml           # DAEMON_SELF — current version
    daemon_self_history/       # all prior versions
    daemon_relational.yaml     # DAEMON_RELATIONAL — current version
    daemon_relational_history/
    holding.yaml               # holding store
    aesthetic_log.yaml         # rolling aesthetic reactions
    threads/                   # one file per thread
    pickup_notes/              # dormancy pickup notes
    memory_queue/              # queued memory writes (server unavailable)

  logs/
    workflow_runs.jsonl        # observability log — one entry per workflow run
    register_inference.jsonl   # register inference history
    inference_calls.jsonl      # mlx-kv-server primitive call log (Stage 3.5)
    tool_calls.jsonl           # Kai SDK tool call log

src/kai_daemon/
  tools/                       # Kai SDK tool implementations
  workflows/                   # all 29 workflow implementations

user.yaml                      # user config — name, preferences, thresholds
daemon-memory-server.yaml      # connection, embedding model, retrieval config, contradiction thresholds
```

## Workflow registry summary

29 workflows across priorities 0–9. See `/workspaces/kai-project/docs/kai-technical.md`
Full `workflows.yaml` section for the complete registry.

| Priority | Category            | Key workflows                                        |
|----------|---------------------|------------------------------------------------------|
| 0        | Initialization      | daemon_seeding, onboarding (seeding gates onboarding)|
| 1        | Live conversation   | personal_assistant                                   |
| 2        | Responsive          | ingest_document, commissioned_inquiry, push_message  |
| 3        | Gap-triggered       | temporal_bridging, daemon_distillation, thread_pickup|
| 4        | Post-conversation   | episodic_flush, relational_update, daily_digest      |
| 5        | Nightly knowledge   | contradiction_detection, unexamined_document_review  |
| 6        | Nightly maintenance | embedding_backfill, transcript_pruning               |
| 7        | Late night          | dormant_thread_writer, associative_retrieval         |
| 8        | Background chained  | daemon_integration, inner_life_thread_pollination    |
| 9        | Deep background     | daemon_inner_thought_generation                      |

## Stage acceptance criteria

### Stage 1
- [x] Scratch space type lifecycle rules enforced; `epistemic_origin` immutable
- [x] DAEMON_SELF and DAEMON_RELATIONAL versioning correct; token budget warnings implemented
- [x] `holding_write` validation rule with all three test cases
- [x] Thread lifecycle state transitions enforced; `time_gap_quality` null at write time
- [x] Observability hooks write on every workflow execution; `memory_server_available` correct
- [x] No personal names anywhere in implementation

### Stage 2
- [x] Both external tools contain no user data paths (automated test)
- [x] PROMPT_F fires on 14-day threshold; A–E rotation configurable
- [x] Bypass valve at configurable rate (default 12%)
- [x] Four-way routing in `daemon_integration`; fascination lifecycle check at `development_count >= 3`
- [x] Pollination deduplication; high-significance signal with 24h TTL
- [x] 7-day ceiling enforced before push evaluation prompt runs
- [x] BORDERLINE pool: review surface in kai-devtools; promote/discard actions work; 30-day auto-expiry enforced

### Stage 3
- [x] Preemption: `suspend` checkpoints/resumes; `restart` terminates/restarts; no blocking
- [x] `episodic_flush`: `suspend` preemption; checkpoint after step 3; working memory gate
- [x] `commissioned_inquiry`: abandonment preserves findings; results surfaced via push not dump
- [x] Contradiction surfacing through discharge, not separate workflow; register gate excludes `urgent`
- [x] Localhost action API implemented; contradiction and BORDERLINE actions work end-to-end
- [x] kai-devtools panel built with all observability surfaces

### Stage 3.5
- [x] Every primitive call logged to `inference_calls.jsonl` with all fields
- [x] Kai SDK initialised with permission context from `workflows.yaml` at engine startup
- [x] `ToolPermissionError` raised and logged when a tool is called outside permitted context
- [x] All tool calls logged to `tool_calls.jsonl` with workflow_id, tool, inputs, outcome

### Stage 4
- [x] Thread stack: `state` derived from rank each turn, never stored independently; stack capped at 2; floating threads separate list, no eviction
- [x] All five salience constants and `drop_threshold` configurable in `user.yaml` under `thread_stack:`
- [x] Retrieval: graceful degradation when memory server unavailable (empty results, no error)
- [x] Retrieval: when a referenced artifact has `chunk_status: pending`, acknowledge naturally ("still reading through it") rather than returning empty results silently
- [x] Discharge: both gates required; contradiction record hydrated via `contradiction_id`; at most one item per turn
- [x] Register inference: correction pathway emits new message, preserves prior response
- [x] Session end: snapshot before workflows; working memory not cleared until `episodic_flush` confirms

## GitHub issue hygiene

At the end of every session, update the project board:

```bash
gh issue close <number> --repo distilledecho/kai-daemon
bash /workspaces/kai-project/setup/project-move.sh <issue-url> "Done"
```

## Review

Run in a **fresh Claude Code session** at the end of each stage:

```
/review stage=<N>
```

Do not close a stage milestone issue until the review issue contains a sign-off line.
Never run `/review` in the same session that did the implementation.
