# Config-Driven Local-First AI Agent

A portable, config-driven agent engine built around four explicit safety
defenses, with a swappable model interface that lets the same workflow run
against a local model or a frontier API by changing config rather than code.

The first workflow built on this engine is a personal **inbox + calendar
daily-briefing agent**: in the morning it reads mail and calendar and produces a
summary, a prioritized task list, and a "prepare me" synthesis; in the afternoon
it follows up to update the task list and stage next steps.

> **Status: early development.** Stage 1 (the four defenses + shared types) is
> implemented. Configuration, the model client, the workflow, and production
> tooling are not yet built. Installation and usage instructions are
> intentionally deferred until the engine is further along — see
> [Roadmap](#roadmap).

## Design principles

- **Local-first, cost-aware.** Cheap, high-frequency work (extract, classify,
  summarize, route, prioritize) runs on a modest local GPU. Only genuine
  reasoning — deep reading of email chains and the final synthesis — escalates to
  a frontier API.
- **Portability floor.** Designed to run on a modest home rig (~12 GB VRAM,
  ~8B-class local model), with a minimal dependency surface and no babysat
  server processes. Anything that violates this floor belongs in the cloud tier.
- **Swappable model interface.** The model endpoint is an OpenAI-compatible
  interface (base URL, model name, context window) configured, never hardcoded.
  Swapping a local model, or switching the frontier provider, is a config edit.
- **Defenses live above the swappable layer.** The four defenses and the tool
  dispatch sit above the model client, so changing the model cannot bypass a
  safety check.

## The four defenses

| Defense | What it does in this workflow |
|---|---|
| **Bounded autonomy** | Two independent caps — thread-investigation (reads, tuned loose) and write-actions (tuned tight). Hitting either ships a partial, flagged result rather than failing silently. |
| **Context compaction** | Each item is split into a frozen structured *spine* (ids, sender, datetimes, booleans) and a fuzzy prose blob. Only the prose is ever summarized; VIP items are carried verbatim. A set-reconciliation check confirms no item was dropped, duplicated, invented, or mutated. |
| **Deterministic verifier** | Re-derives ground truth from the raw payloads (never from model output) and asserts structural invariants: coverage, referential integrity, VIP completeness, temporal sanity, and conclusion legality. Includes a one-directional **urgency floor** that can only promote priority, never demote. |
| **Human-in-the-loop gate** | Gates on action class (any external mutation), never on schedule. Reads never gate; every send/create/modify is shown as a concrete rendered payload for explicit, per-item approval. Auto-send is hard-blocked in v1. |

## Repository layout

```
agent/
  types.py               Shared data structures: the frozen item spine,
                         refinable conclusions, priority/provenance enums,
                         and the degraded-run flag. Everything imports from here.

defenses/
  bounded_autonomy.py    Defense 1 — the two caps and the budget tracker.
  compaction.py          Defense 2 — safe prose summarization + set reconciliation.
  verifier.py            Defense 3 — symmetric invariants + the urgency floor.
  hitl_gate.py           Defense 4 — the action-class approval gate.
```

All defense modules are **model-agnostic**: they operate on data the workflow
hands them and never call a model directly.

## Roadmap

- [x] **Stage 1 — Defenses.** Shared types and the four defense modules.
- [ ] **Stage 2 — Config + model client.** The config schema (endpoints, tier
  mapping, thresholds), the tier-resolving model client that makes
  local-vs-frontier a config choice, and the escalation boundary.
- [ ] **Stage 3 — Path to production.** Secrets handling, SQLite persistence,
  observability, failure recovery, scheduling, distribution, and the swap-in
  test.
- [ ] **Detailed installation & usage docs.** Written once the engine is
  runnable end-to-end.

## Requirements (preliminary)

- Python 3.10+ (the code uses modern type-hint syntax).
- A local OpenAI-compatible model endpoint (e.g. Ollama or vLLM) for the local
  tier — *to be wired in Stage 2*.
- A frontier API key (Claude or Gemini) for the reasoning tier — *to be wired in
  Stage 2*.

Exact dependencies and setup steps will be documented when Stage 3 lands.