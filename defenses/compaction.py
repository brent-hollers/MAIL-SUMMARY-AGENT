"""
defenses/compaction.py
======================
Defense 2 of 4: Context compaction.

PURPOSE (from Stage 1):
Tool output and item prose can grow past what we want to carry in context. We
shorten it -- but inbox data is full of short, critical tokens (sender
addresses, ids, datetimes, the needs_reply boolean) where dropping or flipping
ONE is the failure that matters. Your original engine's number-preservation
check is unsafe here: it would happily pass a summary that dropped "from:
ceo@..." or flipped needs_reply, because those aren't numbers.

THE FIX (the key design move):
Don't compact the critical tokens at all. They live in the FROZEN ItemSpine
(see types.py), which compaction physically cannot mutate. Compaction is
allowed to touch exactly ONE field: `summary_text`, the fuzzy prose blob. So
the safety property is structural -- the protected tokens are protected by the
type system, not by this code remembering to be careful.

That turns the verification from "did the numbers survive?" into SET
RECONCILIATION over the protected spine:
  - every item that went in comes out (no DROP)
  - no item comes out twice (no DUPLICATE)
  - no item comes out that didn't go in (no INVENTION)
  - every surviving item's spine is byte-for-byte unchanged (no critical-token
    mutation) -- guaranteed by frozen, asserted here as defense in depth
And the Stage 1 rule: VIP items are NOT summarized at all. Their prose is
carried verbatim, because a VIP is exactly where we least tolerate fuzzy loss.

MODEL-AGNOSTIC:
This module does NOT call an LLM. The actual summarization is an injected
function (the workflow backs it with the local model). Compaction's SAFETY
logic -- what may change, what gets checked -- is separate from the MECHANISM of
summarizing, so swapping the model cannot weaken the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from agent.types import DegradedFlag, Item, ItemSpine


# ---------------------------------------------------------------------------
# The injected summarizer: text in, shorter text out. No model assumptions.
# ---------------------------------------------------------------------------

# Given the original prose and a target length budget, return shortened prose.
# The workflow supplies an implementation backed by the local model. Compaction
# treats it as a black box -- it never inspects HOW the summary was made, only
# checks the structural invariants afterward.
Summarizer = Callable[[str, int], str]


# ---------------------------------------------------------------------------
# Config for when and how hard to compact.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactionConfig:
    """
    Tunables, mirroring the `thresholds` block of the Stage 2 config schema
    (passed in, never hardcoded).

    - trigger_chars: only summarize a summary_text longer than this. Short prose
      isn't worth a model call and isn't a context-budget problem. (We measure
      in characters, not tokens, to avoid a tokenizer dependency -- keeps the
      portability/minimal-deps floor. Characters are a coarse but adequate proxy
      for "is this blob big enough to bother shortening?")
    - target_chars: the length budget handed to the summarizer for over-trigger
      prose. A goal, not a hard guarantee -- the summarizer is a black box, so we
      verify the RESULT structurally rather than trusting it hit the number.
    - skip_vip: if True (the Stage 1 default), VIP items are never summarized.
    """
    trigger_chars: int = 1000
    target_chars: int = 300
    skip_vip: bool = True


# ---------------------------------------------------------------------------
# Result of a compaction pass: the items (mutated in place) + a report.
# ---------------------------------------------------------------------------

class CompactionStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"


@dataclass
class CompactionReport:
    """
    The outcome of one compaction pass over a list of items.

    `status` is FAIL only if SET RECONCILIATION failed -- i.e. an item was
    dropped, duplicated, or invented, or a spine changed. A FAIL is a serious
    integrity problem (compaction lost data), and like the verifier it records a
    degradation rather than crashing.

    `compacted_ids` and `skipped_vip_ids` are informational, for the run ledger:
    they tell you how much was shortened and how many VIP items were carried
    verbatim, useful for tuning trigger_chars later.
    """
    status: CompactionStatus
    compacted_ids: list[str] = field(default_factory=list)
    skipped_vip_ids: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status is CompactionStatus.PASS

    def record_degradation(self, degraded: DegradedFlag) -> None:
        """Write each reconciliation failure into the shared run flag."""
        for f in self.failures:
            degraded.add(f"compaction integrity failure: {f}")


# ---------------------------------------------------------------------------
# The compactor.
# ---------------------------------------------------------------------------

class Compactor:
    """
    Shortens item prose safely and verifies nothing critical was lost.

    Usage shape (workflow, a later stage):

        compactor = Compactor(config, summarizer=local_model_summarize)
        report = compactor.compact(items)
        if not report.passed:
            report.record_degradation(run_flag)   # ship partial, flagged

    The items are mutated IN PLACE -- their summary_text fields are shortened.
    Their spines are not touched (cannot be -- frozen). The report tells you
    whether the set survived intact.
    """

    def __init__(self, config: CompactionConfig, summarizer: Summarizer) -> None:
        self.config = config
        self._summarize = summarizer

    # ----- Public entry point -----

    def compact(self, items: list[Item]) -> CompactionReport:
        """
        Compact the prose of every eligible item, then verify the set survived.

        Steps:
          1. Snapshot the BEFORE state -- the set of spines, captured before we
             touch anything. This is ground truth for reconciliation.
          2. Compact each eligible item's summary_text in place (skip VIP, skip
             too-short, skip already-empty).
          3. Reconcile the AFTER state against the BEFORE snapshot.
        """
        # --- Step 1: snapshot before doing anything ---
        # We capture the spines as an immutable reference set. Because spines are
        # frozen and hashable (frozen dataclass), we can put them in a set and
        # compare by VALUE -- the @dataclass __eq__ from types.py is exactly what
        # makes this reconciliation meaningful.
        before_spines = [it.spine for it in items]
        before_ids = [s.source_id for s in before_spines]

        report = CompactionReport(status=CompactionStatus.PASS)

        # --- Step 2: compact eligible items in place ---
        for it in items:
            if self._should_skip(it, report):
                continue
            original = it.summary_text
            shortened = self._summarize(original, self.config.target_chars)
            # Guard: a summarizer that returns empty or longer text is
            # misbehaving. We do NOT trust it blindly -- if it returned something
            # unusable, keep the original prose rather than destroying content.
            # (Losing the summary entirely would be worse than not compacting.)
            if shortened and len(shortened) < len(original):
                it.summary_text = shortened
                report.compacted_ids.append(it.spine.source_id)
            # else: leave original in place; no-op, not a failure.

        # --- Step 3: reconcile ---
        self._reconcile(items, before_spines, before_ids, report)

        return report

    # ----- Eligibility -----

    def _should_skip(self, item: Item, report: CompactionReport) -> bool:
        """
        Decide whether an item's prose is left untouched. Reasons to skip:
          - VIP (Stage 1 rule): never summarize a VIP item's prose. Recorded so
            the ledger shows VIP items were carried verbatim.
          - prose shorter than trigger_chars: not worth a model call.
          - empty prose: nothing to shorten.
        """
        if self.config.skip_vip and item.spine.vip_flag:
            report.skipped_vip_ids.append(item.spine.source_id)
            return True
        if len(item.summary_text) < self.config.trigger_chars:
            return True
        if not item.summary_text.strip():
            return True
        return False

    # ----- Reconciliation: the safety check -----

    def _reconcile(
        self,
        items_after: list[Item],
        before_spines: list[ItemSpine],
        before_ids: list[str],
        report: CompactionReport,
    ) -> None:
        """
        SET RECONCILIATION over the protected spine. Asserts the compaction pass
        preserved the item set exactly. Any failure flips the report to FAIL and
        records a specific reason.

        Four checks, mirroring the verifier's coverage logic but scoped to the
        before/after of THIS pass:
          - DROP: a spine present before is missing after.
          - INVENTION: a spine present after was not present before.
          - DUPLICATE: a source_id appears more than once after.
          - MUTATION: a surviving spine differs from its before-snapshot.
                      (Frozen makes this near-impossible, but we check anyway --
                      defense in depth, and it catches an item being REPLACED by
                      a different item with the same id.)
        """
        after_spines = [it.spine for it in items_after]
        after_ids = [s.source_id for s in after_spines]

        before_id_set = set(before_ids)
        after_id_set = set(after_ids)

        # DROP
        for sid in sorted(before_id_set - after_id_set):
            report.failures.append(f"item dropped during compaction: {sid}")

        # INVENTION
        for sid in sorted(after_id_set - before_id_set):
            report.failures.append(f"item invented during compaction: {sid}")

        # DUPLICATE
        if len(after_ids) != len(after_id_set):
            seen: set[str] = set()
            for sid in after_ids:
                if sid in seen:
                    report.failures.append(f"item duplicated during compaction: {sid}")
                seen.add(sid)

        # MUTATION: build id -> spine maps and compare spines that exist in both.
        # Because ItemSpine is a frozen dataclass, value-equality (__eq__) tells
        # us whether the critical tokens are identical. A difference means a
        # spine was swapped out for a different one under the same id.
        before_by_id = {s.source_id: s for s in before_spines}
        after_by_id = {s.source_id: s for s in after_spines}
        for sid in sorted(before_id_set & after_id_set):
            if before_by_id[sid] != after_by_id[sid]:
                report.failures.append(
                    f"spine mutated during compaction for {sid} "
                    "(critical tokens changed)"
                )

        if report.failures:
            report.status = CompactionStatus.FAIL