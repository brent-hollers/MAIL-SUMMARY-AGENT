"""
defenses/verifier.py
====================
Defense 3 of 4: Deterministic verifier.

PURPOSE (from Stage 1):
Prove the briefing is STRUCTURALLY HONEST -- that nothing was dropped,
invented, or misreferenced -- without judging whether the content is *good*.
There is no sum(range()) ground truth for a summary, so we do not verify
quality. Instead we re-derive structural facts from the RAW Gmail/Calendar
payloads (never from model output) and assert invariants that must hold if the
briefing is honest.

TWO SPECIES OF CHECK LIVE HERE, and the distinction is the whole design:

  1. SYMMETRIC INVARIANTS -- fail if output diverges from ground truth in
     EITHER direction (a dropped item and an invented item are both failures).
     These are the five from Stage 1 plus the conclusion-legality check we
     built into types.py:
       - coverage           (counts reconcile, partitioned by class)
       - referential integrity (every reference points to a real source id)
       - VIP completeness    (every in-scope VIP sender appears)
       - temporal sanity     (datetimes parse and sit in a sane window)
       - count reconciliation (a stated integer matches the structured count)
       - conclusion legality  (a changed conclusion implies an eligible thread)

  2. URGENCY FLOOR (your Stage 1 addition) -- ONE-DIRECTIONAL. It only ever
     PROMOTES an item to >= HIGH, never demotes. It is not a pass/fail check;
     it mutates priorities upward and reports what it touched. Its worst case
     is a false positive, which the HITL gate absorbs. We keep it deliberately
     dumb (lexical + temporal pattern matching) -- the moment it tries to judge
     "real urgency vs. figure of speech" it stops being deterministic.

OUTPUT CONTRACT:
The verifier RETURNS results; it does not raise on a failed invariant. It must
run every check and report every problem in one pass, so the workflow can do
retry-then-degrade against a complete picture rather than discovering failures
one at a time across re-runs.

MODEL-AGNOSTIC: nothing here calls an LLM. It compares already-produced briefing
items against re-derived raw facts. A model swap cannot weaken it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Optional

from agent.types import (
    DegradedFlag,
    Item,
    ItemClass,
    Priority,
    PriorityReason,
)


# ---------------------------------------------------------------------------
# Result types: every check reports a structured outcome, never an exception.
# ---------------------------------------------------------------------------

class CheckStatus(str, Enum):
    """Outcome of one invariant check."""
    PASS = "pass"
    FAIL = "fail"


@dataclass
class CheckResult:
    """
    The outcome of a single named invariant.

    `name` identifies which invariant (so a degraded briefing can say exactly
    *which* check failed, per Stage 1). `details` lists the specific offending
    items -- e.g. the source_ids that were dropped -- so a failure is
    actionable, not just "coverage failed."
    """
    name: str
    status: CheckStatus
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS


@dataclass
class VerifierReport:
    """
    The full result of one verifier pass: every invariant's CheckResult, plus
    a record of what the urgency floor promoted.

    The workflow reads `all_passed` to decide retry-vs-ship. On a failed pass it
    reads `failed_checks` to name the specific invariant(s) in the degraded
    flag. `promotions` is informational -- the floor never blocks shipping; it
    just reports which items it raised and why, for the gate and the ledger.
    """
    checks: list[CheckResult] = field(default_factory=list)
    promotions: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def record_degradation(self, degraded: DegradedFlag) -> None:
        """
        Write one degradation reason per failed invariant into the shared flag.
        Called by the workflow only AFTER the retry has also failed -- a first
        failure triggers a retry, not a degradation. We name the invariant and
        include the offending ids so the briefing's degraded banner is specific.
        """
        for check in self.failed_checks:
            detail = "; ".join(check.details) if check.details else "no detail"
            degraded.add(f"verifier '{check.name}' failed: {detail}")


# ---------------------------------------------------------------------------
# Ground truth: the raw facts the verifier re-derives against.
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """
    The re-derived facts the verifier checks the briefing against. This is built
    DIRECTLY from the raw Gmail/Calendar payloads by the workflow's extraction
    step -- NOT from anything the model produced. That provenance is the entire
    point: if we re-derived from model output, we'd be asking the model to grade
    its own homework.

    Fields:
      - message_ids / event_ids: the complete set of in-scope source ids, by
        class. Coverage and referential-integrity checks reconcile against these.
      - vip_message_ids: the subset of message ids whose sender is on the static
        VIP list. VIP completeness checks every one of these survived.
      - now: the run timestamp, injected (not read from the clock inside the
        check) so the temporal checks are deterministic and testable -- a test
        can pin `now` and get repeatable results.
    """
    message_ids: set[str]
    event_ids: set[str]
    vip_message_ids: set[str]
    now: datetime

    @property
    def all_source_ids(self) -> set[str]:
        """Every valid source id, regardless of class -- for referential checks."""
        return self.message_ids | self.event_ids


# ---------------------------------------------------------------------------
# Configuration for the temporal checks and the urgency floor.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifierConfig:
    """
    Tunables the verifier needs. These mirror the `thresholds` block of the
    Stage 2 config schema; they are passed in (not hardcoded) so the same
    verifier code serves anyone's rig.

    - temporal_window_days: how far on either side of `now` a datetime may sit
      before it's "insane." Catches timezone corruption and date-parse errors
      (an event dated three years out, a deadline in 1970).
    - urgency_phrases: the lexical triggers for the urgency floor. Lowercased
      substring matches against subject + summary. Deliberately a config list,
      not code, so you tune it without touching this file.
    - urgency_window_hours: a parsed deadline within this many hours of `now`
      forces a promotion (the temporal half of the floor).
    """
    temporal_window_days: int = 365
    urgency_phrases: tuple[str, ...] = (
        "asap", "as soon as possible", "today", "tonight",
        "by eod", "by end of day", "urgent", "immediately",
    )
    urgency_window_hours: int = 48


# ---------------------------------------------------------------------------
# The verifier itself.
# ---------------------------------------------------------------------------

class Verifier:
    """
    Runs all symmetric invariants and the urgency floor over a briefing's items.

    Lifecycle within a run:
        verifier = Verifier(config)
        report = verifier.verify(items, ground_truth)
        if report.all_passed:
            ... ship ...
        else:
            ... retry once; if still failing, report.record_degradation(flag)
                and ship partial ...

    The urgency floor runs as part of verify() and MUTATES item priorities in
    place (promote-only). That mutation is intentional and safe: the floor can
    only raise priority, and everything it raises still passes through the HITL
    gate for human review.
    """

    def __init__(self, config: VerifierConfig) -> None:
        self.config = config

    # =====================================================================
    # Public entry point.
    # =====================================================================

    def verify(self, items: list[Item], truth: GroundTruth) -> VerifierReport:
        """
        Run every check and the urgency floor. Returns a complete report; does
        not raise on failure (see module docstring for why all-at-once matters).

        Order note: we apply the urgency floor LAST, after the symmetric
        invariants. The invariants check structural honesty of the briefing as
        the model produced it; the floor then adjusts priorities. Running the
        floor first would mean we're verifying a briefing we already modified,
        muddying "did the model produce an honest briefing?" We want those two
        questions answered separately.
        """
        report = VerifierReport()

        # --- Symmetric invariants (each appends one CheckResult) ---
        report.checks.append(self._check_coverage(items, truth))
        report.checks.append(self._check_referential_integrity(items, truth))
        report.checks.append(self._check_vip_completeness(items, truth))
        report.checks.append(self._check_temporal_sanity(items, truth))
        report.checks.append(self._check_conclusion_legality(items))

        # --- Urgency floor (promote-only; records promotions, never fails) ---
        report.promotions = self._apply_urgency_floor(items, truth)

        return report

    # =====================================================================
    # Symmetric invariant: coverage.
    # =====================================================================

    def _check_coverage(self, items: list[Item], truth: GroundTruth) -> CheckResult:
        """
        Every in-scope source item appears in the briefing exactly once, and no
        briefing item came from outside the in-scope set -- partitioned by class.

        This is the symmetric heart of the verifier: it fails on a DROP (a
        source id missing from the briefing) AND on a DUPLICATE (a source id
        appearing twice) AND on an INVENTION (a briefing id not in ground truth).
        If 100 messages came in, exactly 100 distinct message items go out.
        """
        details: list[str] = []

        # Partition the briefing's ids by class.
        briefing_message_ids: list[str] = []
        briefing_event_ids: list[str] = []
        for it in items:
            if it.spine.item_class is ItemClass.MESSAGE:
                briefing_message_ids.append(it.spine.source_id)
            elif it.spine.item_class is ItemClass.EVENT:
                briefing_event_ids.append(it.spine.source_id)

        # Check messages, then events, with identical logic.
        for label, briefing_ids, truth_ids in (
            ("message", briefing_message_ids, truth.message_ids),
            ("event", briefing_event_ids, truth.event_ids),
        ):
            briefing_set = set(briefing_ids)

            # DROP: in ground truth but missing from the briefing.
            dropped = truth_ids - briefing_set
            for sid in sorted(dropped):
                details.append(f"{label} dropped: {sid}")

            # INVENTION: in the briefing but not in ground truth.
            invented = briefing_set - truth_ids
            for sid in sorted(invented):
                details.append(f"{label} invented (not in source): {sid}")

            # DUPLICATE: appears more than once in the briefing.
            if len(briefing_ids) != len(briefing_set):
                seen: set[str] = set()
                for sid in briefing_ids:
                    if sid in seen:
                        details.append(f"{label} duplicated: {sid}")
                    seen.add(sid)

        status = CheckStatus.PASS if not details else CheckStatus.FAIL
        return CheckResult(name="coverage", status=status, details=details)

    # =====================================================================
    # Symmetric invariant: referential integrity.
    # =====================================================================

    def _check_referential_integrity(
        self, items: list[Item], truth: GroundTruth
    ) -> CheckResult:
        """
        Every reference an item makes points to a REAL source id. An item's own
        spine.source_id must exist in ground truth, and (if present) its
        thread_id must belong to a real message. This catches a briefing line
        that references a message/event that doesn't exist -- a hallucinated
        reference -- which coverage alone wouldn't catch if the invented id
        happened not to collide with a real one.
        """
        details: list[str] = []
        valid_ids = truth.all_source_ids

        for it in items:
            sid = it.spine.source_id
            if sid not in valid_ids:
                details.append(f"item references unknown source id: {sid}")

        status = CheckStatus.PASS if not details else CheckStatus.FAIL
        return CheckResult(
            name="referential_integrity", status=status, details=details
        )

    # =====================================================================
    # Symmetric invariant: VIP completeness.
    # =====================================================================

    def _check_vip_completeness(
        self, items: list[Item], truth: GroundTruth
    ) -> CheckResult:
        """
        Every in-scope message from a VIP sender appears in the briefing. VIP
        membership is a STATIC list lookup, not a judgment, so this is fully
        deterministic -- and a silently-missing VIP is the highest-cost briefing
        failure, so it gets its own dedicated check rather than relying on
        coverage. (Coverage would also catch a dropped VIP, but a dedicated
        check names it as a VIP omission specifically, which is the failure you
        most want spelled out in the degraded banner.)
        """
        details: list[str] = []
        briefing_message_ids = {
            it.spine.source_id
            for it in items
            if it.spine.item_class is ItemClass.MESSAGE
        }

        missing_vips = truth.vip_message_ids - briefing_message_ids
        for sid in sorted(missing_vips):
            details.append(f"VIP message missing from briefing: {sid}")

        status = CheckStatus.PASS if not details else CheckStatus.FAIL
        return CheckResult(name="vip_completeness", status=status, details=details)

    # =====================================================================
    # Symmetric invariant: temporal sanity.
    # =====================================================================

    def _check_temporal_sanity(
        self, items: list[Item], truth: GroundTruth
    ) -> CheckResult:
        """
        Every datetime an item carries parses as a valid ISO-8601 timestamp and
        falls within a sane window around `now`. Catches timezone corruption and
        date-parse errors cheaply -- an event dated three years out, a deadline
        the parser mangled into 1970. We check spine.datetime_iso and the
        deadline conclusion's value (when present).
        """
        details: list[str] = []
        window = timedelta(days=self.config.temporal_window_days)
        lower = truth.now - window
        upper = truth.now + window

        def check_one(label: str, sid: str, raw: Optional[str]) -> None:
            if raw is None:
                return  # absence is fine; only a PRESENT-but-bad value fails
            parsed = _parse_iso(raw)
            if parsed is None:
                details.append(f"{label} unparseable for {sid}: {raw!r}")
                return
            if not (lower <= parsed <= upper):
                details.append(
                    f"{label} out of sane window for {sid}: {raw} "
                    f"(window {lower.date()}..{upper.date()})"
                )

        for it in items:
            sid = it.spine.source_id
            check_one("datetime", sid, it.spine.datetime_iso)
            # deadline_ts is a Conclusion; its value is the parsed deadline or None
            deadline_value = it.deadline_ts.value
            if isinstance(deadline_value, str):
                check_one("deadline", sid, deadline_value)

        status = CheckStatus.PASS if not details else CheckStatus.FAIL
        return CheckResult(name="temporal_sanity", status=status, details=details)

    # =====================================================================
    # Symmetric invariant: conclusion legality (the types.py rule, enforced).
    # =====================================================================

    def _check_conclusion_legality(self, items: list[Item]) -> CheckResult:
        """
        The rule we built into types.py, now verified after the fact: if a
        refinable conclusion (needs_reply / deadline_ts) CHANGED from its
        snippet-provisional value, the item MUST be structurally eligible for
        deep reasoning (spine.is_thread == True).

        Why verify it here when Item.refine_conclusion() already enforces it at
        write time? Defense in depth. refine_conclusion() stops the common path,
        but buggy code could assign a Conclusion's .value directly, bypassing
        the method. This check keys on the IMMUTABLE spine.is_thread, which
        cannot be forged, and so catches any illegitimate flip regardless of how
        it happened. A single message whose needs_reply flipped had no legal
        path to change -- that's the failure.
        """
        details: list[str] = []

        for it in items:
            for name in ("needs_reply", "deadline_ts"):
                conclusion = getattr(it, name)
                if conclusion.changed_from_provisional and not it.spine.is_thread:
                    details.append(
                        f"'{name}' changed on non-thread {it.spine.source_id} "
                        f"(ineligible for deep reasoning)"
                    )

        status = CheckStatus.PASS if not details else CheckStatus.FAIL
        return CheckResult(
            name="conclusion_legality", status=status, details=details
        )

    # =====================================================================
    # The urgency floor (one-directional; promotes only).
    # =====================================================================

    def _apply_urgency_floor(
        self, items: list[Item], truth: GroundTruth
    ) -> list[str]:
        """
        Promote any item matching an urgency trigger to at least HIGH. NEVER
        demotes. Returns a list of human-readable promotion records (for the
        gate and the ledger), tagged with WHY each was promoted.

        Two trigger kinds, both deterministic:
          - LEXICAL: an urgency phrase appears in subject or summary_text.
          - TEMPORAL: a parsed deadline falls within urgency_window_hours of now.

        Provenance: when we promote, we set priority_reason to URGENCY_FLOOR so
        the gate can show the item was machine-flagged, not model-judged -- and
        so you can later measure how often the floor catches urgency the local
        model missed. We do NOT overwrite a priority the model already set to
        >= HIGH (no need to promote), and we do NOT relabel a VIP-driven
        priority (VIP provenance is more specific and we preserve it).
        """
        promotions: list[str] = []
        urgency_cutoff = truth.now + timedelta(hours=self.config.urgency_window_hours)
        phrases = self.config.urgency_phrases

        for it in items:
            # If already at least HIGH, the floor has nothing to do -- skip,
            # preserving whatever reason it already carries (model or VIP).
            if it.priority.at_least(Priority.HIGH):
                continue

            triggered_by: Optional[str] = None

            # --- Lexical trigger: dumb, case-insensitive substring match ---
            haystack = " ".join(
                part for part in (it.spine.subject, it.summary_text) if part
            ).lower()
            for phrase in phrases:
                if phrase in haystack:
                    triggered_by = f"phrase '{phrase}'"
                    break

            # --- Temporal trigger: deadline within the urgency window ---
            if triggered_by is None:
                deadline_value = it.deadline_ts.value
                if isinstance(deadline_value, str):
                    parsed = _parse_iso(deadline_value)
                    if parsed is not None and truth.now <= parsed <= urgency_cutoff:
                        triggered_by = f"deadline within {self.config.urgency_window_hours}h"

            # --- Promote if triggered ---
            if triggered_by is not None:
                it.priority = Priority.HIGH
                it.priority_reason = PriorityReason.URGENCY_FLOOR
                promotions.append(
                    f"{it.spine.source_id} promoted to HIGH by {triggered_by}"
                )

        return promotions


# ---------------------------------------------------------------------------
# Module-level helper: tolerant ISO-8601 parsing.
# ---------------------------------------------------------------------------

def _parse_iso(raw: str) -> Optional[datetime]:
    """
    Parse an ISO-8601 string into a timezone-aware datetime, or return None if
    it can't be parsed. Returning None (rather than raising) lets the temporal
    check report a bad value as a check failure instead of crashing the run.

    We normalize a trailing 'Z' (Zulu/UTC) to +00:00 because Python's
    fromisoformat historically didn't accept 'Z' directly, and Gmail/Calendar
    timestamps commonly use it. Any datetime that parses naive (no tzinfo) is
    assumed UTC -- a deliberate, documented assumption so comparisons against
    `now` (which is tz-aware) don't raise.
    """
    if not isinstance(raw, str) or not raw:
        return None
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed