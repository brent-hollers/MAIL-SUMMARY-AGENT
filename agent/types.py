"""
agent/types.py
==============
Shared data structures used across all four defenses and the workflow.

Design rationale:

The "spine" is the structured, IMMUTABLE core of every inbox/calendar item.
Two distinct kinds of thing live in the spine, and both are frozen for the
same reason -- they are derived directly from the source payload and never
change within a run:

  1. Durable identity tokens (source_id, sender, subject, datetimes) -- the
     things the verifier re-derives ground truth against.
  2. Structural facts (is_thread, thread_message_count) -- properties of the
     raw payload that also act as ELIGIBILITY GATES for later processing.

The key insight that shaped this design: the thing that is truly write-once is
not a conclusion like "needs_reply", but the STRUCTURAL FACT that determines
whether that conclusion is allowed to be revised. A thread's message count
comes straight off the payload and never changes; whether an item *needs a
reply* is a conclusion that a deep read may legitimately refine. So we freeze
the cause (is_thread) and leave the effect (needs_reply) mutable -- but the
effect can only legally change when the frozen cause permits it. The verifier
checks that rule against the IMMUTABLE field, which cannot be forged.

Nothing in this module imports or calls an LLM. The defenses operate on data
the workflow hands them; that separation is what stops a model swap from
bypassing a defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums: small closed vocabularies. Using enums instead of bare strings means
# a typo ("hihg") fails at construction time, not silently downstream.
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """
    Priority buckets for an item. Inherits from `str` so it serializes cleanly
    to JSON/SQLite as its value, while still being a typed enum in Python.

    Ordering matters for the urgency floor: the verifier promotes items to at
    least HIGH, never demotes. `_order` gives a comparable rank without relying
    on enum definition order (so reordering this enum later can't silently
    break the floor's comparisons).
    """
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

    @property
    def _order(self) -> int:
        return {"low": 0, "normal": 1, "high": 2, "urgent": 3}[self.value]

    def at_least(self, other: "Priority") -> bool:
        """True if self is >= other in priority rank. Used by the urgency floor."""
        return self._order >= other._order


class PriorityReason(str, Enum):
    """
    WHY an item got its priority. Provenance is required: we must distinguish a priority 
    the local model judged from one a deterministic rule forced. This lets the HITL gate 
    show the reason, and lets you later measure how often the local model under-catches 
    urgency on its own.
    """
    MODEL = "model"                  # the local model's own judgment
    URGENCY_FLOOR = "urgency_floor"  # forced by deterministic lexical/temporal rule
    VIP = "vip"                      # forced because sender is on the static VIP list


class ItemClass(str, Enum):
    """What kind of source object an item came from. Lets the verifier partition
    coverage counts by class (messages vs events reconcile separately)."""
    MESSAGE = "message"
    EVENT = "event"


class DerivedFrom(str, Enum):
    """
    Provenance for a refinable conclusion (needs_reply, deadline_ts): was it set
    from the cheap snippet pass, or refined by an escalated deep read?

    This is NOT the enforcement mechanism -- the frozen `is_thread` field is.
    This is the audit trail the verifier reads alongside is_thread to confirm a
    change was legitimate. We record it for observability and so the gate can
    show "this conclusion came from a full-thread read, not just the snippet."
    """
    SNIPPET = "snippet"      # cheap header+snippet pass
    DEEP_READ = "deep_read"  # escalated full-body / full-thread read


# ---------------------------------------------------------------------------
# The Spine: the protected, immutable core of every item.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemSpine:
    """
    The immutable spine of one inbox or calendar item.

    `frozen=True` is the enforcement mechanism, not a style choice. Once the
    spine is extracted from the raw Gmail/Calendar payload, NOTHING downstream
    -- not compaction, not the model, not the renderer -- can mutate a field
    here. An attempt raises, rather than silently flipping a value or rewriting
    a sender. Two categories of field live here:

      - Durable identity tokens: copied verbatim from the source payload.
      - Structural facts: is_thread / thread_message_count, which also gate
        whether deep reasoning is permitted for this item.

    None of these fields is ever produced by an LLM. They are all either copied
    verbatim or derived deterministically (parsed timestamp, VIP list lookup,
    message-count comparison).

    NOTE on "immutable": frozen means immutable WITHIN a run. A later run (e.g.
    the afternoon debrief) re-extracts from fresh payloads and may see a higher
    thread_message_count if new mail arrived -- that is correct, it is new
    ground truth for a new run. Cross-run identity is handled by dedup keys
    (Stage 3 persistence), not by this field.
    """
    # --- Identity (used by the verifier's referential-integrity checks) ---
    source_id: str                   # Gmail message id or Calendar event id -- the durable key
    item_class: ItemClass            # MESSAGE or EVENT
    thread_id: Optional[str] = None  # Gmail thread id; None for events

    # --- Critical tokens (must survive compaction byte-for-byte) ---
    sender: Optional[str] = None     # email address, verbatim; None for events
    subject: Optional[str] = None    # subject / event title, verbatim
    datetime_iso: Optional[str] = None  # event start or message timestamp, ISO 8601

    # --- Static-lookup flag ---
    vip_flag: bool = False           # is sender on the static VIP list? (lookup, not judgment)

    # --- Structural facts that ALSO gate deep-reasoning eligibility ---
    is_thread: bool = False          # message_count > 1, straight from payload -- write-once
    thread_message_count: int = 1    # actual count in the thread; 1 for a single message

    def __post_init__(self) -> None:
        # source_id keys every downstream check -- fail loudly at construction
        # rather than producing an un-referenceable item the verifier chokes on.
        if not self.source_id:
            raise ValueError("ItemSpine.source_id is required and cannot be empty")
        # Internal consistency: is_thread must agree with the count. Catches an
        # extraction bug at the source rather than letting an inconsistent spine
        # corrupt the eligibility gate.
        if self.is_thread and self.thread_message_count <= 1:
            raise ValueError(
                f"is_thread=True but thread_message_count={self.thread_message_count}; "
                "a thread must have >1 message"
            )
        if not self.is_thread and self.thread_message_count != 1:
            raise ValueError(
                f"is_thread=False but thread_message_count={self.thread_message_count}; "
                "a non-thread must have exactly 1 message"
            )

    @property
    def deep_reasoning_allowed(self) -> bool:
        """
        The eligibility gate. Deep reasoning (frontier `thread_deep_read`) is
        PERMITTED for this item only if it is structurally a thread. A single
        message has no chain to reason over, so escalating it would just burn
        frontier tokens on one snippet.

        Both the escalation logic (Stage 2) and the verifier (below) read this
        off the frozen spine, so eligibility cannot be forged: nothing can flip
        a single message into a "thread" to sneak it past the cap into a
        frontier call.
        """
        return self.is_thread


# ---------------------------------------------------------------------------
# A refinable conclusion: value + provenance, with a legality check that keys
# on the frozen spine.
# ---------------------------------------------------------------------------

@dataclass
class Conclusion:
    """
    A conclusion that MAY be refined by an escalated deep read: needs_reply and
    deadline_ts are the two we model this way.

    This is intentionally lightweight -- it is NOT the guarded-transition
    machinery from the earlier draft. The enforcement of "may this change?"
    does not live here; it lives in the frozen `is_thread` field on the spine,
    which the verifier checks. This struct just carries the current value, the
    value it was first given from the snippet pass, and which pass last set it.
    Keeping the provisional value lets the verifier detect *whether* a change
    happened, then check that the change was permitted.
    """
    value: object                          # the current value (bool for needs_reply, str|None for deadline)
    provisional_value: object              # the value the snippet pass first assigned
    source: DerivedFrom = DerivedFrom.SNIPPET  # which pass last wrote `value`

    @property
    def changed_from_provisional(self) -> bool:
        """True if a later pass altered the snippet's original conclusion."""
        return self.value != self.provisional_value


# ---------------------------------------------------------------------------
# The full Item: spine + refinable conclusions + compactible prose + priority.
# ---------------------------------------------------------------------------

@dataclass
class Item:
    """
    A full briefing item: the immutable spine plus the mutable parts.

    The compaction defense's whole premise is the split between protected and
    fuzzy:
      - `spine`        -> frozen, never summarized, never mutated
      - `summary_text` -> the ONLY field compaction is allowed to shorten
      - `needs_reply` / `deadline_ts` -> refinable conclusions (see legality below)
      - `priority` / `priority_reason` -> assigned during prioritization

    Legality of conclusion changes is NOT enforced by this class -- it is
    checked by the verifier against `spine.is_thread`. The rule:

        if a conclusion changed from its provisional (snippet) value,
        then spine.is_thread MUST be True.

    A single message whose needs_reply flipped had no legitimate path to change
    (it is ineligible for deep reasoning), so that is a verifier failure --
    something corrupted it. Because the check keys on the immutable is_thread,
    it cannot be fooled by a forged provenance tag.
    """
    spine: ItemSpine

    # Refinable conclusions. Defaults construct a SNIPPET-sourced conclusion
    # whose value and provisional value agree (i.e. "not yet changed").
    needs_reply: Conclusion = field(
        default_factory=lambda: Conclusion(value=False, provisional_value=False)
    )
    deadline_ts: Conclusion = field(
        default_factory=lambda: Conclusion(value=None, provisional_value=None)
    )

    # Fuzzy / assigned fields.
    summary_text: str = ""                        # compaction may shorten THIS only
    priority: Priority = Priority.NORMAL
    priority_reason: PriorityReason = PriorityReason.MODEL

    def refine_conclusion(self, name: str, new_value: object) -> None:
        """
        Apply a deep-read refinement to a conclusion ('needs_reply' or
        'deadline_ts'). This method REFUSES the refinement if the item is not
        eligible for deep reasoning -- a defense-in-depth backstop so an
        ineligible change can't even be constructed, separate from the
        verifier's after-the-fact check.

        We deliberately enforce here AND verify later: this method stops the
        common case at write time; the verifier catches anything that bypassed
        this method (e.g. direct field assignment by buggy code).
        """
        if not self.spine.deep_reasoning_allowed:
            raise ValueError(
                f"Cannot refine '{name}' on source_id={self.spine.source_id}: "
                "item is not a thread and is ineligible for deep reasoning"
            )
        target = getattr(self, name)
        if not isinstance(target, Conclusion):
            raise ValueError(f"'{name}' is not a refinable Conclusion")
        target.value = new_value
        target.source = DerivedFrom.DEEP_READ


# ---------------------------------------------------------------------------
# Degraded flag + run result: how partial-with-flag (your answer 12) is carried.
# ---------------------------------------------------------------------------

@dataclass
class DegradedFlag:
    """
    Records that a run did not complete cleanly. Per Stage 1/3, a tripped cap or
    a twice-failed verifier does NOT crash the run -- it ships partial output
    with this flag attached, and the flag must surface IN the briefing, not just
    in a log.

    A run is degraded iff `reasons` is non-empty.
    """
    reasons: list[str] = field(default_factory=list)

    @property
    def is_degraded(self) -> bool:
        return len(self.reasons) > 0

    def add(self, reason: str) -> None:
        """Append a degradation cause. Called by any defense that ships partial."""
        self.reasons.append(reason)