"""
defenses/hitl_gate.py
=====================
Defense 4 of 4: Enforced human-in-the-loop gate.

PURPOSE (from Stage 1):
This is the only thing between the agent and your real inbox/calendar. It
blocks every CONSEQUENTIAL action -- anything that mutates external state --
behind explicit, per-item human approval, and presents the CONCRETE rendered
payload (this body, to this recipient, at this time) so you approve the actual
thing, not a description of it.

THE RULES THIS FILE ENFORCES (all from Stage 1):
  - Gate on ACTION CLASS, not time of day. A morning send gates exactly like an
    afternoon one. The gate knows nothing about schedule.
  - Reads NEVER gate. Fetching, classifying, prioritizing touch nothing in the
    world. Gating them would bury you in approvals and train you to rubber-stamp
    -- which destroys the gate's value.
  - Per-item, explicit approval. No "approve all," no default-yes. Silence is a
    rejection, not an approval.
  - "Basic email" auto-send is HARD-BLOCKED for v1. The agent may DRAFT (text is
    harmless); it may never auto-send. The classification "basic" is never
    permission to skip the gate.
  - FAIL CLOSED: the default for any action is "gate it." An action skips the
    gate only if it is explicitly on the read-only allow-list. An unknown action
    type gates.

WHAT THIS MODULE IS NOT:
It does not send, create, or modify anything. It is a gate, not a dispatcher.
It takes proposed actions, blocks for your decision, and returns only the
approved ones. Actual dispatch happens in the workflow layer, AFTER approval,
and an action is marked done only after dispatch SUCCEEDS (Stage 3 crash-safety:
a crash between approval and send re-surfaces the action rather than silently
sending or dropping it).

The decision callback is injected (see ApprovalGate.__init__). This file never
assumes HOW approval is collected -- CLI prompt, a queued review UI, a test
stub. That keeps the gate's logic testable and the collection mechanism
swappable, the same seam philosophy as the model client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from agent.types import DegradedFlag


# ---------------------------------------------------------------------------
# Action taxonomy: what kinds of actions exist, and which are consequential.
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """
    Every kind of action the agent can propose. The gate's entire safety posture
    rests on classifying these correctly into read-only vs. consequential.

    Read-only actions touch nothing in the world; consequential actions mutate
    external state (send mail, change a calendar). When you add a new action
    type later, you MUST also place it in one of the two sets below -- and if you
    forget, the fail-closed default treats it as consequential, so the failure
    mode of forgetting is "too much gating," never "an ungated mutation."
    """
    # --- Read-only: never gate ---
    FETCH_THREAD = "fetch_thread"        # pull a full email thread for deep read
    CLASSIFY = "classify"                # label an item
    PRIORITIZE = "prioritize"            # assign priority
    DRAFT_REPLY = "draft_reply"          # produce reply TEXT -- not sending it

    # --- Consequential: always gate ---
    SEND_EMAIL = "send_email"            # send a reply / new mail
    CREATE_EVENT = "create_event"        # create a calendar event
    MODIFY_EVENT = "modify_event"        # change an existing event
    DELETE_EVENT = "delete_event"        # remove an event


# The READ-ONLY allow-list. ONLY these skip the gate. Anything not in this set
# -- including any future or unrecognized action type -- gates. This is the
# fail-closed default in one data structure: membership here is the *only* way
# to be ungated.
_READ_ONLY_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.FETCH_THREAD,
    ActionType.CLASSIFY,
    ActionType.PRIORITIZE,
    ActionType.DRAFT_REPLY,
})


def is_consequential(action_type: ActionType) -> bool:
    """
    True if an action must be gated. Defined as 'NOT explicitly read-only', so
    the default for anything unrecognized is to gate it. We never enumerate the
    consequential actions for this decision -- enumerating read-only ones and
    negating is what makes the default fail closed.
    """
    return action_type not in _READ_ONLY_ACTIONS


# ---------------------------------------------------------------------------
# A proposed action and the human's decision on it.
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    """The human's verdict on one proposed consequential action."""
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ProposedAction:
    """
    One consequential action the agent wants to take, with the CONCRETE payload
    rendered for human review.

    `frozen=True` matters here: once an action is proposed and shown to you, its
    payload must not change before dispatch. Approving "send body X to A" and
    then dispatching a mutated "body Y to B" would defeat the gate. Immutability
    guarantees the thing you approved is byte-for-byte the thing that gets
    dispatched.

    Fields:
      - action_type: drives the gate decision (consequential vs not).
      - source_id: the message/event this action relates to, for the audit trail.
      - rendered_payload: the ACTUAL content for review -- the full email body
        and recipient, or the full event details. NOT a summary, NOT a count.
        This is the "stating it in text doesn't count; show the real thing" rule.
      - metadata: structured copy of the payload (recipient, subject, start time)
        for the audit log and for the dispatcher to act on after approval.
    """
    action_type: ActionType
    source_id: str
    rendered_payload: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A consequential action with an empty rendered payload is a bug: you
        # cannot meaningfully approve nothing. Fail loudly at construction.
        if is_consequential(self.action_type) and not self.rendered_payload.strip():
            raise ValueError(
                f"consequential action {self.action_type.value} for "
                f"{self.source_id} has an empty rendered_payload; "
                "there is nothing for the human to review"
            )


@dataclass
class GateRecord:
    """
    The outcome for one proposed action after it passed through the gate. This
    is the gate-audit row from Stage 3 persistence: every consequential action
    proposed, and your approve/reject decision, recorded. Non-negotiable for a
    write-capable agent -- when it acts in your name, you need a record of what
    was approved and what wasn't.
    """
    action: ProposedAction
    decision: Decision
    # `dispatched` starts False and is set True by the workflow ONLY after the
    # action is successfully dispatched. The gate never sets this -- it's the
    # crash-safety seam: approved-but-not-yet-dispatched is a distinct, visible
    # state, so a crash in that window re-surfaces the action.
    dispatched: bool = False


# ---------------------------------------------------------------------------
# The gate itself.
# ---------------------------------------------------------------------------

# The injected approval collector: given a ProposedAction, return a Decision.
# Injecting this (rather than hardcoding a CLI prompt) is what makes the gate
# testable and the collection mechanism swappable. A test passes a stub that
# returns APPROVED; production passes one that actually prompts you.
ApprovalCallback = Callable[[ProposedAction], Decision]


class ApprovalGate:
    """
    The gate. The workflow routes EVERY proposed action through `process()`.
    Read-only actions pass straight through (returned as auto-approved records,
    no human involved). Consequential actions are shown to the human via the
    injected callback and recorded with the resulting decision.

    Usage shape (workflow, a later stage):

        gate = ApprovalGate(approval_callback=prompt_user, degraded=run_flag)
        records = gate.process(proposed_actions)
        approved = [r.action for r in records if r.decision is Decision.APPROVED]
        # ... dispatch each approved action; on success, mark r.dispatched = True ...
    """

    def __init__(
        self,
        approval_callback: ApprovalCallback,
        degraded: DegradedFlag,
    ) -> None:
        # How we ask the human. Injected -- the gate never assumes the mechanism.
        self._approval_callback = approval_callback
        # Shared run flag: if the human rejects actions, that's not a degradation
        # (a rejection is the gate working as intended). But we DO flag if an
        # auto-send was attempted and blocked -- see _reject_autosend below.
        self._degraded = degraded

    # ----- Public entry point -----

    def process(self, actions: list[ProposedAction]) -> list[GateRecord]:
        """
        Route every proposed action through the gate. Returns one GateRecord per
        action, in the same order. Read-only actions are recorded as auto-
        approved without bothering the human; consequential actions are shown
        for a per-item decision.

        Per-item is enforced structurally: we loop and call the callback once
        per consequential action. There is no path that approves a batch in one
        decision -- each action gets its own explicit verdict.
        """
        records: list[GateRecord] = []

        for action in actions:
            if not is_consequential(action.action_type):
                # Read-only: passes straight through, no human, no gate.
                records.append(
                    GateRecord(action=action, decision=Decision.APPROVED)
                )
                continue

            # Consequential: must be reviewed. Ask the human for THIS action.
            decision = self._approval_callback(action)
            records.append(GateRecord(action=action, decision=decision))

        return records

    # ----- Hard block on auto-send (v1 rule) -----

    def assert_no_autosend(self, action: ProposedAction) -> None:
        """
        Enforce the v1 hard block: the agent may DRAFT replies but may never
        AUTO-SEND. This is called by the workflow at the point where an action
        is created, to catch any code path that tries to construct a send action
        flagged as 'automatic' (e.g. from a misclassified 'basic' email).

        We treat an attempted auto-send as a degradation AND raise, because it
        means some upstream logic tried to bypass the human entirely -- exactly
        the failure the gate exists to prevent. 'Basic' is never permission to
        skip review; only the human's explicit approval lets a send through.
        """
        if action.action_type is ActionType.SEND_EMAIL and action.metadata.get("auto_send"):
            self._degraded.add(
                f"blocked attempted auto-send for {action.source_id}; "
                "auto-send is disabled in v1 -- routed to manual approval instead"
            )
            raise AutoSendBlocked(
                f"auto-send is hard-blocked in v1 (source_id={action.source_id})"
            )


# ---------------------------------------------------------------------------
# Exception for the auto-send hard block.
# ---------------------------------------------------------------------------

class AutoSendBlocked(Exception):
    """
    Raised when upstream logic attempts an auto-send while the v1 hard block is
    in force. Loud by design: an attempted auto-send is a safety-relevant event,
    not something to swallow. The workflow catches this, downgrades the action
    to a normal gated send (human must approve), and the run continues degraded-
    flagged so you can see it happened.
    """
    pass