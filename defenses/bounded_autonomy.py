"""
defenses/bounded_autonomy.py
============================
Defense 1 of 4: Bounded autonomy.

PURPOSE (from Stage 1):
A run is allowed only so much expensive work before it must stop and ship what
it has. This module is the accountant: it tracks consumption against caps and,
when a cap is reached, records a degradation reason instead of crashing. Per
your answer 12, hitting a cap is a PARTIAL-SUCCESS condition (ship a flagged
briefing), never a fatal one -- and never a SILENT one. A cap that stops work
without surfacing why would make the briefing lie by omission ("you didn't see
the 11th important thread"), so every exhausted cap writes a human-readable
reason into the shared DegradedFlag.

WHY TWO SEPARATE CAPS (not one global step budget):
They bound different risks and should be tuned in opposite directions.
  - investigation cap (reads): each frontier deep-read of a thread costs tokens.
    A runaway here wastes money. Tune LOOSE -- we'd rather over-read than miss
    an important thread.
  - write-action cap (writes): each consequential action sends mail or mutates a
    calendar. A runaway here does real-world damage in your name. Tune TIGHT.
This module also serves double duty as the cap half of the Stage 2 escalation
boundary: the investigation cap is exactly what bounds how many times the
local->frontier deep-read rule may fire per run.

WHAT THIS MODULE IS NOT:
It knows nothing about emails, models, or tools. It counts abstract "actions"
against named budgets. That keeps it reusable and means swapping the model
cannot bypass or alter it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.types import DegradedFlag


# ---------------------------------------------------------------------------
# Exception: raised when a write would exceed its cap.
# ---------------------------------------------------------------------------

class WriteBudgetExceeded(Exception):
    """
    Raised when the workflow attempts to stage a consequential write action
    after the write cap is already exhausted.

    WHY WRITES RAISE BUT READS DON'T (the asymmetry, in code form):
    For reads, "out of budget" means "stop investigating and ship what we have"
    -- a soft landing, handled by checking `can_investigate()` before reading.
    For writes, a request to act *beyond* the safety cap is not a soft landing;
    it means the agent is trying to do more consequential things than we deemed
    safe for one run. We make that loud -- an exception the workflow must catch
    and handle deliberately -- rather than letting it slip by as a silent
    no-op. A swallowed write-cap breach is precisely the failure mode (agent
    quietly trying to over-act) we most want visible.
    """
    pass


# ---------------------------------------------------------------------------
# A single budget: one counter with a ceiling.
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """
    One named budget: a ceiling, a running count, and the bookkeeping to tell
    whether there's room left. Two of these live inside the BudgetTracker below
    (one for reads, one for writes).

    Kept as its own small type rather than four loose variables so each budget
    carries its own name in logs/degradation messages, and so adding a third
    budget later (if a future workflow needs one) is a one-line addition.
    """
    name: str        # human-readable, e.g. "investigation" -- appears in degradation reasons
    limit: int       # the ceiling; once `used` reaches this, the budget is exhausted
    used: int = 0    # how many actions consumed so far this run

    @property
    def remaining(self) -> int:
        """How many actions are still permitted. Never negative."""
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        """True once consumption has reached (or somehow passed) the ceiling."""
        return self.used >= self.limit

    def consume(self, n: int = 1) -> None:
        """
        Record that `n` actions were used. We allow `used` to be incremented
        normally; callers are expected to CHECK before consuming (reads) or to
        let the tracker enforce the ceiling (writes). This method itself is a
        dumb counter -- the policy about what to do at the ceiling lives in the
        BudgetTracker, so this stays trivially correct.
        """
        self.used += n


# ---------------------------------------------------------------------------
# The tracker: owns both budgets and the shared degraded flag.
# ---------------------------------------------------------------------------

class BudgetTracker:
    """
    The single object the workflow consults to ask "am I still allowed to do
    this?" It owns the two budgets and a reference to the run's DegradedFlag, so
    that exhausting a budget and recording the degradation happen together --
    you can't trip a cap without the reason being written.

    Usage shape (the workflow will call it like this in a later stage):

        tracker = BudgetTracker(investigation_limit=10, write_limit=15,
                                degraded=run_degraded_flag)

        # READ path -- check first, then act:
        if tracker.can_investigate():
            tracker.record_investigation()
            ... do the deep read ...
        else:
            ... skip; tracker has already flagged the run degraded ...

        # WRITE path -- ask the tracker to authorize; it raises if over cap:
        tracker.authorize_write()      # raises WriteBudgetExceeded if no room
        ... stage the consequential action ...
    """

    def __init__(
        self,
        investigation_limit: int,
        write_limit: int,
        degraded: DegradedFlag,
    ) -> None:
        # Two independent budgets, named so their degradation messages are
        # self-explanatory in the briefing and the logs.
        self.investigation = Budget(name="investigation", limit=investigation_limit)
        self.write = Budget(name="write-action", limit=write_limit)

        # The SHARED degraded flag for this run. Shared (not owned) on purpose:
        # the verifier and compaction write to the same flag, so the briefing
        # ends up with one consolidated list of every reason it ran degraded.
        self.degraded = degraded

        # Latches so we record each cap's degradation reason ONCE, even if the
        # workflow asks "can I investigate?" twenty more times after exhaustion.
        # Without these, a loop that keeps checking would spam the briefing with
        # twenty identical "investigation cap hit" lines.
        self._investigation_flagged = False
        self._write_flagged = False

    # ----- READ PATH: check-before-acting, soft landing on exhaustion -----

    def can_investigate(self) -> bool:
        """
        Ask whether another thread deep-read is permitted this run.

        Returns True if there's budget left. Returns False if exhausted -- and
        on the FIRST False, records the degradation reason (once, via the
        latch). The caller checks this before every deep read; a False means
        "stop investigating, ship what you have," which is the soft landing the
        read path is designed around.
        """
        if not self.investigation.exhausted:
            return True
        # Exhausted: flag the run degraded the first time we hit this point.
        if not self._investigation_flagged:
            unread = "one or more"  # the workflow can pass a precise count later
            self.degraded.add(
                f"investigation cap reached ({self.investigation.limit}); "
                f"{unread} thread(s) left un-investigated"
            )
            self._investigation_flagged = True
        return False

    def record_investigation(self, n: int = 1) -> None:
        """Record that a deep read happened. Call AFTER can_investigate() said yes."""
        self.investigation.consume(n)

    # ----- WRITE PATH: authorize-or-raise, hard stop on breach -----

    def authorize_write(self) -> None:
        """
        Authorize one consequential write, or refuse loudly.

        Unlike reads, writes don't get a "check and quietly skip" path. The
        workflow calls this immediately before staging any send/create/modify.
        If budget remains, it's consumed and the call returns. If not, it
        records the degradation AND raises WriteBudgetExceeded, forcing the
        workflow to handle an over-cap write deliberately rather than letting
        it slip through. This is the tight, loud control the write surface
        warrants.
        """
        if self.write.exhausted:
            if not self._write_flagged:
                self.degraded.add(
                    f"write-action cap reached ({self.write.limit}); "
                    "further actions were blocked"
                )
                self._write_flagged = True
            raise WriteBudgetExceeded(
                f"write-action budget of {self.write.limit} exhausted"
            )
        # Room remains: consume one unit and allow the write to proceed.
        self.write.consume(1)

    # ----- Introspection: for logging / the run ledger (Stage 3) -----

    def snapshot(self) -> dict:
        """
        A small dict of current consumption, for the per-run ledger and logs.
        Pure read -- no side effects. The Stage 3 observability layer records
        this so you can later query "how often did the investigation cap fire?"
        which is the signal for whether your local model under-catches and you
        need to widen the cap or escalate prioritization.
        """
        return {
            "investigation_used": self.investigation.used,
            "investigation_limit": self.investigation.limit,
            "write_used": self.write.used,
            "write_limit": self.write.limit,
            "degraded": self.degraded.is_degraded,
        }