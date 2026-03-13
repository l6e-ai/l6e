"""l6e exceptions."""
from __future__ import annotations


class BudgetExceeded(Exception):
    """Raised when a pipeline run has exhausted its budget and halt mode is active."""

    def __init__(self, spent: float, budget: float, reason: str = "") -> None:
        self.spent = spent
        self.budget = budget
        self.reason = reason
        super().__init__(
            f"Budget exceeded: spent ${spent:.6f} of ${budget:.6f} budget. {reason}".strip()
        )


class LatencySLAExceeded(Exception):
    """Not raised in v0.1; reserved for v0.2 latency SLA enforcement.

    Kept importable for forward-compatibility so user code like
    ``except LatencySLAExceeded`` compiles today without error.
    """

    def __init__(self, elapsed_ms: float, sla_ms: float) -> None:
        self.elapsed_ms = elapsed_ms
        self.sla_ms = sla_ms
        super().__init__(
            f"Latency SLA exceeded: {elapsed_ms:.1f}ms elapsed, SLA is {sla_ms:.1f}ms"
        )
