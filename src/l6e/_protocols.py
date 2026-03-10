"""Protocol interfaces for l6e collaborators.

All concrete implementations and fakes must satisfy these protocols.
Using Protocol (structural subtyping) keeps the core free of concrete dependencies.
"""
from __future__ import annotations

from typing import Protocol

from l6e._types import CallRecord, GateDecision, PromptComplexity, RunSummary


class ICostEstimator(Protocol):
    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD for the given token counts."""
        ...


class IRunStore(Protocol):
    @property
    def run_id(self) -> str:
        """Unique identifier for this pipeline run."""
        ...

    @property
    def budget(self) -> float:
        """Total budget in USD for this run."""
        ...

    def record_call(self, record: CallRecord) -> None:
        """Append a completed call record to the run."""
        ...

    def spent(self) -> float:
        """Total USD spent so far in this run."""
        ...

    def remaining(self) -> float:
        """Remaining budget in USD."""
        ...

    def call_count(self) -> int:
        """Number of calls recorded so far."""
        ...

    def to_summary(self) -> RunSummary:
        """Build a RunSummary from current state."""
        ...

    def export(self) -> RunSummary:
        """Cloud telemetry seam — returns summary for upload. Same as to_summary() in OSS."""
        ...


class ILocalRouter(Protocol):
    def best_local_model(self) -> str | None:
        """Return the best locally available model tag, or None if no local inference available."""
        ...


class IConstraintGate(Protocol):
    def check(
        self,
        store: IRunStore,
        model: str,
        estimated_cost: float,
        stage: str | None,
        complexity: PromptComplexity | None,
    ) -> GateDecision:
        """Decide whether to allow, reroute, or halt a pending LLM call."""
        ...
