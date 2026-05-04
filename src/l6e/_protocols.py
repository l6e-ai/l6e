"""Protocol interfaces for l6e collaborators.

All concrete implementations and fakes must satisfy these protocols.
Using Protocol (structural subtyping) keeps the core free of concrete dependencies.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from l6e._types import CallRecord, GateDecision, PromptComplexity, RunSummary


class ICostEstimator(Protocol):
    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
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

    def spent(self) -> Decimal:
        """Total USD spent so far in this run."""
        ...

    def remaining(self) -> Decimal:
        """Remaining budget in USD."""
        ...

    def call_count(self) -> int:
        """Number of calls recorded so far."""
        ...

    def to_summary(self) -> RunSummary:
        """Build a RunSummary from current state."""
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
        estimated_cost: Decimal,
        stage: str | None,
        complexity: PromptComplexity | None,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        cohort_hint: str | None = None,
        prompts: list[str] | None = None,
    ) -> GateDecision:
        """Decide whether to allow, reroute, or halt a pending LLM call.

        ``user_id``, ``tenant_id``, and ``cohort_hint`` are Margin-tier
        identity hints. Pure ``ConstraintGate`` ignores them; cloud-sync or
        cohort-aware gates may use them to select calibrated cost factors or
        per-tenant routing policy. All three are optional and default to
        ``None`` so pre-Margin call sites remain unchanged.

        ``prompts`` are the user-role prompt strings extracted from the
        pending request. Pure ``ConstraintGate`` ignores them; the
        cloud-sync ``RemoteConstraintGate`` consumes them when
        ``CloudConfig.privacy_tier="embeddings"`` to invoke the
        customer-supplied embedder before issuing the cloud call.
        Optional and defaulted to ``None`` so pre-embeddings call sites
        and custom gate implementations remain protocol-compliant.
        """
        ...
