"""In-memory run store for l6e budget enforcement.

``InMemoryRunStore`` implements ``IRunStore`` and holds all ``CallRecord``
objects for one pipeline run entirely in memory. It tracks cumulative spend,
computes counterfactual costs when calls are rerouted to cheaper models, and
produces a ``RunSummary`` at the end of the run.

**source field and MCP tracking**

The ``source`` constructor parameter identifies the origin of the run. It
defaults to ``"pipeline"`` for direct SDK usage. The l6e MCP server sets it
to ``"mcp"`` when a session is started via the ``l6e_session_start`` tool, so
that runs initiated through MCP tool calls can be distinguished from SDK-driven
runs in ``runs.jsonl`` and any downstream analytics.
"""
from __future__ import annotations

import threading
from decimal import Decimal

from l6e._protocols import ICostEstimator
from l6e._types import CallRecord, PipelinePolicy, RunSummary


class InMemoryRunStore:
    """Implements IRunStore. Holds all CallRecords for one pipeline run in memory.

    Injected with an ICostEstimator so it can compute the counterfactual cost
    (what model_requested would have cost) when a call is rerouted to a cheaper
    model — the delta becomes savings_usd in the RunSummary.

    Thread-safety: ``record_call`` is safe to call from multiple threads.
    Note: ``to_summary()`` and ``budget_status()``-style reads are NOT protected
    by this lock — callers must not share a store across threads while writes
    are still in flight unless they add external coordination.
    """

    def __init__(
        self,
        run_id: str,
        policy: PipelinePolicy,
        estimator: ICostEstimator,
        source: str = "pipeline",
    ) -> None:
        self._run_id = run_id
        self._policy = policy
        self._estimator = estimator
        self._source = source
        self._records: list[CallRecord] = []
        self._total_cost: Decimal = Decimal("0")
        self._counterfactual_cost: Decimal = Decimal("0")
        self._lock = threading.Lock()

    # --- IRunStore protocol ---

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def budget(self) -> float:
        return self._policy.budget

    def record_call(self, record: CallRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._total_cost += record.cost_usd

            if record.rerouted and record.model_requested != record.model_used:
                # Compute what the requested model would have cost at the same token counts.
                counterfactual = self._estimator.estimate(
                    model=record.model_requested,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                )
                # If estimator returns 0 (unknown model), fall back to actual cost — no savings.
                self._counterfactual_cost += max(counterfactual, record.cost_usd)
            else:
                self._counterfactual_cost += record.cost_usd

    def spent(self) -> Decimal:
        return self._total_cost

    def remaining(self) -> Decimal:
        """Return the remaining budget in USD (budget minus total cost so far)."""
        return Decimal(str(self._policy.budget)) - self._total_cost

    def call_count(self) -> int:
        return len(self._records)

    def to_summary(self) -> RunSummary:
        reroutes = sum(1 for r in self._records if r.rerouted)
        savings = max(Decimal("0"), self._counterfactual_cost - self._total_cost)
        return RunSummary(
            run_id=self._run_id,
            policy=self._policy,
            total_cost=self._total_cost,
            calls_made=len(self._records),
            reroutes=reroutes,
            savings_usd=savings,
            records=tuple(self._records),
            source=self._source,
        )

    def export(self) -> RunSummary:
        """Cloud telemetry seam — identical to to_summary() in OSS."""
        return self.to_summary()
