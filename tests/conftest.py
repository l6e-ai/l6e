"""Shared fakes for l6e unit tests. Protocol-compliant, no mocks."""
from __future__ import annotations

from decimal import Decimal

from l6e._protocols import IConstraintGate, ICostEstimator, ILocalRouter, IRunStore
from l6e._types import (
    CallRecord,
    GateDecision,
    PipelinePolicy,
    PromptComplexity,
    RunSummary,
)


class FakeStore:
    """Fake IRunStore for unit tests."""

    def __init__(
        self,
        budget: float,
        spent_amount: Decimal | float,
        run_id: str = "fake-run-id",
    ) -> None:
        self._budget = budget
        self._spent = Decimal(str(spent_amount))
        self._run_id = run_id
        self._records: list[CallRecord] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def budget(self) -> float:
        return self._budget

    def record_call(self, record: CallRecord) -> None:
        self._records.append(record)

    def spent(self) -> Decimal:
        return self._spent

    def remaining(self) -> Decimal:
        return Decimal(str(self._budget)) - self._spent

    def call_count(self) -> int:
        return len(self._records)

    def to_summary(self) -> RunSummary:
        return RunSummary(
            run_id=self._run_id,
            policy=PipelinePolicy(budget=self._budget),
            total_cost=self._spent,
            calls_made=len(self._records),
            reroutes=0,
            savings_usd=Decimal("0"),
            records=tuple(self._records),
        )

    def export(self) -> RunSummary:
        return self.to_summary()


class FakeRouter:
    """Fake ILocalRouter for unit tests."""

    def __init__(self, model: str | None = "ollama/qwen2.5:7b") -> None:
        self._model = model

    def best_local_model(self) -> str | None:
        return self._model


class FakeCostEstimator:
    """Fake ICostEstimator for unit tests."""

    def __init__(self, cost: Decimal | float = Decimal("0.01")) -> None:
        self._cost = Decimal(str(cost))

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        return self._cost


class FakeGate:
    """Fake IConstraintGate. Returns a fixed GateDecision regardless of inputs."""

    def __init__(self, decision: GateDecision) -> None:
        self._decision = decision

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
    ) -> GateDecision:
        return self._decision


class SpyGate:
    """Records the last check() call args while returning a fixed decision."""

    def __init__(self, decision: GateDecision) -> None:
        self._decision = decision
        self.last_stage: str | None = None
        self.last_model: str | None = None
        self.last_user_id: str | None = None
        self.last_tenant_id: str | None = None
        self.last_cohort_hint: str | None = None

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
    ) -> GateDecision:
        self.last_stage = stage
        self.last_model = model
        self.last_user_id = user_id
        self.last_tenant_id = tenant_id
        self.last_cohort_hint = cohort_hint
        return self._decision


class FakeLog:
    """In-memory drop-in for LocalRunLog. Records appended summaries."""

    def __init__(self) -> None:
        self.entries: list[RunSummary] = []

    def append(self, summary: RunSummary) -> None:
        self.entries.append(summary)

    def read_recent(self, n: int = 100) -> list[RunSummary]:
        return list(self.entries[-n:])


# Satisfy type checker that fakes fully implement their protocols.
_: ICostEstimator = FakeCostEstimator()
__: ILocalRouter = FakeRouter()
___: IRunStore = FakeStore(budget=1.0, spent_amount=0.0)
____: IConstraintGate = FakeGate(
    GateDecision(action="allow", target_model="gpt-4o", reason="allow")
)
_____: IConstraintGate = SpyGate(
    GateDecision(action="allow", target_model="gpt-4o", reason="allow")
)
