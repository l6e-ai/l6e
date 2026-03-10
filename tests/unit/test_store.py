"""Unit tests for store.py — InMemoryRunStore."""
from __future__ import annotations

import pytest

from l6e._types import (
    CallRecord,
    PipelinePolicy,
)


def make_policy(budget: float = 1.00) -> PipelinePolicy:
    return PipelinePolicy(budget=budget)


def make_record(
    *,
    call_index: int = 0,
    model_requested: str = "gpt-4o",
    model_used: str = "gpt-4o",
    cost_usd: float = 0.01,
    rerouted: bool = False,
    stage: str | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> CallRecord:
    return CallRecord(
        call_index=call_index,
        model_requested=model_requested,
        model_used=model_used,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
        rerouted=rerouted,
        elapsed_ms=100.0,
        stage=stage,
    )


@pytest.fixture
def estimator():
    from tests.conftest import FakeCostEstimator

    return FakeCostEstimator(cost=0.05)


@pytest.fixture
def store(estimator):
    from l6e.store import InMemoryRunStore

    return InMemoryRunStore(
        run_id="test-run-1",
        policy=make_policy(budget=1.00),
        estimator=estimator,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_fresh_store_spent_is_zero(store) -> None:
    assert store.spent() == 0.0


def test_fresh_store_remaining_equals_budget(store) -> None:
    assert store.remaining() == 1.00


def test_fresh_store_call_count_is_zero(store) -> None:
    assert store.call_count() == 0


def test_run_id_property(store) -> None:
    assert store.run_id == "test-run-1"


def test_budget_property(store) -> None:
    assert store.budget == 1.00


# ---------------------------------------------------------------------------
# After record_call
# ---------------------------------------------------------------------------


def test_spent_accumulates_after_record(store) -> None:
    store.record_call(make_record(cost_usd=0.10))
    store.record_call(make_record(call_index=1, cost_usd=0.05))
    assert pytest.approx(store.spent()) == 0.15


def test_remaining_decreases_after_record(store) -> None:
    store.record_call(make_record(cost_usd=0.25))
    assert pytest.approx(store.remaining()) == 0.75


def test_call_count_increments(store) -> None:
    store.record_call(make_record(call_index=0))
    store.record_call(make_record(call_index=1))
    assert store.call_count() == 2


# ---------------------------------------------------------------------------
# to_summary / export
# ---------------------------------------------------------------------------


def test_to_summary_reroutes_count(store) -> None:
    store.record_call(make_record(call_index=0, rerouted=False))
    store.record_call(make_record(call_index=1, rerouted=True, model_used="ollama/qwen2.5:7b"))
    store.record_call(make_record(call_index=2, rerouted=True, model_used="ollama/qwen2.5:7b"))
    summary = store.to_summary()
    assert summary.reroutes == 2


def test_to_summary_records_is_tuple(store) -> None:
    store.record_call(make_record())
    summary = store.to_summary()
    assert isinstance(summary.records, tuple)


def test_to_summary_records_length(store) -> None:
    store.record_call(make_record(call_index=0))
    store.record_call(make_record(call_index=1))
    summary = store.to_summary()
    assert len(summary.records) == 2


def test_to_summary_total_cost(store) -> None:
    store.record_call(make_record(call_index=0, cost_usd=0.03))
    store.record_call(make_record(call_index=1, cost_usd=0.07))
    summary = store.to_summary()
    assert pytest.approx(summary.total_cost) == 0.10


def test_export_equals_to_summary(store) -> None:
    store.record_call(make_record())
    assert store.export() == store.to_summary()


# ---------------------------------------------------------------------------
# savings_usd — counterfactual cost
# ---------------------------------------------------------------------------


def test_savings_zero_when_no_reroutes(store) -> None:
    # model_requested == model_used → no savings
    store.record_call(make_record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        cost_usd=0.05,
        rerouted=False,
    ))
    assert store.to_summary().savings_usd == 0.0


def test_savings_positive_when_rerouted_to_cheaper_model() -> None:
    """Rerouted call: model_requested would have cost more than model_used."""
    from l6e.store import InMemoryRunStore

    class TieredEstimator:
        """Returns different costs per model to simulate cheaper reroute."""
        def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
            if "gpt-4o" in model and "mini" not in model:
                return 0.10
            return 0.01  # cheaper model

    store = InMemoryRunStore(
        run_id="r1",
        policy=make_policy(budget=1.00),
        estimator=TieredEstimator(),
    )
    store.record_call(make_record(
        model_requested="gpt-4o",
        model_used="ollama/qwen2.5:7b",
        cost_usd=0.01,
        rerouted=True,
        prompt_tokens=100,
        completion_tokens=50,
    ))
    summary = store.to_summary()
    assert summary.savings_usd > 0.0


def test_to_summary_calls_made(store) -> None:
    for i in range(5):
        store.record_call(make_record(call_index=i))
    assert store.to_summary().calls_made == 5
