"""Unit tests for ctx.call() — universal LLM wrapper on PipelineContext."""
from __future__ import annotations

import pytest

from l6e._types import GateDecision, OnBudgetExceeded, PipelinePolicy, StageRoutingHint
from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext
from tests.conftest import FakeCostEstimator, FakeGate, FakeLog, FakeRouter, FakeStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW = GateDecision(action="allow", target_model="gpt-4o", reason="allow")
_REROUTE = GateDecision(
    action="reroute", target_model="ollama/qwen2.5:7b", reason="stage_routing:local"
)
_HALT = GateDecision(action="halt", target_model="gpt-4o", reason="budget_pressure:halt")

_FAKE_RESPONSE = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}


def make_ctx(
    *,
    policy: PipelinePolicy | None = None,
    gate: FakeGate | None = None,
) -> PipelineContext:
    from l6e._classify import PromptComplexityClassifier

    pol = policy or PipelinePolicy(budget=1.00)
    store = FakeStore(budget=pol.budget, spent_amount=0.0)
    return PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=gate or FakeGate(_ALLOW),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )


def make_fn(response: object = None):
    """Return a fake LLM callable that records calls."""
    calls: list[tuple[str, list[dict[str, str]]]] = []

    def fn(model: str, messages: list[dict[str, str]]) -> object:
        calls.append((model, messages))
        return response if response is not None else _FAKE_RESPONSE

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


_MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Allow path
# ---------------------------------------------------------------------------


def test_call_executes_fn_with_original_model_on_allow() -> None:
    fn = make_fn()
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    ctx.call(fn=fn, model="gpt-4o", messages=_MESSAGES)
    assert fn.calls[0][0] == "gpt-4o"  # type: ignore[attr-defined]


def test_call_returns_fn_response_on_allow() -> None:
    sentinel = {"custom": "response", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    fn = make_fn(response=sentinel)
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    result = ctx.call(fn=fn, model="gpt-4o", messages=_MESSAGES)
    assert result is sentinel


def test_call_records_response_after_execution() -> None:
    from l6e._classify import PromptComplexityClassifier

    pol = PipelinePolicy(budget=1.00)
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=FakeGate(_ALLOW),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    ctx.call(fn=make_fn(), model="gpt-4o", messages=_MESSAGES)
    assert store.call_count() == 1


# ---------------------------------------------------------------------------
# Reroute path
# ---------------------------------------------------------------------------


def test_call_executes_fn_with_target_model_on_reroute() -> None:
    fn = make_fn()
    ctx = make_ctx(gate=FakeGate(_REROUTE))
    ctx.call(fn=fn, model="gpt-4o", messages=_MESSAGES)
    assert fn.calls[0][0] == "ollama/qwen2.5:7b"  # type: ignore[attr-defined]


def test_call_sets_rerouted_flag_in_record() -> None:
    from l6e._classify import PromptComplexityClassifier

    pol = PipelinePolicy(budget=1.00)
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=FakeGate(_REROUTE),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    ctx.call(fn=make_fn(), model="gpt-4o", messages=_MESSAGES)
    record = store._records[0]
    assert record.rerouted is True


def test_call_record_model_requested_is_original_model() -> None:
    from l6e._classify import PromptComplexityClassifier

    pol = PipelinePolicy(budget=1.00)
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=FakeGate(_REROUTE),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    ctx.call(fn=make_fn(), model="gpt-4o", messages=_MESSAGES)
    record = store._records[0]
    assert record.model_requested == "gpt-4o"
    assert record.model_used == "ollama/qwen2.5:7b"


# ---------------------------------------------------------------------------
# Halt path
# ---------------------------------------------------------------------------


def test_call_raises_budget_exceeded_on_halt() -> None:
    policy = PipelinePolicy(budget=1.00, on_budget_exceeded=OnBudgetExceeded.RAISE)
    ctx = make_ctx(policy=policy, gate=FakeGate(_HALT))
    with pytest.raises(BudgetExceeded):
        ctx.call(fn=make_fn(), model="gpt-4o", messages=_MESSAGES)


def test_call_fn_not_called_on_halt() -> None:
    fn = make_fn()
    policy = PipelinePolicy(budget=1.00, on_budget_exceeded=OnBudgetExceeded.RAISE)
    ctx = make_ctx(policy=policy, gate=FakeGate(_HALT))
    with pytest.raises(BudgetExceeded):
        ctx.call(fn=fn, model="gpt-4o", messages=_MESSAGES)
    assert len(fn.calls) == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# elapsed_ms captured
# ---------------------------------------------------------------------------


def test_call_captures_elapsed_ms_in_record() -> None:
    from l6e._classify import PromptComplexityClassifier

    pol = PipelinePolicy(budget=1.00)
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=FakeGate(_ALLOW),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    ctx.call(fn=make_fn(), model="gpt-4o", messages=_MESSAGES)
    record = store._records[0]
    assert record.elapsed_ms >= 0.0


# ---------------------------------------------------------------------------
# Stage passed to gate
# ---------------------------------------------------------------------------


def test_call_with_stage_routing_local_reroutes() -> None:
    from l6e._classify import PromptComplexityClassifier
    from l6e.gate import ConstraintGate

    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"summarization": StageRoutingHint.LOCAL},
    )
    store = FakeStore(budget=1.00, spent_amount=0.0)
    gate = ConstraintGate(policy=policy, router=FakeRouter(model="ollama/qwen2.5:7b"))
    ctx = PipelineContext(
        run_id="test-run",
        policy=policy,
        gate=gate,
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    fn = make_fn()
    ctx.call(fn=fn, model="gpt-4o", messages=_MESSAGES, stage="summarization")
    assert fn.calls[0][0] == "ollama/qwen2.5:7b"  # type: ignore[attr-defined]
