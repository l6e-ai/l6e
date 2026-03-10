"""Unit tests for adapters/crewai.py — L6eStepCallback."""
from __future__ import annotations

import pytest

from l6e._classify import PromptComplexityClassifier
from l6e._types import GateDecision, PipelinePolicy
from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext
from tests.conftest import FakeCostEstimator, FakeGate, FakeLog, FakeStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW = GateDecision(action="allow", target_model="gpt-4o", reason="allow")
_HALT = GateDecision(action="halt", target_model="gpt-4o", reason="budget_pressure:halt")
_REROUTE = GateDecision(
    action="reroute", target_model="ollama/qwen2.5:7b", reason="budget_pressure:reroute"
)


def make_ctx(
    *,
    policy: PipelinePolicy | None = None,
    gate: FakeGate | None = None,
    spent: float = 0.0,
) -> PipelineContext:
    pol = policy or PipelinePolicy(budget=1.00)
    store = FakeStore(budget=pol.budget, spent_amount=spent)
    return PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=gate or FakeGate(_ALLOW),
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.0),
    )


def make_callback(ctx: PipelineContext, stage: str | None = None):
    from l6e.adapters.crewai import L6eStepCallback
    return L6eStepCallback(ctx, stage=stage)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_callback_does_not_raise_when_gate_allows() -> None:
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    cb = make_callback(ctx)
    cb(object())  # step_output is ignored


def test_step_callback_raises_budget_exceeded_when_gate_halts() -> None:
    ctx = make_ctx(gate=FakeGate(_HALT))
    cb = make_callback(ctx)
    with pytest.raises(BudgetExceeded):
        cb(object())


def test_step_callback_does_not_raise_when_gate_reroutes() -> None:
    """Reroute is advisory in v0.1 — step proceeds."""
    ctx = make_ctx(gate=FakeGate(_REROUTE))
    cb = make_callback(ctx)
    cb(object())  # should not raise
