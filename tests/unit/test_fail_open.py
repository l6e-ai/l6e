"""Iron-rule tests: the gate fails open on every failure path.

Companion to ``pivot-docs/cost-benchmark-margin-thesis/05-integration-architecture.md``
and the L6E-41 ticket. Every test here asserts that a broken
collaborator or invalid input degrades to "allow" (for ``advise``) or
"pass-through to fn" (for ``call``), never an uncaught exception into
customer code.

If you add a new public entry point on ``PipelineContext`` — or a new
collaborator it delegates to — add a corresponding test here. This file
is the iron-rule regression fence: CI enforces that no path from
``ctx.call`` / ``ctx.advise`` can throw.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from l6e._classify import PromptComplexityClassifier
from l6e._types import (
    BudgetMode,
    CallRecord,
    GateDecision,
    OnBudgetExceeded,
    PipelinePolicy,
    PromptComplexity,
    RunSummary,
)
from l6e.pipeline import PipelineContext
from tests.conftest import FakeCostEstimator, FakeGate, FakeLog, FakeStore

# ---------------------------------------------------------------------------
# Broken collaborator doubles
# ---------------------------------------------------------------------------


class _BrokenGate:
    """Gate that always raises. Simulates a corrupted rule-set or a bug."""

    def check(self, *args: Any, **kwargs: Any) -> GateDecision:
        raise RuntimeError("gate_exploded")


class _BrokenEstimator:
    """Cost estimator that raises — e.g. litellm pricing table panic."""

    def estimate(self, *args: Any, **kwargs: Any) -> Decimal:
        raise RuntimeError("estimator_exploded")


class _BrokenStore(FakeStore):
    """Store whose every read/write raises. Worst-case disk/DB failure."""

    def record_call(self, record: CallRecord) -> None:
        raise RuntimeError("store_exploded")

    def spent(self) -> Decimal:
        raise RuntimeError("store_exploded")

    def remaining(self) -> Decimal:
        raise RuntimeError("store_exploded")

    def call_count(self) -> int:
        raise RuntimeError("store_exploded")

    def to_summary(self) -> RunSummary:
        raise RuntimeError("store_exploded")


class _BrokenClassifier:
    """Classifier that raises — safe classifier wrapper should swallow."""

    def classify(self, *args: Any, **kwargs: Any) -> PromptComplexity:
        raise RuntimeError("classifier_exploded")


class _BrokenLog:
    """Log writer that raises on append. Disk full, permission denied, etc."""

    def append(self, summary: RunSummary) -> None:
        raise RuntimeError("log_exploded")

    def read_recent(self, n: int = 100) -> list[RunSummary]:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ALLOW = GateDecision(action="allow", target_model="gpt-4o", reason="allow")
_PROMPTS = ["Summarize this document."]
_FAKE_RESPONSE = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
_MESSAGES = [{"role": "user", "content": "Hello"}]


def _make_ctx(
    *,
    gate: Any = None,
    estimator: Any = None,
    store: Any = None,
    classifier: Any = None,
    log: Any = None,
    policy: PipelinePolicy | None = None,
) -> PipelineContext:
    pol = policy or PipelinePolicy(budget=1.00)
    return PipelineContext(
        run_id="iron-rule-test",
        policy=pol,
        gate=gate or FakeGate(_ALLOW),
        store=store or FakeStore(budget=pol.budget, spent_amount=0.0),
        log=log or FakeLog(),
        classifier=classifier or PromptComplexityClassifier(),
        estimator=estimator or FakeCostEstimator(cost=0.01),
    )


def _fn_echo(*, model: str, messages: list[dict[str, str]]) -> dict:
    """Stand-in for a customer's LLM call. Returns a well-formed response."""
    return {"model": model, "messages": messages, **_FAKE_RESPONSE}


# ---------------------------------------------------------------------------
# Iron rule — scenario 5 from the matrix: gate binary crashes in-process
# ---------------------------------------------------------------------------


class TestCallFailsOpenOnCollaboratorException:
    """``ctx.call`` must pass through to the customer's fn on any gate-side crash."""

    def test_call_passes_through_when_gate_raises(self) -> None:
        ctx = _make_ctx(gate=_BrokenGate())
        resp = ctx.call(
            fn=_fn_echo, model="gpt-4o", messages=_MESSAGES, stage="planning",
        )
        assert resp["model"] == "gpt-4o", (
            "Broken gate must not override the customer's requested model"
        )
        assert resp["messages"] == _MESSAGES

    def test_call_passes_through_when_estimator_raises(self) -> None:
        ctx = _make_ctx(estimator=_BrokenEstimator())
        resp = ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES)
        assert resp["model"] == "gpt-4o"

    def test_call_passes_through_when_classifier_raises(self) -> None:
        ctx = _make_ctx(classifier=_BrokenClassifier())
        resp = ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES)
        assert resp["model"] == "gpt-4o"

    def test_call_passes_through_when_store_raises(self) -> None:
        # Store.record_call happens AFTER fn has already returned. The
        # customer must still get their response; telemetry is
        # best-effort.
        ctx = _make_ctx(store=_BrokenStore(budget=1.00, spent_amount=0.0))
        resp = ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES)
        assert resp["model"] == "gpt-4o"

    def test_call_never_raises_even_when_every_collaborator_is_broken(self) -> None:
        """Belt-and-braces: complete gate failure still yields a response."""
        ctx = _make_ctx(
            gate=_BrokenGate(),
            estimator=_BrokenEstimator(),
            classifier=_BrokenClassifier(),
            store=_BrokenStore(budget=1.00, spent_amount=0.0),
            log=_BrokenLog(),
        )
        resp = ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES)
        assert resp["model"] == "gpt-4o"


class TestAdviseFailsOpen:
    """``ctx.advise`` must return allow on any internal exception."""

    def test_advise_returns_allow_when_gate_raises(self) -> None:
        ctx = _make_ctx(gate=_BrokenGate())
        decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS)
        assert decision.action == "allow"
        assert decision.target_model == "gpt-4o"
        assert decision.reason == "fail_open:gate_exception"

    def test_advise_returns_allow_when_estimator_raises(self) -> None:
        ctx = _make_ctx(estimator=_BrokenEstimator())
        decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS)
        assert decision.action == "allow"
        assert decision.reason == "fail_open:gate_exception"

    def test_advise_returns_allow_when_classifier_raises(self) -> None:
        ctx = _make_ctx(classifier=_BrokenClassifier())
        decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS)
        assert decision.action == "allow"

    def test_advise_handles_none_prompts_safely(self) -> None:
        ctx = _make_ctx()
        decision = ctx.advise(model="gpt-4o", prompts=[])
        assert decision.action == "allow"

    def test_advise_survives_non_string_prompt_entries(self) -> None:
        """Malformed prompt list (e.g. dicts leaked from caller) must not crash."""
        ctx = _make_ctx()
        decision = ctx.advise(
            model="gpt-4o",
            prompts=[{"nested": "object"}],  # type: ignore[list-item]
        )
        assert decision.action == "allow"


class TestRecordFailsOpen:
    def test_record_does_not_raise_when_response_extraction_fails(self) -> None:
        ctx = _make_ctx()
        # ``object()`` lacks usage shape; extract_token_usage returns 0,0
        # but any downstream mutation must not crash.
        record = ctx.record(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            response=object(),
            elapsed_ms=12.0,
        )
        assert record is not None

    def test_record_does_not_raise_when_estimator_raises(self) -> None:
        ctx = _make_ctx(estimator=_BrokenEstimator())
        record = ctx.record(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            response=_FAKE_RESPONSE,
            elapsed_ms=12.0,
        )
        assert record.cost_usd == Decimal("0")

    def test_record_does_not_raise_when_store_raises(self) -> None:
        ctx = _make_ctx(store=_BrokenStore(budget=1.00, spent_amount=0.0))
        record = ctx.record(
            model_requested="gpt-4o",
            model_used="gpt-4o",
            response=_FAKE_RESPONSE,
            elapsed_ms=12.0,
        )
        assert record is not None


class TestStatusHelpersFailOpen:
    def test_budget_status_returns_safe_snapshot_on_store_failure(self) -> None:
        ctx = _make_ctx(store=_BrokenStore(budget=1.00, spent_amount=0.0))
        status = ctx.budget_status()
        assert status.budget_pressure == "low"
        assert status.spent_usd == Decimal("0")

    def test_run_summary_returns_empty_on_store_failure(self) -> None:
        ctx = _make_ctx(store=_BrokenStore(budget=1.00, spent_amount=0.0))
        summary = ctx.run_summary()
        assert summary.total_cost == Decimal("0")
        assert summary.records == ()


class TestContextManagerExitFailsOpen:
    def test_exit_swallows_log_write_failures(self) -> None:
        ctx = _make_ctx(log=_BrokenLog())
        ctx.__enter__()
        # Must not raise. Disk-full at pipeline exit is not a
        # customer-facing error.
        ctx.__exit__(None, None, None)

    def test_exit_swallows_store_summary_failures(self) -> None:
        ctx = _make_ctx(store=_BrokenStore(budget=1.00, spent_amount=0.0))
        ctx.__enter__()
        ctx.__exit__(None, None, None)

    def test_exit_does_not_suppress_caller_exception(self) -> None:
        """Iron rule requires fail-open, but we must never *suppress* a
        real exception from the caller's ``with`` body. That would
        silently hide production bugs."""
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="from caller"), ctx:
            raise ValueError("from caller")


# ---------------------------------------------------------------------------
# Iron rule — scenario 6: customer misconfigures policy
# ---------------------------------------------------------------------------


class TestPolicyValidation:
    def test_rejects_negative_budget(self) -> None:
        with pytest.raises(ValueError, match="budget must be >= 0"):
            PipelinePolicy(budget=-1.0)

    def test_rejects_nan_budget(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            PipelinePolicy(budget=float("nan"))

    def test_rejects_inf_budget(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            PipelinePolicy(budget=float("inf"))

    def test_rejects_reroute_threshold_above_one(self) -> None:
        with pytest.raises(ValueError, match="reroute_threshold"):
            PipelinePolicy(budget=1.0, reroute_threshold=2.0)

    def test_rejects_negative_reroute_threshold(self) -> None:
        with pytest.raises(ValueError, match="reroute_threshold"):
            PipelinePolicy(budget=1.0, reroute_threshold=-0.1)

    def test_rejects_negative_unknown_model_cost(self) -> None:
        with pytest.raises(ValueError, match="unknown_model_cost"):
            PipelinePolicy(budget=1.0, unknown_model_cost_per_1k_tokens=-0.01)

    def test_accepts_zero_budget(self) -> None:
        """Zero budget is a legitimate "observe-only" mode — everything
        halts, nothing executes. Not a misconfig on its own."""
        pol = PipelinePolicy(budget=0.0)
        assert pol.budget == 0.0

    def test_accepts_threshold_at_boundaries(self) -> None:
        PipelinePolicy(budget=1.0, reroute_threshold=0.0)
        PipelinePolicy(budget=1.0, reroute_threshold=1.0)


# ---------------------------------------------------------------------------
# Iron rule interaction: halt behavior survives collaborator failures
# ---------------------------------------------------------------------------


class TestHaltRespectedEvenWhenStoreBroken:
    def test_on_budget_exceeded_raise_still_works_with_broken_store(self) -> None:
        """A customer who opts into RAISE still gets BudgetExceeded on a
        gate-decided halt, even if store.spent() is broken. We just
        surface spent=0 in the exception rather than leaking the store
        bug to the customer."""
        from l6e.exceptions import BudgetExceeded

        halt_decision = GateDecision(
            action="halt", target_model="gpt-4o", reason="budget_pressure:halt",
        )
        policy = PipelinePolicy(
            budget=1.0, on_budget_exceeded=OnBudgetExceeded.RAISE,
        )
        ctx = _make_ctx(
            policy=policy,
            gate=FakeGate(halt_decision),
            store=_BrokenStore(budget=1.00, spent_amount=0.0),
        )
        with pytest.raises(BudgetExceeded):
            ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES)


# ---------------------------------------------------------------------------
# Iron rule — stage routing still functional with healthy collaborators
# ---------------------------------------------------------------------------


def test_happy_path_unaffected_by_fail_open_wrappers() -> None:
    """Regression: fail-open wrappers must not alter the happy path."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_overrides={"expensive": BudgetMode.HALT},
        on_budget_exceeded=OnBudgetExceeded.RETURN_EMPTY,
    )
    ctx = _make_ctx(
        policy=policy, gate=FakeGate(_ALLOW),
    )
    resp = ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES, stage="any")
    assert resp["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# CI fence: enumerate every public method and prove it never raises
# ---------------------------------------------------------------------------


def test_no_public_method_raises_with_fully_broken_collaborators() -> None:
    """Iron-rule fence. If a future refactor adds a public method that
    bypasses the fail-open wrapper, this test should catch it — it
    exercises every method that could reasonably be called with all
    collaborators broken, and asserts nothing raises.

    Intentionally broad: we'd rather over-test and lock the contract
    than discover a regression in production.
    """
    ctx = _make_ctx(
        gate=_BrokenGate(),
        estimator=_BrokenEstimator(),
        classifier=_BrokenClassifier(),
        store=_BrokenStore(budget=1.00, spent_amount=0.0),
        log=_BrokenLog(),
    )

    # call()
    assert ctx.call(fn=_fn_echo, model="gpt-4o", messages=_MESSAGES) is not None
    # advise()
    assert ctx.advise(model="gpt-4o", prompts=_PROMPTS).action == "allow"
    # record()
    assert ctx.record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        response=_FAKE_RESPONSE,
        elapsed_ms=1.0,
    ) is not None
    # run_summary()
    assert ctx.run_summary() is not None
    # budget_status()
    assert ctx.budget_status() is not None
    # __exit__()
    ctx.__exit__(None, None, None)
