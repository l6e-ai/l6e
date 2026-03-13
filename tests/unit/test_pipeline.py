"""Unit tests for pipeline.py — PipelineContext."""
from __future__ import annotations

import pytest

from l6e._types import (
    BudgetMode,
    GateDecision,
    OnBudgetExceeded,
    PipelinePolicy,
    StageRoutingHint,
)
from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext
from tests.conftest import FakeCostEstimator, FakeGate, FakeLog, FakeRouter, FakeStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW = GateDecision(action="allow", target_model="gpt-4o", reason="allow")
_HALT = GateDecision(action="halt", target_model="gpt-4o", reason="budget_pressure:halt")
_REROUTE = GateDecision(
    action="reroute", target_model="ollama/qwen2.5:7b", reason="stage_routing:local"
)


def make_ctx(
    *,
    policy: PipelinePolicy | None = None,
    gate: FakeGate | None = None,
    store: FakeStore | None = None,
    log: FakeLog | None = None,
    estimator: FakeCostEstimator | None = None,
) -> PipelineContext:
    from l6e._classify import PromptComplexityClassifier

    pol = policy or PipelinePolicy(budget=1.00)
    return PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=gate or FakeGate(_ALLOW),
        store=store or FakeStore(budget=pol.budget, spent_amount=0.0),
        log=log or FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=estimator or FakeCostEstimator(cost=0.01),
    )


_PROMPTS = ["Summarize this document."]
_FAKE_RESPONSE = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_pipeline_factory_returns_pipeline_context() -> None:
    from l6e.pipeline import pipeline

    ctx = pipeline(policy=PipelinePolicy(budget=1.00))
    assert isinstance(ctx, PipelineContext)


def test_pipeline_context_enter_returns_self() -> None:
    ctx = make_ctx()
    assert ctx.__enter__() is ctx


def test_pipeline_context_exit_does_not_raise() -> None:
    ctx = make_ctx()
    ctx.__enter__()
    ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# advise()
# ---------------------------------------------------------------------------


def test_advise_returns_allow_when_gate_allows() -> None:
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS)
    assert decision.action == "allow"


def test_advise_returns_halt_when_gate_halts() -> None:
    ctx = make_ctx(gate=FakeGate(_HALT))
    decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS)
    assert decision.action == "halt"


def test_advise_returns_reroute_with_local_model() -> None:
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"summarization": StageRoutingHint.LOCAL},
    )
    from l6e._classify import PromptComplexityClassifier
    from l6e.gate import ConstraintGate

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
    decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS, stage="summarization")
    assert decision.action == "reroute"
    assert decision.target_model.startswith("ollama/")


def test_advise_passes_stage_to_gate() -> None:
    policy = PipelinePolicy(
        budget=1.00,
        stage_overrides={"critical_stage": BudgetMode.HALT},
    )
    from l6e._classify import PromptComplexityClassifier
    from l6e.gate import ConstraintGate

    store = FakeStore(budget=1.00, spent_amount=0.0)
    gate = ConstraintGate(policy=policy, router=FakeRouter())
    ctx = PipelineContext(
        run_id="test-run",
        policy=policy,
        gate=gate,
        store=store,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )
    decision = ctx.advise(model="gpt-4o", prompts=_PROMPTS, stage="critical_stage")
    assert decision.action == "halt"


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


def test_record_accumulates_cost_in_store() -> None:
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = make_ctx(store=store, estimator=FakeCostEstimator(cost=0.05))
    ctx.record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        response=_FAKE_RESPONSE,
        elapsed_ms=200.0,
    )
    assert store.call_count() == 1


def test_record_returns_call_record() -> None:
    from l6e._types import CallRecord

    ctx = make_ctx()
    rec = ctx.record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        response=_FAKE_RESPONSE,
        elapsed_ms=150.0,
        stage="summarization",
    )
    assert isinstance(rec, CallRecord)
    assert rec.model_requested == "gpt-4o"
    assert rec.model_used == "gpt-4o"
    assert rec.stage == "summarization"
    assert rec.elapsed_ms == pytest.approx(150.0)


def test_record_sets_rerouted_flag() -> None:
    ctx = make_ctx()
    rec = ctx.record(
        model_requested="gpt-4o",
        model_used="ollama/qwen2.5:7b",
        response=_FAKE_RESPONSE,
        elapsed_ms=80.0,
        rerouted=True,
    )
    assert rec.rerouted is True


def test_record_extracts_token_counts_from_response() -> None:
    ctx = make_ctx()
    rec = ctx.record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        response=_FAKE_RESPONSE,
        elapsed_ms=100.0,
    )
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 50


def test_record_call_index_increments() -> None:
    ctx = make_ctx()
    r1 = ctx.record(
        model_requested="gpt-4o", model_used="gpt-4o",
        response=_FAKE_RESPONSE, elapsed_ms=50.0,
    )
    r2 = ctx.record(
        model_requested="gpt-4o", model_used="gpt-4o",
        response=_FAKE_RESPONSE, elapsed_ms=50.0,
    )
    assert r2.call_index == r1.call_index + 1


# ---------------------------------------------------------------------------
# budget_status()
# ---------------------------------------------------------------------------


def test_budget_status_pressure_low_when_fresh() -> None:
    ctx = make_ctx(store=FakeStore(budget=1.00, spent_amount=0.0))
    status = ctx.budget_status()
    assert status.budget_pressure == "low"


def test_budget_status_pressure_moderate() -> None:
    ctx = make_ctx(store=FakeStore(budget=1.00, spent_amount=0.60))
    status = ctx.budget_status()
    assert status.budget_pressure == "moderate"


def test_budget_status_pressure_high() -> None:
    ctx = make_ctx(store=FakeStore(budget=1.00, spent_amount=0.85))
    status = ctx.budget_status()
    assert status.budget_pressure == "high"


def test_budget_status_pressure_critical_near_exhausted() -> None:
    ctx = make_ctx(store=FakeStore(budget=1.00, spent_amount=0.96))
    status = ctx.budget_status()
    assert status.budget_pressure == "critical"


def test_budget_status_pct_used_matches_spent() -> None:
    ctx = make_ctx(store=FakeStore(budget=1.00, spent_amount=0.40))
    status = ctx.budget_status()
    assert status.pct_used == pytest.approx(40.0)


def test_budget_status_fields_correct() -> None:
    store = FakeStore(budget=2.00, spent_amount=0.50, run_id="run-42")
    ctx = make_ctx(policy=PipelinePolicy(budget=2.00), store=store)
    status = ctx.budget_status()
    assert status.run_id == "run-42"
    assert status.spent_usd == pytest.approx(0.50)
    assert status.remaining_usd == pytest.approx(1.50)
    assert status.budget_usd == pytest.approx(2.00)


# ---------------------------------------------------------------------------
# __exit__ writes run log
# ---------------------------------------------------------------------------


def test_exit_appends_to_run_log(tmp_path) -> None:
    from l6e.pipeline import pipeline

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    ctx = pipeline(run_id="run-log-test", policy=PipelinePolicy(budget=1.00), log_path=log_path)
    with ctx:
        pass
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


def test_exit_appends_even_on_exception(tmp_path) -> None:
    from l6e.pipeline import pipeline

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    ctx = pipeline(run_id="run-exc-test", policy=PipelinePolicy(budget=1.00), log_path=log_path)
    with pytest.raises(ValueError), ctx:
        raise ValueError("intentional")
    assert log_path.exists()
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# Halt enforcement via ctx.call()
# ---------------------------------------------------------------------------


def test_call_raises_budget_exceeded_on_halt_with_raise_mode() -> None:
    policy = PipelinePolicy(budget=1.00, on_budget_exceeded=OnBudgetExceeded.RAISE)
    ctx = make_ctx(policy=policy, gate=FakeGate(_HALT))
    with pytest.raises(BudgetExceeded):
        ctx.call(fn=lambda m, msgs: {}, model="gpt-4o", messages=[])


def test_call_returns_empty_on_halt_with_empty_mode() -> None:
    policy = PipelinePolicy(budget=1.00, on_budget_exceeded=OnBudgetExceeded.RETURN_EMPTY)
    ctx = make_ctx(policy=policy, gate=FakeGate(_HALT))
    result = ctx.call(fn=lambda m, msgs: {"content": "real"}, model="gpt-4o", messages=[])
    assert result == ""


def test_call_returns_fallback_on_halt_with_fallback_mode() -> None:
    policy = PipelinePolicy(
        budget=1.00,
        on_budget_exceeded=OnBudgetExceeded.RETURN_FALLBACK,
        fallback_result="fallback-answer",
    )
    ctx = make_ctx(policy=policy, gate=FakeGate(_HALT))
    result = ctx.call(fn=lambda m, msgs: {}, model="gpt-4o", messages=[])
    assert result == "fallback-answer"


# ---------------------------------------------------------------------------
# _estimate_prompt_tokens tiktoken failure fallback (lines 38-39)
# ---------------------------------------------------------------------------


def test_estimate_prompt_tokens_fallback_when_tiktoken_fails(monkeypatch) -> None:
    """Lines 38-39: if tiktoken raises, falls back to max(1, len(text) // 4)."""
    import tiktoken

    from l6e.pipeline import _estimate_prompt_tokens

    def _raise(_name):
        raise RuntimeError("no tiktoken")

    monkeypatch.setattr(tiktoken, "get_encoding", _raise)

    text = "hello world"
    result = _estimate_prompt_tokens([text])
    expected = max(1, len(text) // 4)
    assert result == expected


def test_estimate_prompt_tokens_fallback_minimum_is_one(monkeypatch) -> None:
    """Even for an empty string, fallback returns at least 1."""
    import tiktoken

    from l6e.pipeline import _estimate_prompt_tokens

    def _raise(_name):
        raise RuntimeError("no tiktoken")

    monkeypatch.setattr(tiktoken, "get_encoding", _raise)

    result = _estimate_prompt_tokens([""])
    assert result >= 1


# ---------------------------------------------------------------------------
# run_summary() (line 148)
# ---------------------------------------------------------------------------


def test_run_summary_returns_run_summary_with_correct_run_id() -> None:
    from l6e._types import RunSummary

    store = FakeStore(budget=1.00, spent_amount=0.0, run_id="my-run")
    ctx = make_ctx(store=store)
    summary = ctx.run_summary()
    assert isinstance(summary, RunSummary)
    assert summary.run_id == "my-run"


def test_run_summary_reflects_recorded_calls() -> None:
    from l6e._types import RunSummary

    store = FakeStore(budget=1.00, spent_amount=0.0, run_id="run-x")
    ctx = make_ctx(store=store)
    ctx.record(
        model_requested="gpt-4o",
        model_used="gpt-4o",
        response=_FAKE_RESPONSE,
        elapsed_ms=100.0,
    )
    summary = ctx.run_summary()
    assert isinstance(summary, RunSummary)
    assert summary.calls_made == 1


# ---------------------------------------------------------------------------
# run_id optional — UUID default (Fix 6)
# ---------------------------------------------------------------------------


def test_pipeline_factory_generates_uuid_run_id_when_omitted() -> None:
    """Calling pipeline(policy=...) with no run_id must auto-generate a UUID."""
    import re

    from l6e.pipeline import pipeline

    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    policy = PipelinePolicy(budget=1.00)
    ctx1 = pipeline(policy=policy)
    ctx2 = pipeline(policy=policy)

    assert UUID_RE.match(ctx1.run_id), f"Expected UUID, got: {ctx1.run_id}"
    assert UUID_RE.match(ctx2.run_id), f"Expected UUID, got: {ctx2.run_id}"
    assert ctx1.run_id != ctx2.run_id, "Two calls must produce different run_ids"


def test_pipeline_factory_uses_supplied_run_id() -> None:
    """When run_id is explicitly supplied it must be used verbatim."""
    from l6e.pipeline import pipeline

    ctx = pipeline(run_id="my-run", policy=PipelinePolicy(budget=1.00))
    assert ctx.run_id == "my-run"


# ---------------------------------------------------------------------------
# Thread-safety — call_index uniqueness (Fix 7)
# ---------------------------------------------------------------------------


def test_record_call_index_is_thread_safe() -> None:
    """10 threads × 20 calls on a real pipeline() must each receive a unique
    call_index.  A missing lock on _call_index += 1 can produce duplicates."""
    import threading

    from l6e.pipeline import pipeline

    THREADS = 10
    CALLS_PER_THREAD = 20
    _FAKE = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    ctx = pipeline(policy=PipelinePolicy(budget=1000.00))

    call_indices: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(CALLS_PER_THREAD):
            rec = ctx.record(
                model_requested="gpt-4o",
                model_used="gpt-4o",
                response=_FAKE,
                elapsed_ms=1.0,
            )
            with lock:
                call_indices.append(rec.call_index)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(call_indices) == THREADS * CALLS_PER_THREAD
    assert len(set(call_indices)) == len(call_indices), (
        f"Duplicate call_index values detected: "
        f"{[i for i in call_indices if call_indices.count(i) > 1]}"
    )
