"""Unit tests for adapters/langchain.py — L6eCallbackHandler."""
from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from l6e._classify import PromptComplexityClassifier
from l6e._types import GateDecision, PipelinePolicy
from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext
from tests.conftest import FakeCostEstimator, FakeGate, FakeLog, FakeStore, SpyGate

# Skip every test in this module if langchain_core is not installed.
langchain_core = pytest.importorskip("langchain_core")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOW = GateDecision(action="allow", target_model="gpt-4o", reason="allow")
_HALT = GateDecision(action="halt", target_model="gpt-4o", reason="budget_pressure:halt")
_REROUTE = GateDecision(
    action="reroute", target_model="ollama/qwen2.5:7b", reason="stage_routing:local"
)

_SERIALIZED: dict = {"id": ["langchain_community", "chat_models", "openai", "ChatOpenAI"]}
_PROMPTS = ["Summarize this document."]
_FAKE_RESPONSE = type("FakeLLMResult", (), {
    "llm_output": {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}},
})()


def make_ctx(
    *,
    policy: PipelinePolicy | None = None,
    gate: FakeGate | SpyGate | None = None,
    store: FakeStore | None = None,
) -> PipelineContext:
    pol = policy or PipelinePolicy(budget=1.00)
    st = store or FakeStore(budget=pol.budget, spent_amount=0.0)
    return PipelineContext(
        run_id="test-run",
        policy=pol,
        gate=gate or FakeGate(_ALLOW),
        store=st,
        log=FakeLog(),
        classifier=PromptComplexityClassifier(),
        estimator=FakeCostEstimator(cost=0.01),
    )


def make_handler(ctx: PipelineContext, *, infer_stage: bool = True):
    from l6e.adapters.langchain import L6eCallbackHandler
    return L6eCallbackHandler(ctx, infer_stage=infer_stage)


def _run_id() -> UUID:
    return uuid4()


# ---------------------------------------------------------------------------
# on_llm_start — allow / halt
# ---------------------------------------------------------------------------


def test_on_llm_start_allow_does_not_raise() -> None:
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    handler = make_handler(ctx)
    rid = _run_id()
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=rid)


def test_on_llm_start_halt_raises_budget_exceeded() -> None:
    ctx = make_ctx(gate=FakeGate(_HALT))
    handler = make_handler(ctx)
    with pytest.raises(BudgetExceeded):
        handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=_run_id())


# ---------------------------------------------------------------------------
# on_llm_start — stage extraction
# ---------------------------------------------------------------------------


def test_on_llm_start_extracts_stage_from_tags() -> None:
    spy = SpyGate(_ALLOW)
    ctx = make_ctx(gate=spy)
    handler = make_handler(ctx)
    handler.on_llm_start(
        _SERIALIZED, _PROMPTS, run_id=_run_id(),
        tags=["l6e_stage:summarization", "other_tag"],
    )
    assert spy.last_stage == "summarization"


def test_on_llm_start_no_stage_tag_passes_none() -> None:
    spy = SpyGate(_ALLOW)
    ctx = make_ctx(gate=spy)
    handler = make_handler(ctx, infer_stage=False)
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=_run_id(), tags=["unrelated"])
    assert spy.last_stage is None


def test_on_llm_start_no_tags_kwarg_passes_none() -> None:
    spy = SpyGate(_ALLOW)
    ctx = make_ctx(gate=spy)
    handler = make_handler(ctx, infer_stage=False)
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=_run_id())
    assert spy.last_stage is None


# ---------------------------------------------------------------------------
# on_llm_start — model name extraction
# ---------------------------------------------------------------------------


def test_on_llm_start_extracts_model_from_invocation_params() -> None:
    spy = SpyGate(_ALLOW)
    ctx = make_ctx(gate=spy)
    handler = make_handler(ctx)
    handler.on_llm_start(
        _SERIALIZED, _PROMPTS, run_id=_run_id(),
        invocation_params={"model": "gpt-4o"},
    )
    assert spy.last_model == "gpt-4o"


def test_on_llm_start_falls_back_to_serialized_name() -> None:
    spy = SpyGate(_ALLOW)
    ctx = make_ctx(gate=spy)
    handler = make_handler(ctx)
    handler.on_llm_start(
        {"name": "gpt-4o-mini"}, _PROMPTS, run_id=_run_id(),
    )
    assert spy.last_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# on_llm_end
# ---------------------------------------------------------------------------


def test_on_llm_end_records_call() -> None:
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = make_ctx(gate=FakeGate(_ALLOW), store=store)
    handler = make_handler(ctx)
    rid = _run_id()
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=rid)
    handler.on_llm_end(_FAKE_RESPONSE, run_id=rid)
    assert store.call_count() == 1


def test_on_llm_end_elapsed_ms_nonnegative() -> None:
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = make_ctx(gate=FakeGate(_ALLOW), store=store)
    handler = make_handler(ctx)
    rid = _run_id()
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=rid)
    handler.on_llm_end(_FAKE_RESPONSE, run_id=rid)
    record = store._records[0]
    assert record.elapsed_ms >= 0.0


def test_on_llm_end_missing_run_id_does_not_raise() -> None:
    ctx = make_ctx(gate=FakeGate(_ALLOW))
    handler = make_handler(ctx)
    # on_llm_end called without a prior on_llm_start for this run_id
    handler.on_llm_end(_FAKE_RESPONSE, run_id=_run_id())


def test_on_llm_end_llm_output_token_usage_extracted() -> None:
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = make_ctx(gate=FakeGate(_ALLOW), store=store)
    handler = make_handler(ctx)
    rid = _run_id()
    handler.on_llm_start(_SERIALIZED, _PROMPTS, run_id=rid)
    handler.on_llm_end(_FAKE_RESPONSE, run_id=rid)
    record = store._records[0]
    assert record.prompt_tokens == 100
    assert record.completion_tokens == 50


def test_on_llm_end_reroute_records_target_model_as_model_used() -> None:
    store = FakeStore(budget=1.00, spent_amount=0.0)
    ctx = make_ctx(gate=FakeGate(_REROUTE), store=store)
    handler = make_handler(ctx)
    rid = _run_id()
    handler.on_llm_start(
        _SERIALIZED, _PROMPTS, run_id=rid,
        invocation_params={"model": "gpt-4o"},
    )
    handler.on_llm_end(_FAKE_RESPONSE, run_id=rid)
    record = store._records[0]
    assert record.model_requested == "gpt-4o"
    assert record.model_used == "ollama/qwen2.5:7b"
    assert record.rerouted is True


# ---------------------------------------------------------------------------
# ImportError guard — lines 14-15: langchain_core absent
# ---------------------------------------------------------------------------


def test_import_error_raised_when_langchain_core_absent(monkeypatch) -> None:
    """Lines 14-15: importing the langchain adapter without langchain_core raises ImportError."""
    import importlib
    import sys

    # Remove the cached module so the import guard runs fresh
    monkeypatch.delitem(sys.modules, "l6e.adapters.langchain", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_core", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "langchain_core.callbacks", None)  # type: ignore[arg-type]

    with pytest.raises(ImportError, match="langchain-core is required"):
        importlib.import_module("l6e.adapters.langchain")
