"""PipelineContext — the central object for a pipeline run.

Wires gate, store, estimator, classifier, and log together.
No LLM calls live here — adapters execute, PipelineContext advises and records.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import Literal

from l6e._classify import PromptComplexityClassifier
from l6e._log import LocalRunLog
from l6e._protocols import IConstraintGate, ICostEstimator, IRunStore
from l6e._response import extract_token_usage
from l6e._types import (
    BudgetStatus,
    CallRecord,
    GateDecision,
    OnBudgetExceeded,
    PipelinePolicy,
    PromptComplexity,
    RunSummary,
)
from l6e.exceptions import BudgetExceeded


def _estimate_prompt_tokens(prompts: list[str]) -> int:
    """Best-effort prompt token estimate via tiktoken, fallback to char/4."""
    text = "\n".join(prompts)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _pressure(pct: float) -> Literal["low", "moderate", "high", "critical"]:
    if pct < 50.0:
        return "low"
    if pct < 80.0:
        return "moderate"
    if pct < 95.0:
        return "high"
    return "critical"


class PipelineContext:
    """Runs one budget-scoped pipeline.

    Created directly for injection-heavy tests, or via the ``pipeline()``
    factory for production use with default concrete collaborators.
    """

    def __init__(
        self,
        run_id: str,
        policy: PipelinePolicy,
        gate: IConstraintGate,
        store: IRunStore,
        log: LocalRunLog,
        classifier: PromptComplexityClassifier,
        estimator: ICostEstimator,
    ) -> None:
        self._run_id = run_id
        self._policy = policy
        self._gate = gate
        self._store = store
        self._log = log
        self._classifier = classifier
        self._estimator = estimator
        self._call_index: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> PipelineContext:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Always write the run log, even on exception. Never suppresses."""
        self._log.append(self._store.to_summary())
        return False

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def advise(
        self,
        model: str,
        prompts: list[str],
        stage: str | None = None,
    ) -> GateDecision:
        """Gate-check a pending call. Does not execute the call."""
        complexity = self._classifier.classify(prompts[0] if prompts else "", stage)
        prompt_tokens = _estimate_prompt_tokens(prompts)
        estimated_cost = self._estimator.estimate(model, prompt_tokens, 0)
        return self._gate.check(
            self._store,
            model=model,
            estimated_cost=estimated_cost,
            stage=stage,
            complexity=complexity,
        )

    def record(
        self,
        model_requested: str,
        model_used: str,
        response: object,
        elapsed_ms: float,
        stage: str | None = None,
        complexity: PromptComplexity | None = None,
        rerouted: bool = False,
    ) -> CallRecord:
        """Record a completed call. Extracts token usage, estimates cost, appends to store."""
        prompt_tokens, completion_tokens = extract_token_usage(response)
        cost = self._estimator.estimate(model_used, prompt_tokens, completion_tokens)
        record = CallRecord(
            call_index=self._call_index,
            model_requested=model_requested,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            rerouted=rerouted,
            elapsed_ms=elapsed_ms,
            stage=stage,
            prompt_complexity=complexity,
        )
        self._store.record_call(record)
        self._call_index += 1
        return record

    def run_summary(self) -> RunSummary:
        """Full record of every call made in this pipeline run."""
        return self._store.to_summary()

    def budget_status(self) -> BudgetStatus:
        """Zero-token snapshot of current pipeline economics."""
        spent = self._store.spent()
        budget = self._policy.budget
        pct = (spent / budget * 100.0) if budget > 0 else 0.0
        summary = self._store.to_summary()
        return BudgetStatus(
            run_id=self._store.run_id,
            spent_usd=spent,
            remaining_usd=self._store.remaining(),
            budget_usd=budget,
            calls_made=self._store.call_count(),
            reroutes=summary.reroutes,
            pct_used=pct,
            budget_pressure=_pressure(pct),
        )

    def call(
        self,
        fn: Callable[[str, list[dict[str, str]]], object],
        model: str,
        messages: list[dict[str, str]],
        stage: str | None = None,
        complexity: PromptComplexity | None = None,
    ) -> object:
        """Advise on model, execute fn, record response.

        allow   → fn(model, messages)
        reroute → fn(decision.target_model, messages), rerouted=True in record
        halt    → behaviour determined by policy.on_budget_exceeded
        """
        prompts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not prompts:
            prompts = [""]

        decision = self.advise(model=model, prompts=prompts, stage=stage)

        if decision.action == "halt":
            return self._handle_halt(decision)

        rerouted = decision.action == "reroute"
        effective_model = decision.target_model

        t0 = time.perf_counter()
        response = fn(effective_model, messages)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.record(
            model_requested=model,
            model_used=effective_model,
            response=response,
            elapsed_ms=elapsed_ms,
            stage=stage,
            complexity=complexity,
            rerouted=rerouted,
        )
        return response

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _handle_halt(self, decision: GateDecision) -> object:
        mode = self._policy.on_budget_exceeded
        if mode == OnBudgetExceeded.RAISE:
            raise BudgetExceeded(
                spent=self._store.spent(),
                budget=self._policy.budget,
                reason=decision.reason,
            )
        if mode == OnBudgetExceeded.RETURN_FALLBACK:
            return self._policy.fallback_result or ""
        # RETURN_PARTIAL and RETURN_EMPTY both return empty string pre-call
        return ""


def pipeline(
    run_id: str,
    policy: PipelinePolicy,
    log_path: Path | None = None,
    router: object | None = None,
) -> PipelineContext:
    """Construct a fully-wired PipelineContext with default concrete collaborators.

    Args:
        run_id:   Unique identifier for this pipeline run.
        policy:   Budget and routing policy.
        log_path: Override the default `.l6e/runs.jsonl` log location.
        router:   Custom router implementing ``best_local_model() -> str | None``.
                  Defaults to ``LocalRouter`` (hardware-aware Ollama detection).
                  Pass a test double here to control routing in notebooks or tests
                  without touching any other part of the pipeline.
    """
    from l6e.costs import LiteLLMCostEstimator
    from l6e.gate import ConstraintGate
    from l6e.router import LocalRouter
    from l6e.store import InMemoryRunStore

    estimator = LiteLLMCostEstimator()
    effective_router = router if router is not None else LocalRouter()
    gate = ConstraintGate(policy=policy, router=effective_router)
    store = InMemoryRunStore(run_id=run_id, policy=policy, estimator=estimator)
    log = LocalRunLog(path=log_path) if log_path is not None else LocalRunLog()
    classifier = PromptComplexityClassifier()

    return PipelineContext(
        run_id=run_id,
        policy=policy,
        gate=gate,
        store=store,
        log=log,
        classifier=classifier,
        estimator=estimator,
    )
