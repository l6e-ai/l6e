"""PipelineContext — the central object for a pipeline run.

Wires gate, store, estimator, classifier, and log together.
No LLM calls live here — adapters execute, PipelineContext advises and records.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Literal

from l6e._classify import PromptComplexityClassifier
from l6e._log import LocalRunLog
from l6e._protocols import IConstraintGate, ICostEstimator, ILocalRouter, IRunStore
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

    Thread-safety: ``record()`` and ``call()`` are safe to call from multiple
    threads — the call_index counter is incremented under a lock.
    Note: ``run_summary()`` and ``budget_status()`` reads are not protected
    by this lock.  Do not share a ``PipelineContext`` across threads while
    writes are still in flight without external coordination.
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
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        """The unique identifier for this pipeline run."""
        return self._run_id

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
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        cohort_hint: str | None = None,
    ) -> GateDecision:
        """Gate-check a pending call. Does not execute the call.

        ``user_id`` / ``tenant_id`` / ``cohort_hint`` are Margin-tier identity
        hints forwarded to the gate. Pure local gates ignore them.
        """
        complexity = self._classifier.classify(prompts[0] if prompts else "", stage)
        prompt_tokens = _estimate_prompt_tokens(prompts)
        estimated_cost = self._estimator.estimate(model, prompt_tokens, 0)
        return self._gate.check(
            self._store,
            model=model,
            estimated_cost=estimated_cost,
            stage=stage,
            complexity=complexity,
            user_id=user_id,
            tenant_id=tenant_id,
            cohort_hint=cohort_hint,
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
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        cohort_hint: str | None = None,
    ) -> CallRecord:
        """Record a completed call. Extracts token usage, estimates cost, appends to store.

        ``user_id`` / ``tenant_id`` / ``cohort_hint`` are persisted on the
        ``CallRecord`` for downstream telemetry (RunSummary, SaaS profiler).
        """
        prompt_tokens, completion_tokens = extract_token_usage(response)
        cost = self._estimator.estimate(model_used, prompt_tokens, completion_tokens)
        with self._lock:
            call_index = self._call_index
            self._call_index += 1
        record = CallRecord(
            call_index=call_index,
            model_requested=model_requested,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            rerouted=rerouted,
            elapsed_ms=elapsed_ms,
            stage=stage,
            prompt_complexity=complexity,
            user_id=user_id,
            tenant_id=tenant_id,
            cohort_hint=cohort_hint,
        )
        self._store.record_call(record)
        return record

    def run_summary(self) -> RunSummary:
        """Full record of every call made in this pipeline run."""
        return self._store.to_summary()

    def budget_status(self) -> BudgetStatus:
        """Zero-token snapshot of current pipeline economics."""
        spent = self._store.spent()
        budget = Decimal(str(self._policy.budget))
        pct = float(spent / budget * 100) if self._policy.budget > 0 else 0.0
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
        fn: Callable[..., object],
        model: str,
        messages: list[dict[str, str]],
        stage: str | None = None,
        complexity: PromptComplexity | None = None,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        cohort_hint: str | None = None,
    ) -> object:
        """Advise on model, execute fn, record response.

        Runs the full gate-advise → execute → record cycle in one call.
        This is the primary entry point for adapters that want automatic
        budget enforcement without manually calling ``advise`` and ``record``.

        Gate actions:

        - **allow** — calls ``fn(model=..., messages=...)`` unchanged.
        - **reroute** — calls ``fn(model=decision.target_model, messages=...)``; the
          ``CallRecord`` marks ``rerouted=True`` so savings are tracked.
        - **halt** — does *not* call ``fn``; behaviour is determined by
          ``policy.on_budget_exceeded`` (raise, return fallback, return empty).

        Args:
            fn: Called as ``fn(model=..., messages=...)`` with keyword arguments.
                Must return a response object understood by ``extract_token_usage``.
            model: The model the caller *wants* to use.  The gate may
                substitute a cheaper local model on reroute.
            messages: OpenAI-style chat messages list.  User-role content is
                extracted to estimate prompt tokens for the gate decision.
            stage: Optional pipeline stage label (e.g. ``"draft"``,
                ``"review"``).  Passed through to the gate and ``CallRecord``.
            complexity: Pre-computed prompt complexity.  When ``None`` the
                classifier derives it from the first user message.
            user_id: Optional Margin-tier end-user identity. Forwarded to the
                gate and persisted on ``CallRecord`` for downstream
                telemetry. Ignored by the OSS ``ConstraintGate``.
            tenant_id: Optional Margin-tier tenant / organisation identity.
                Same semantics as ``user_id``.
            cohort_hint: Optional free-form cohort label (e.g. ``"enterprise"``,
                ``"trial"``) used by cloud-sync gates to select calibrated
                cost factors. Persisted on ``CallRecord``.

        Returns:
            The raw response object returned by ``fn``, or the policy's
            fallback value when the gate halts and ``on_budget_exceeded``
            is not ``RAISE``.

        Raises:
            BudgetExceeded: If the gate halts and
                ``policy.on_budget_exceeded == OnBudgetExceeded.RAISE``.
        """
        prompts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not prompts:
            prompts = [""]

        decision = self.advise(
            model=model,
            prompts=prompts,
            stage=stage,
            user_id=user_id,
            tenant_id=tenant_id,
            cohort_hint=cohort_hint,
        )

        if decision.action == "halt":
            return self._handle_halt(decision)

        rerouted = decision.action == "reroute"
        effective_model = decision.target_model

        t0 = time.perf_counter()
        response = fn(model=effective_model, messages=messages)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.record(
            model_requested=model,
            model_used=effective_model,
            response=response,
            elapsed_ms=elapsed_ms,
            stage=stage,
            complexity=complexity,
            rerouted=rerouted,
            user_id=user_id,
            tenant_id=tenant_id,
            cohort_hint=cohort_hint,
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
                budget=Decimal(str(self._policy.budget)),
                reason=decision.reason,
            )
        if mode == OnBudgetExceeded.RETURN_FALLBACK:
            return self._policy.fallback_result or ""
        # RETURN_PARTIAL and RETURN_EMPTY both return empty string pre-call
        return ""


def pipeline(
    policy: PipelinePolicy,
    run_id: str | None = None,
    log_path: Path | None = None,
    router: ILocalRouter | None = None,
    source: str = "pipeline",
) -> PipelineContext:
    """Construct a fully-wired PipelineContext with default concrete collaborators.

    Args:
        policy:   Budget and routing policy.
        run_id:   Unique identifier for this pipeline run.  When omitted a
                  UUID v4 is generated automatically.
        log_path: Override the default `.l6e/runs.jsonl` log location.
        router:   Custom router implementing ``best_local_model() -> str | None``.
                  Defaults to ``LocalRouter`` (hardware-aware Ollama detection).
                  Pass a test double here to control routing in notebooks or tests
                  without touching any other part of the pipeline.
        source:   Origin of the run — ``"pipeline"`` for OSS runs, ``"mcp"`` for
                  MCP session runs. Written to ``RunSummary.source`` in the log.
    """
    from l6e.costs import LiteLLMCostEstimator
    from l6e.gate import ConstraintGate
    from l6e.router import LocalRouter
    from l6e.store import InMemoryRunStore

    effective_run_id = run_id if run_id is not None else str(uuid.uuid4())
    estimator = LiteLLMCostEstimator(
        fallback_cost_per_1k_tokens=policy.unknown_model_cost_per_1k_tokens
    )
    effective_router = router if router is not None else LocalRouter()
    gate = ConstraintGate(policy=policy, router=effective_router)
    store = InMemoryRunStore(
        run_id=effective_run_id, policy=policy, estimator=estimator, source=source
    )
    log = LocalRunLog(path=log_path) if log_path is not None else LocalRunLog()
    classifier = PromptComplexityClassifier()

    return PipelineContext(
        run_id=effective_run_id,
        policy=policy,
        gate=gate,
        store=store,
        log=log,
        classifier=classifier,
        estimator=estimator,
    )
