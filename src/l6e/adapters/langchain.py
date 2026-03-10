"""LangChain adapter for l6e — L6eCallbackHandler.

Install the langchain extra to use this module:
    pip install 'l6e[langchain]'
"""
from __future__ import annotations

import time
from typing import Any, Literal
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:
    raise ImportError(
        "langchain-core is required for L6eCallbackHandler. "
        "Install it with: pip install 'l6e[langchain]'"
    ) from exc

from l6e._classify import PromptComplexityClassifier
from l6e._types import GateDecision, PromptComplexity
from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext

# Pending call state stored between on_llm_start and on_llm_end.
# (stage, stage_source, model, decision, t0)
_StageSource = Literal["declared", "inferred"]
_PendingEntry = tuple[str | None, _StageSource | None, str, GateDecision, float]

# Maps classifier complexity → a canonical stage name used when inferring.
_COMPLEXITY_TO_STAGE: dict[PromptComplexity, str] = {
    PromptComplexity.LOW: "retrieval",
    PromptComplexity.MEDIUM: "formatting",
    PromptComplexity.HIGH: "reasoning",
}


def _extract_model(serialized: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """Extract the actual model string from LangChain callback args.

    invocation_params["model"] / ["model_name"] holds the real model string
    (e.g. "gpt-4o"). serialized["id"][-1] is the class name ("ChatOpenAI").
    """
    ip = kwargs.get("invocation_params") or {}
    model = ip.get("model") or ip.get("model_name")
    if isinstance(model, str) and model:
        return model
    fallback = serialized.get("name")
    if isinstance(fallback, str) and fallback:
        return fallback
    return "unknown"


def _extract_stage(tags: list[str] | None) -> str | None:
    return next(
        (t.split(":", 1)[1] for t in (tags or []) if t.startswith("l6e_stage:")),
        None,
    )


class L6eCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that enforces l6e pipeline budget constraints.

    Attach to any LangChain chain or LLM::

        handler = L6eCallbackHandler(ctx)
        chain.invoke(input, config={"callbacks": [handler]})

    **Stage declaration (explicit)** — tag the chain at configuration time::

        chain.with_config(tags=["l6e_stage:summarization"]).invoke(...)

    **Stage inference (automatic, default)** — omit the tag entirely::

        chain.invoke(input, config={"callbacks": [handler]})

    When no ``l6e_stage:`` tag is present and ``infer_stage=True`` (the default),
    the classifier runs on the first prompt and maps complexity → stage:
    LOW → retrieval, MEDIUM → formatting, HIGH → reasoning.
    ``CallRecord.stage`` is set to the inferred name and
    ``CallRecord.prompt_complexity`` reflects the classifier output.

    Set ``infer_stage=False`` to keep the old behaviour (``stage=None`` when
    no tag is present).
    """

    def __init__(self, ctx: PipelineContext, *, infer_stage: bool = True) -> None:
        super().__init__()
        self._ctx = ctx
        self._infer_stage = infer_stage
        self._classifier = PromptComplexityClassifier()
        self._pending: dict[UUID, _PendingEntry] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        declared_stage = _extract_stage(tags)
        model = _extract_model(serialized, kwargs)
        t0 = time.perf_counter()

        stage: str | None
        stage_source: _StageSource | None

        if declared_stage is not None:
            stage = declared_stage
            stage_source = "declared"
        elif self._infer_stage and prompts:
            complexity = self._classifier.classify(prompts[0], stage=None)
            stage = _COMPLEXITY_TO_STAGE[complexity]
            stage_source = "inferred"
        else:
            stage = None
            stage_source = None

        decision = self._ctx.advise(model=model, prompts=prompts, stage=stage)

        if decision.action == "halt":
            status = self._ctx.budget_status()
            raise BudgetExceeded(
                spent=status.spent_usd,
                budget=status.budget_usd,
                reason=decision.reason,
            )

        self._pending[run_id] = (stage, stage_source, model, decision, t0)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        entry = self._pending.pop(run_id, None)
        if entry is None:
            return

        stage, stage_source, model_requested, decision, t0 = entry
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rerouted = decision.action == "reroute"

        self._ctx.record(
            model_requested=model_requested,
            model_used=decision.target_model,
            response=response,
            elapsed_ms=elapsed_ms,
            stage=stage,
            rerouted=rerouted,
        )
