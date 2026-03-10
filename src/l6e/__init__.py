"""l6e — pipeline-scoped model choice enforcement for AI agents.

Primary entry points:

    Universal (raw LiteLLM / OpenAI SDK):
        import l6e
        with l6e.pipeline("run-id", policy) as ctx:
            response = ctx.call(fn, model="gpt-4o", messages=[...], stage="summarization")

    LangChain:
        from l6e.adapters.langchain import L6eCallbackHandler

    CrewAI:
        from l6e.adapters.crewai import L6eStepCallback
"""
from l6e._types import (
    BudgetMode,
    BudgetStatus,
    OnBudgetExceeded,
    PipelinePolicy,
    RunSummary,
    StageRoutingHint,
)
from l6e.exceptions import BudgetExceeded, LatencySLAExceeded
from l6e.pipeline import PipelineContext, pipeline

__all__ = [
    "pipeline",
    "PipelineContext",
    "BudgetExceeded",
    "LatencySLAExceeded",
    "RunSummary",
    "BudgetMode",
    "BudgetStatus",
    "PipelinePolicy",
    "OnBudgetExceeded",
    "StageRoutingHint",
]
__version__ = "0.1.0"
