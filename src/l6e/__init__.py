"""l6e — pipeline-scoped model choice enforcement for AI agents.

Primary entry points:

    Universal (raw LiteLLM / OpenAI SDK):
        import l6e
        with l6e.pipeline(policy) as ctx:
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
    PromptComplexity,
    RunSummary,
    StageRoutingHint,
    UnknownModelPricingMode,
)
from l6e.exceptions import BudgetExceeded
from l6e.exceptions import LatencySLAExceeded as LatencySLAExceeded
from l6e.pipeline import PipelineContext, pipeline

__all__ = [
    "pipeline",
    "PipelineContext",
    "BudgetExceeded",
    "RunSummary",
    "BudgetMode",
    "BudgetStatus",
    "PipelinePolicy",
    "OnBudgetExceeded",
    "PromptComplexity",
    "StageRoutingHint",
    "UnknownModelPricingMode",
]
__version__ = "0.4.1"
