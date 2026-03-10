"""CrewAI adapter for l6e — L6eStepCallback.

No crewai import required. L6eStepCallback is a plain callable that CrewAI
invokes after each agent step with a Union[AgentAction, AgentFinish] object
(from langchain_core.agents). We do not inspect the step output — enforcement
is based purely on the gate decision from PipelineContext.

Usage::

    from l6e.adapters.crewai import L6eStepCallback

    crew = Crew(
        agents=agents,
        tasks=tasks,
        step_callback=L6eStepCallback(ctx, stage="agent_step"),
    )
"""
from __future__ import annotations

from l6e.exceptions import BudgetExceeded
from l6e.pipeline import PipelineContext


class L6eStepCallback:
    """Enforces l6e budget constraints between CrewAI agent steps.

    Calls ``ctx.advise()`` on each step. If the gate returns ``halt``,
    raises ``BudgetExceeded`` to stop the crew. Allow and reroute decisions
    are advisory in v0.1 — the step proceeds in both cases.
    """

    def __init__(self, ctx: PipelineContext, stage: str | None = None) -> None:
        self._ctx = ctx
        self._stage = stage

    def __call__(self, step_output: object) -> None:
        decision = self._ctx.advise(
            model="unknown",
            prompts=[""],
            stage=self._stage,
        )
        if decision.action == "halt":
            status = self._ctx.budget_status()
            raise BudgetExceeded(
                spent=status.spent_usd,
                budget=status.budget_usd,
                reason=decision.reason,
            )
