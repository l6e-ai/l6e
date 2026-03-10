"""ConstraintGate — enforcement decision core for l6e.

Decision only: no LLM calls, no execution, no side effects.
Returns a GateDecision(action, target_model, reason) for each pending call.
"""
from __future__ import annotations

from l6e._protocols import ILocalRouter, IRunStore
from l6e._types import BudgetMode, GateDecision, PipelinePolicy, PromptComplexity, StageRoutingHint


def _allow(model: str, reason: str = "allow") -> GateDecision:
    return GateDecision(action="allow", target_model=model, reason=reason)


def _halt(model: str, reason: str) -> GateDecision:
    return GateDecision(action="halt", target_model=model, reason=reason)


def _reroute(target: str, reason: str) -> GateDecision:
    return GateDecision(action="reroute", target_model=target, reason=reason)


class ConstraintGate:
    """Reads PipelinePolicy + IRunStore state + ILocalRouter and returns a GateDecision.

    Priority order:
    1. stage_overrides  — explicit BudgetMode per stage, wins over everything
    2. stage_routing    — tier hint per stage
    3. over-budget      — estimated_cost would push spend past budget
    4. budget pressure  — spent/budget >= reroute_threshold
    5. allow            — budget healthy, no stage constraint
    """

    def __init__(self, policy: PipelinePolicy, router: ILocalRouter) -> None:
        self._policy = policy
        self._router = router

    def check(
        self,
        store: IRunStore,
        model: str,
        estimated_cost: float,
        stage: str | None,
        complexity: PromptComplexity | None,
    ) -> GateDecision:
        policy = self._policy

        # ------------------------------------------------------------------
        # 1. stage_overrides — explicit BudgetMode per stage
        # ------------------------------------------------------------------
        if stage is not None and stage in policy.stage_overrides:
            override = policy.stage_overrides[stage]
            if override == BudgetMode.HALT:
                return _halt(model, "stage_override:halt")
            if override == BudgetMode.REROUTE:
                return self._do_reroute(model, "stage_override:reroute")
            # WARN is informational only at v0.1 — fall through to allow
            return _allow(model, "stage_override:warn")

        # ------------------------------------------------------------------
        # 2. stage_routing — tier hint per stage
        # ------------------------------------------------------------------
        if stage is not None and stage in policy.stage_routing:
            hint = policy.stage_routing[stage]
            if hint == StageRoutingHint.LOCAL:
                return self._do_reroute(model, "stage_routing:local")
            if hint == StageRoutingHint.CLOUD_STANDARD:
                return _allow(model, "stage_routing:cloud_standard")
            if hint == StageRoutingHint.CLOUD_FRONTIER:
                return _allow(model, "allow:frontier_protected")
            # INHERIT → fall through to budget pressure

        # ------------------------------------------------------------------
        # 3. Over-budget guard — estimated call would exceed total budget
        # ------------------------------------------------------------------
        if store.spent() + estimated_cost > policy.budget:
            return _halt(model, "budget_pressure:halt")

        # ------------------------------------------------------------------
        # 4. Budget pressure — spent/budget >= reroute_threshold
        # ------------------------------------------------------------------
        if policy.budget > 0 and store.spent() / policy.budget >= policy.reroute_threshold:
            mode = policy.budget_mode
            if mode == BudgetMode.HALT:
                return _halt(model, "budget_pressure:halt")
            if mode == BudgetMode.REROUTE:
                return self._do_reroute(model, "budget_pressure:reroute")
            # WARN
            return _allow(model, "warn:budget_pressure")

        # ------------------------------------------------------------------
        # 5. Default: allow
        # ------------------------------------------------------------------
        return _allow(model)

    def _do_reroute(self, requested_model: str, reason: str) -> GateDecision:
        """Attempt reroute to local; fall back to halt if no local model."""
        local = self._router.best_local_model()
        if local is None:
            return _halt(requested_model, reason + ":no_local_model")
        return _reroute(local, reason)
