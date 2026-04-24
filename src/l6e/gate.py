"""ConstraintGate — enforcement decision core for l6e.

Decision only: no LLM calls, no execution, no side effects.
Returns a GateDecision(action, target_model, reason) for each pending call.

The pure decision logic lives in ``l6e._gate_core`` so cloud
(``hosted-edge``/``/v1/authorize``) can enforce identical semantics.
This class layers local-router resolution on top: when the core asks
for a reroute, we consult ``ILocalRouter.best_local_model()`` and
halt cleanly when no local model is available.
"""
from __future__ import annotations

from decimal import Decimal

from l6e._gate_core import GateCoreOutcome, decide
from l6e._protocols import ILocalRouter, IRunStore
from l6e._types import GateDecision, PipelinePolicy, PromptComplexity


def _allow(model: str, reason: str = "allow") -> GateDecision:
    return GateDecision(action="allow", target_model=model, reason=reason)


def _halt(model: str, reason: str) -> GateDecision:
    return GateDecision(action="halt", target_model=model, reason=reason)


def _reroute(target: str, reason: str) -> GateDecision:
    return GateDecision(action="reroute", target_model=target, reason=reason)


class ConstraintGate:
    """Reads PipelinePolicy + IRunStore state + ILocalRouter and returns a GateDecision.

    Priority order (delegated to ``l6e._gate_core.decide``):
    1. stage_overrides  — explicit BudgetMode per stage, wins over everything
    2. over-budget      — estimated_cost would push spend past budget
    3. stage_routing    — tier hint per stage
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
        estimated_cost: Decimal,
        stage: str | None,
        complexity: PromptComplexity | None,
        *,
        user_id: str | None = None,
        tenant_id: str | None = None,
        cohort_hint: str | None = None,
    ) -> GateDecision:
        # Identity kwargs are accepted for protocol compatibility and future
        # cohort-aware routing; the OSS ConstraintGate does not act on them.
        del user_id, tenant_id, cohort_hint
        policy = self._policy

        outcome = decide(
            budget=Decimal(str(policy.budget)),
            spent=store.spent(),
            estimated_cost=estimated_cost,
            budget_mode=policy.budget_mode,
            reroute_threshold=Decimal(str(policy.reroute_threshold)),
            stage=stage,
            complexity=complexity,
            stage_overrides=policy.stage_overrides,
            stage_routing=policy.stage_routing,
        )

        return self._materialize(model, outcome)

    def _materialize(self, model: str, outcome: GateCoreOutcome) -> GateDecision:
        """Resolve a pure outcome into a concrete ``GateDecision``.

        Every reroute in the OSS gate needs a local-model target.
        ``wants_local_reroute`` captures the cases (stage override
        REROUTE, stage_routing LOCAL, budget-pressure REROUTE). When
        the router has nothing to offer we halt cleanly with a stable
        ``:no_local_model`` suffix — the integration tests match on it.
        """
        if outcome.action == "allow":
            return _allow(model, outcome.reason)
        if outcome.action == "halt":
            return _halt(model, outcome.reason)
        if outcome.wants_local_reroute:
            local = self._router.best_local_model()
            if local is None:
                return _halt(model, outcome.reason + ":no_local_model")
            return _reroute(local, outcome.reason)
        # Defensive: a reroute that doesn't want local resolution has
        # no meaningful target in the OSS gate. Halt rather than pick
        # an arbitrary model.
        return _halt(model, outcome.reason + ":no_local_model")
