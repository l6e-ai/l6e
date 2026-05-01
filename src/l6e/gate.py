"""ConstraintGate — enforcement decision core for l6e.

Decision only: no LLM calls, no execution, no side effects.
Returns a GateDecision(action, target_model, reason) for each pending call.

The pure decision logic lives in ``l6e._gate_core`` so the cloud
authorize endpoint can enforce identical semantics. This class layers
local-router resolution on top: when the core asks for a reroute, we
consult ``ILocalRouter.best_local_model()`` and halt cleanly when no
local model is available.

``RemoteConstraintGate`` (sibling, opt-in) extends the local gate with a
synchronous ``POST /v1/authorize`` call that activates the SDK
cloud-sync integration tier. It always wraps an inner local gate and
falls open to the local decision on every failure path — the gate fails
open, always.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from l6e._gate_core import GateCoreOutcome, decide
from l6e._protocols import ILocalRouter, IRunStore
from l6e._types import GateDecision, PipelinePolicy, PromptComplexity
from l6e.cloud import CloudConfig, _post_authorize

logger = logging.getLogger(__name__)

# Stable, machine-greppable reason prefixes published on fail-open
# decisions so operators can pivot dashboards and alerts on them. The
# ``fail_open:cloud_*`` family mirrors the convention pipeline.py
# already uses for in-process exceptions (``fail_open:gate_exception``).
_FAIL_OPEN_CLOUD_NETWORK = "fail_open:cloud_network"
_FAIL_OPEN_CLOUD_BUILD_BODY = "fail_open:cloud_build_body"
_FAIL_OPEN_CLOUD_MAP = "fail_open:cloud_response_map"


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


# ---------------------------------------------------------------------------
# RemoteConstraintGate — opt-in SDK cloud-sync.
# ---------------------------------------------------------------------------


def _decimal_or_none(value: Any) -> Decimal | None:
    """Coerce a server-supplied numeric to ``Decimal`` or ``None``.

    The sanitizer in ``l6e.cloud`` has already rejected NaN / inf /
    negative / non-numeric, so this is a thin str-roundtrip to preserve
    Decimal precision across the JSON parser.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _apply_cloud_response(
    *,
    response: dict[str, Any],
    model: str,
    local_router: ILocalRouter,
) -> GateDecision:
    """Map a sanitized ``/v1/authorize`` response into a ``GateDecision``.

    Pure function — no I/O, no side effects. Kept extracted so a future
    fire-and-forget allow-path mode can reuse the mapping without
    re-running the HTTP call.

    Reroute target resolution: prefer the server's
    ``routed_model_suggestion`` when present. Otherwise consult the
    local router so cloud-sync still works for callers on a
    metadata-tier policy. If neither yields a target, halt cleanly
    rather than allow with an arbitrary model — same fail-safe stance
    as ``ConstraintGate._materialize``.
    """
    action = response["action"]  # sanitizer guarantees membership
    base_reason = response.get("gate_reason") or f"server_calibrated:{action}"

    calibration_source = response.get("calibration_source")
    calibration_factor = _decimal_or_none(response.get("calibration_factor"))
    predicted_mean = _decimal_or_none(response.get("predicted_cost_mean_usd"))
    predicted_p95 = _decimal_or_none(response.get("predicted_cost_p95_usd"))
    policy_id = response.get("policy_id_applied")

    if action == "halt":
        return GateDecision(
            action="halt",
            target_model=model,
            reason=base_reason,
            calibration_source=calibration_source,
            calibration_factor=calibration_factor,
            predicted_cost_mean_usd=predicted_mean,
            predicted_cost_p95_usd=predicted_p95,
            policy_id_applied=policy_id,
        )

    if action == "reroute":
        target = response.get("routed_model_suggestion")
        if not target:
            target = local_router.best_local_model()
        if not target:
            return GateDecision(
                action="halt",
                target_model=model,
                reason=base_reason + ":no_reroute_target",
                calibration_source=calibration_source,
                calibration_factor=calibration_factor,
                predicted_cost_mean_usd=predicted_mean,
                predicted_cost_p95_usd=predicted_p95,
                policy_id_applied=policy_id,
            )
        return GateDecision(
            action="reroute",
            target_model=target,
            reason=base_reason,
            calibration_source=calibration_source,
            calibration_factor=calibration_factor,
            predicted_cost_mean_usd=predicted_mean,
            predicted_cost_p95_usd=predicted_p95,
            policy_id_applied=policy_id,
        )

    # action == "allow"
    return GateDecision(
        action="allow",
        target_model=model,
        reason=base_reason,
        calibration_source=calibration_source,
        calibration_factor=calibration_factor,
        predicted_cost_mean_usd=predicted_mean,
        predicted_cost_p95_usd=predicted_p95,
        policy_id_applied=policy_id,
    )


def _decorate_local_fallback(
    decision: GateDecision, *, fallback_reason: str,
) -> GateDecision:
    """Stamp ``calibration_source='local_fallback'`` and a fail-open reason
    onto a local-gate decision so downstream telemetry can distinguish
    "cloud said allow" from "cloud was unreachable, local said allow"."""
    return GateDecision(
        action=decision.action,
        target_model=decision.target_model,
        reason=fallback_reason,
        calibration_source="local_fallback",
        calibration_factor=None,
        predicted_cost_mean_usd=None,
        predicted_cost_p95_usd=None,
        policy_id_applied=None,
    )


class RemoteConstraintGate(ConstraintGate):
    """Cloud-sync gate. Wraps a local ``ConstraintGate`` and POSTs to
    ``{cloud.base_url}/v1/authorize`` on every ``check()``.

    Failure handling — the gate fails open, always:

    - any HTTP failure (timeout, network, non-200, malformed JSON,
      garbage envelope) → delegate to ``super().check(...)`` and stamp
      ``calibration_source='local_fallback'`` plus a stable
      ``fail_open:cloud_*`` reason on the returned ``GateDecision``.
    - any unexpected exception in body construction or response mapping
      → same: fall through to local. Never raises into customer code.

    Identity plumbing: ``user_id`` / ``tenant_id`` / ``cohort_hint`` (the
    optional kwargs already accepted by ``ctx.call`` / ``ctx.advise`` /
    ``ctx.record``) are forwarded into the request body verbatim. The
    presence of any one of them flips the server into its identity-aware
    response mode.

    Latency posture: synchronous block. The cloud HTTP call honors
    ``cfg.latency_deadline_ms`` as a hard local timeout; on exceed the
    request is treated as down and we fail open. A future fire-and-forget
    allow-path mode would reuse this class's helpers without changing
    the public contract.
    """

    def __init__(
        self,
        policy: PipelinePolicy,
        router: ILocalRouter,
        cloud: CloudConfig,
    ) -> None:
        super().__init__(policy=policy, router=router)
        self._cloud = cloud

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
        # Iron-rule outer try: nothing in this method may surface an
        # exception to ``PipelineContext.advise``. Any unhandled error
        # collapses to the local gate.
        try:
            return self._check_with_cloud(
                store=store,
                model=model,
                estimated_cost=estimated_cost,
                stage=stage,
                complexity=complexity,
                user_id=user_id,
                tenant_id=tenant_id,
                cohort_hint=cohort_hint,
            )
        except Exception:
            logger.warning("remote_gate_unhandled_exception", exc_info=True)
            local = super().check(
                store,
                model=model,
                estimated_cost=estimated_cost,
                stage=stage,
                complexity=complexity,
                user_id=user_id,
                tenant_id=tenant_id,
                cohort_hint=cohort_hint,
            )
            return _decorate_local_fallback(
                local, fallback_reason=_FAIL_OPEN_CLOUD_NETWORK,
            )

    def _check_with_cloud(
        self,
        *,
        store: IRunStore,
        model: str,
        estimated_cost: Decimal,
        stage: str | None,
        complexity: PromptComplexity | None,
        user_id: str | None,
        tenant_id: str | None,
        cohort_hint: str | None,
    ) -> GateDecision:
        # Pre-compute local fallback once; every failure path returns it.
        # Building it here also catches any local-gate misconfiguration
        # before we issue the network call (cheaper failure mode).
        local_decision = super().check(
            store,
            model=model,
            estimated_cost=estimated_cost,
            stage=stage,
            complexity=complexity,
            user_id=user_id,
            tenant_id=tenant_id,
            cohort_hint=cohort_hint,
        )

        try:
            body = self._build_body(
                store=store,
                model=model,
                estimated_cost=estimated_cost,
                stage=stage,
                complexity=complexity,
                user_id=user_id,
                tenant_id=tenant_id,
                cohort_hint=cohort_hint,
            )
        except Exception:
            logger.warning("remote_gate_build_body_failed", exc_info=True)
            return _decorate_local_fallback(
                local_decision, fallback_reason=_FAIL_OPEN_CLOUD_BUILD_BODY,
            )

        response = _post_authorize(self._cloud, body)
        if response is None:
            # ``_post_authorize`` already logged the specific failure
            # mode at WARNING. We don't have enough information here to
            # distinguish timeout vs 5xx vs bad-json without re-doing
            # the call, so the fail-open reason is the generic network
            # tag. Operators looking at why a fallback happened pivot
            # on the ``cloud_authorize_*`` log entries, not this string.
            return _decorate_local_fallback(
                local_decision, fallback_reason=_FAIL_OPEN_CLOUD_NETWORK,
            )

        try:
            return _apply_cloud_response(
                response=response, model=model, local_router=self._router,
            )
        except Exception:
            logger.warning("remote_gate_response_map_failed", exc_info=True)
            return _decorate_local_fallback(
                local_decision, fallback_reason=_FAIL_OPEN_CLOUD_MAP,
            )

    def _build_body(
        self,
        *,
        store: IRunStore,
        model: str,
        estimated_cost: Decimal,
        stage: str | None,
        complexity: PromptComplexity | None,
        user_id: str | None,
        tenant_id: str | None,
        cohort_hint: str | None,
    ) -> dict[str, Any]:
        """Assemble the ``/v1/authorize`` request body.

        Identity fields (``user_id`` / ``tenant_id`` / ``cohort_hint``)
        are forwarded only when supplied, so callers without identity
        plumbing keep landing on the server's non-identity response
        shape. The schema is additive: unknown fields are ignored on
        older server versions, preserving deployment ordering freedom.
        """
        budget = Decimal(str(self._policy.budget))
        body: dict[str, Any] = {
            "session_id": store.run_id,
            "model": model,
            "tool_name": stage if stage is not None else "sdk_call",
            "estimated_cost_usd": float(estimated_cost),
            "budget_usd": float(budget),
            "spent_usd": float(store.spent()),
            "client": "l6e_sdk",
            "latency_deadline_ms": self._cloud.latency_deadline_ms,
        }
        if stage is not None:
            body["stage"] = stage
        if complexity is not None:
            body["prompt_complexity"] = complexity.value
        if user_id is not None:
            body["user_id"] = user_id
        if tenant_id is not None:
            body["tenant_id"] = tenant_id
        if cohort_hint is not None:
            body["cohort_hint"] = cohort_hint
        return body
