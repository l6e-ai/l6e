"""Pure gate-decision core — the single source of truth for l6e enforcement semantics.

This module is imported by:

- ``l6e.gate.ConstraintGate`` (OSS, in-process): wraps ``decide`` with an
  ``ILocalRouter`` to resolve reroute targets to a concrete local model.
- ``hosted-edge`` ``/v1/authorize`` (cloud): mirrors this logic on the
  server side so SDK callers and MCP clients get identical decisions.
  The cloud copy lives in ``hosted-edge/src/enforcement/gate_core.py`` and
  is kept honest by the golden parity matrix in
  ``shared_fixtures/gate_parity_matrix.json`` (see L6E-40).

Keep this module pure:

- No I/O, no logging, no side effects.
- No store / router / HTTP dependencies.
- Inputs must be normalized before calling (``estimated_cost`` already
  multiplied by any calibration factor, ``spent`` already in the same
  units as ``budget``).

Priority ladder (matches docstring on ``ConstraintGate``):

1. ``stage_overrides``  — explicit ``BudgetMode`` per stage; wins over
   budget pressure.
2. Over-budget guard    — would ``spent + estimated_cost`` exceed the
   hard budget ceiling? Halt.
3. ``stage_routing``    — tier hint per stage (``local`` / ``cloud_standard``
   / ``cloud_frontier`` / ``inherit``).
4. Budget pressure      — ``spent / budget >= reroute_threshold``.
5. Allow                — default.

``complexity`` is accepted today but not consulted by any path. It's
reserved for future Margin-layer routing decisions and kept on the
signature so server and core agree on the accepted input surface. Do
not remove.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from l6e._types import BudgetMode, PromptComplexity, StageRoutingHint

GateAction = Literal["allow", "reroute", "halt"]


@dataclass(frozen=True)
class GateCoreOutcome:
    """Pure gate decision — carries enough for any caller to resolve a target.

    ``action`` is the canonical allow/reroute/halt verdict. ``reason``
    is a stable, machine-greppable tag; parity tests match the prefix
    (e.g. ``"budget_pressure:"``) rather than the full string so cloud
    and core can drift freely on suffixes if needed.

    ``reroute_tier`` is set when the reroute was driven by
    ``stage_routing`` (so callers can route to the requested tier
    instead of whatever their default is). ``wants_local_reroute`` is
    the stronger signal: ``True`` when the decision specifically demands
    a local model (stage override REROUTE, stage_routing LOCAL, or
    budget-pressure REROUTE). Cloud callers with no local model can
    treat this as a hint or escalate to ``halt`` per deployment policy.
    """

    action: GateAction
    reason: str
    reroute_tier: StageRoutingHint | None = None
    wants_local_reroute: bool = False
    budget_pressure_triggered: bool = False


def decide(
    *,
    budget: Decimal,
    spent: Decimal,
    estimated_cost: Decimal,
    budget_mode: BudgetMode = BudgetMode.HALT,
    reroute_threshold: Decimal = Decimal("0.8"),
    stage: str | None = None,
    complexity: PromptComplexity | None = None,
    stage_overrides: dict[str, BudgetMode] | None = None,
    stage_routing: dict[str, StageRoutingHint] | None = None,
) -> GateCoreOutcome:
    """Return a pure gate outcome from normalized inputs.

    Args:
        budget: Hard budget ceiling (USD).
        spent: Spend accumulated so far in the run (USD).
        estimated_cost: Projected calibrated cost of this specific
            call (USD). Callers are expected to have already applied
            any calibration factor.
        budget_mode: Global action to take when ``spent / budget >=
            reroute_threshold``. Defaults to ``HALT``.
        reroute_threshold: Ratio at which budget pressure triggers.
            Defaults to ``0.8``.
        stage: Optional stage label for this call (e.g. ``"planning"``,
            ``"final_reasoning"``).
        complexity: Optional ``PromptComplexity`` hint. Reserved; not
            consulted today.
        stage_overrides: Optional per-stage ``BudgetMode`` overrides.
        stage_routing: Optional per-stage ``StageRoutingHint`` hints.

    Returns:
        ``GateCoreOutcome`` with the decision plus enough metadata for
        deployment-specific reroute-target resolution.
    """
    del complexity  # reserved; parity contract surface only

    stage_overrides = stage_overrides or {}
    stage_routing = stage_routing or {}

    # ------------------------------------------------------------------
    # 1. stage_overrides — explicit BudgetMode per stage wins over all
    #    budget-pressure logic. We still apply the over-budget guard
    #    *after* this for ``WARN`` / unknown-mode fall-through paths.
    # ------------------------------------------------------------------
    if stage is not None and stage in stage_overrides:
        override = stage_overrides[stage]
        if override == BudgetMode.HALT:
            return GateCoreOutcome(
                action="halt",
                reason="stage_override:halt",
            )
        if override == BudgetMode.REROUTE:
            return GateCoreOutcome(
                action="reroute",
                reason="stage_override:reroute",
                wants_local_reroute=True,
            )
        return GateCoreOutcome(action="allow", reason="stage_override:warn")

    # ------------------------------------------------------------------
    # 2. Over-budget guard — projected call would push spend past the
    #    hard ceiling. Halts regardless of mode or stage routing.
    # ------------------------------------------------------------------
    if spent + estimated_cost > budget:
        return GateCoreOutcome(
            action="halt",
            reason="budget_pressure:halt",
            budget_pressure_triggered=True,
        )

    # ------------------------------------------------------------------
    # 3. stage_routing — tier hint per stage. ``INHERIT`` falls through.
    # ------------------------------------------------------------------
    if stage is not None and stage in stage_routing:
        hint = stage_routing[stage]
        if hint == StageRoutingHint.LOCAL:
            return GateCoreOutcome(
                action="reroute",
                reason="stage_routing:local",
                reroute_tier=StageRoutingHint.LOCAL,
                wants_local_reroute=True,
            )
        if hint == StageRoutingHint.CLOUD_STANDARD:
            return GateCoreOutcome(
                action="allow",
                reason="stage_routing:cloud_standard",
                reroute_tier=StageRoutingHint.CLOUD_STANDARD,
            )
        if hint == StageRoutingHint.CLOUD_FRONTIER:
            return GateCoreOutcome(
                action="allow",
                reason="allow:frontier_protected",
                reroute_tier=StageRoutingHint.CLOUD_FRONTIER,
            )
        # INHERIT → fall through.

    # ------------------------------------------------------------------
    # 4. Budget pressure — spent / budget vs reroute_threshold.
    # ------------------------------------------------------------------
    if budget > 0:
        spent_ratio = spent / budget
        is_over_threshold = spent_ratio >= reroute_threshold
    else:
        is_over_threshold = False

    if is_over_threshold:
        if budget_mode == BudgetMode.HALT:
            return GateCoreOutcome(
                action="halt",
                reason="budget_pressure:halt",
                budget_pressure_triggered=True,
            )
        if budget_mode == BudgetMode.REROUTE:
            return GateCoreOutcome(
                action="reroute",
                reason="budget_pressure:reroute",
                wants_local_reroute=True,
                budget_pressure_triggered=True,
            )
        return GateCoreOutcome(
            action="allow",
            reason="warn:budget_pressure",
            budget_pressure_triggered=True,
        )

    # ------------------------------------------------------------------
    # 5. Default allow.
    # ------------------------------------------------------------------
    return GateCoreOutcome(action="allow", reason="allow")
