"""Unit tests for gate.py — ConstraintGate decision logic."""
from __future__ import annotations

from decimal import Decimal

from l6e._types import BudgetMode, PipelinePolicy, StageRoutingHint
from tests.conftest import FakeRouter, FakeStore


def make_gate(policy: PipelinePolicy, router: FakeRouter | None = None):
    from l6e.gate import ConstraintGate

    return ConstraintGate(policy=policy, router=router or FakeRouter())


def check(
    gate,
    store,
    *,
    model: str = "gpt-4o",
    cost: Decimal = Decimal("0.01"),
    stage: str | None = None,
    complexity: float | None = None,
):
    return gate.check(
        store, 
        model=model, 
        estimated_cost=cost, 
        stage=stage, 
        complexity=complexity,
    )


# ---------------------------------------------------------------------------
# The three canonical moat cases from the plan spec
# ---------------------------------------------------------------------------


def test_stage_routing_local_reroutes_to_local_model() -> None:
    """stage_routing LOCAL → reroute to local model."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"summarization": StageRoutingHint.LOCAL},
    )
    gate = make_gate(policy, FakeRouter(model="ollama/qwen2.5:7b"))
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store, stage="summarization")

    assert decision.action == "reroute"
    assert decision.target_model == "ollama/qwen2.5:7b"
    assert "stage_routing" in decision.reason


def test_stage_override_halt_overrides_global_reroute_mode() -> None:
    """stage_override HALT wins over global REROUTE mode."""
    policy = PipelinePolicy(
        budget=1.00,
        budget_mode=BudgetMode.REROUTE,
        stage_overrides={"final_reasoning": BudgetMode.HALT},
    )
    gate = make_gate(policy, FakeRouter())
    store = FakeStore(budget=1.00, spent_amount=0.95)
    decision = check(gate, store, stage="final_reasoning")

    assert decision.action == "halt"


def test_stage_routing_cloud_frontier_always_allows() -> None:
    """stage_routing CLOUD_FRONTIER → always allow."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"reasoning": StageRoutingHint.CLOUD_FRONTIER},
    )
    gate = make_gate(policy, FakeRouter())
    store = FakeStore(budget=1.00, spent_amount=0.20)
    decision = check(gate, store, stage="reasoning")

    assert decision.action == "allow"
    assert decision.target_model == "gpt-4o"


# ---------------------------------------------------------------------------
# Budget pressure paths
# ---------------------------------------------------------------------------


def test_budget_healthy_no_stage_hints_allows() -> None:
    """Budget well under threshold, no stage hints → allow."""
    policy = PipelinePolicy(budget=1.00)
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store)

    assert decision.action == "allow"
    assert decision.target_model == "gpt-4o"


def test_budget_pressure_reroute_mode_reroutes() -> None:
    """Budget >= reroute_threshold (0.8) with REROUTE mode → reroute."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.REROUTE)
    gate = make_gate(policy, FakeRouter(model="ollama/qwen2.5:7b"))
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store)

    assert decision.action == "reroute"
    assert decision.target_model == "ollama/qwen2.5:7b"
    assert "budget_pressure" in decision.reason


def test_budget_pressure_halt_mode_halts() -> None:
    """Budget >= reroute_threshold with HALT mode → halt."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.HALT)
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store)

    assert decision.action == "halt"
    assert "budget_pressure" in decision.reason


def test_budget_pressure_warn_mode_allows_with_reason() -> None:
    """Budget >= reroute_threshold with WARN mode → allow with warn reason."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.WARN)
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store)

    assert decision.action == "allow"
    assert "warn" in decision.reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_stage_routing_local_no_local_model_halts() -> None:
    """stage_routing LOCAL but router returns None → halt with no_local_model reason."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"extraction": StageRoutingHint.LOCAL},
    )
    gate = make_gate(policy, FakeRouter(model=None))
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store, stage="extraction")

    assert decision.action == "halt"
    assert "no_local_model" in decision.reason


def test_stage_none_falls_through_to_budget_pressure() -> None:
    """stage=None skips stage logic and hits budget pressure."""
    policy = PipelinePolicy(
        budget=1.00,
        budget_mode=BudgetMode.REROUTE,
        stage_routing={"summarization": StageRoutingHint.LOCAL},
    )
    gate = make_gate(policy, FakeRouter(model="ollama/qwen2.5:7b"))
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store, stage=None)

    assert decision.action == "reroute"
    assert "budget_pressure" in decision.reason


def test_stage_routing_inherit_falls_through_to_budget_pressure() -> None:
    """StageRoutingHint.INHERIT falls through to budget pressure logic."""
    policy = PipelinePolicy(
        budget=1.00,
        budget_mode=BudgetMode.HALT,
        stage_routing={"summarization": StageRoutingHint.INHERIT},
    )
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store, stage="summarization")

    assert decision.action == "halt"
    assert "budget_pressure" in decision.reason


def test_estimated_cost_would_exceed_budget_halts() -> None:
    """estimated_cost alone would push spend over budget → halt regardless of mode."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.REROUTE)
    gate = make_gate(policy, FakeRouter(model="ollama/qwen2.5:7b"))
    store = FakeStore(budget=1.00, spent_amount=0.90)
    decision = check(gate, store, cost=Decimal("0.20"))

    assert decision.action == "halt"


def test_stage_override_warn_falls_through_to_allow() -> None:
    """stage_override WARN is informational only → allow."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_overrides={"summarization": BudgetMode.WARN},
    )
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store, stage="summarization")

    assert decision.action == "allow"


def test_stage_override_reroute_reroutes() -> None:
    """stage_override REROUTE → reroute to local model."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_overrides={"summarization": BudgetMode.REROUTE},
    )
    gate = make_gate(policy, FakeRouter(model="ollama/llama3.2:3b"))
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store, stage="summarization")

    assert decision.action == "reroute"
    assert decision.target_model == "ollama/llama3.2:3b"


def test_stage_routing_cloud_standard_allows() -> None:
    """stage_routing CLOUD_STANDARD → allow."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"drafting": StageRoutingHint.CLOUD_STANDARD},
    )
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.10)
    decision = check(gate, store, model="gpt-4o-mini", stage="drafting")

    assert decision.action == "allow"
    assert "stage_routing" in decision.reason


def test_budget_pressure_reroute_mode_no_local_halts() -> None:
    """Budget pressure REROUTE but no local model → halt with no_local_model reason."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.REROUTE)
    gate = make_gate(policy, FakeRouter(model=None))
    store = FakeStore(budget=1.00, spent_amount=0.85)
    decision = check(gate, store)

    assert decision.action == "halt"
    assert "no_local_model" in decision.reason


def test_exact_threshold_triggers_pressure() -> None:
    """spent / budget exactly at reroute_threshold (0.8) triggers pressure."""
    policy = PipelinePolicy(budget=1.00, budget_mode=BudgetMode.HALT)
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.80)
    decision = check(gate, store)

    assert decision.action == "halt"


def test_cloud_frontier_stage_halts_when_call_would_exceed_budget() -> None:
    """CLOUD_FRONTIER does not exempt a call from the hard budget ceiling."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"reasoning": StageRoutingHint.CLOUD_FRONTIER},
    )
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.90)
    decision = check(gate, store, cost=Decimal("0.20"), stage="reasoning")

    assert decision.action == "halt"


def test_cloud_standard_stage_halts_when_call_would_exceed_budget() -> None:
    """CLOUD_STANDARD does not exempt a call from the hard budget ceiling."""
    policy = PipelinePolicy(
        budget=1.00,
        stage_routing={"drafting": StageRoutingHint.CLOUD_STANDARD},
    )
    gate = make_gate(policy)
    store = FakeStore(budget=1.00, spent_amount=0.90)
    decision = check(gate, store, cost=Decimal("0.20"), stage="drafting")

    assert decision.action == "halt"


def test_zero_budget_always_allows() -> None:
    """budget=0 skips pressure check entirely and always allows."""
    policy = PipelinePolicy(budget=0, budget_mode=BudgetMode.HALT)
    gate = make_gate(policy)
    store = FakeStore(budget=0, spent_amount=0.0)
    decision = check(gate, store, cost=Decimal("0.00"))

    assert decision.action == "allow"
