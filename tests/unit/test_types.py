"""Unit tests for _types.py — value objects, enums, and PipelinePolicy.from_toml()."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_budget_mode_values() -> None:
    from l6e._types import BudgetMode

    assert BudgetMode.HALT == "halt"
    assert BudgetMode.REROUTE == "reroute"
    assert BudgetMode.WARN == "warn"


def test_on_budget_exceeded_values() -> None:
    from l6e._types import OnBudgetExceeded

    assert OnBudgetExceeded.RAISE == "raise"
    assert OnBudgetExceeded.RETURN_PARTIAL == "partial"
    assert OnBudgetExceeded.RETURN_EMPTY == "empty"
    assert OnBudgetExceeded.RETURN_FALLBACK == "fallback"


def test_prompt_complexity_values() -> None:
    from l6e._types import PromptComplexity

    assert PromptComplexity.LOW == "low"
    assert PromptComplexity.MEDIUM == "medium"
    assert PromptComplexity.HIGH == "high"


def test_stage_routing_hint_values() -> None:
    from l6e._types import StageRoutingHint

    assert StageRoutingHint.LOCAL == "local"
    assert StageRoutingHint.CLOUD_STANDARD == "cloud_standard"
    assert StageRoutingHint.CLOUD_FRONTIER == "cloud_frontier"
    assert StageRoutingHint.INHERIT == "inherit"


def test_pipeline_policy_defaults() -> None:
    from l6e._types import BudgetMode, OnBudgetExceeded, PipelinePolicy

    policy = PipelinePolicy(budget=1.00)
    assert policy.budget == 1.00
    assert policy.budget_mode == BudgetMode.HALT
    assert policy.on_budget_exceeded == OnBudgetExceeded.RAISE
    assert policy.fallback_result is None
    assert policy.latency_sla is None
    assert policy.reroute_threshold == 0.8
    assert policy.stage_routing == {}
    assert policy.stage_overrides == {}


def test_pipeline_policy_is_frozen() -> None:
    from l6e._types import PipelinePolicy

    policy = PipelinePolicy(budget=1.00)
    with pytest.raises((AttributeError, TypeError)):
        policy.budget = 2.00  # type: ignore[misc]


def test_pipeline_policy_from_toml_round_trip() -> None:
    from l6e._types import BudgetMode, OnBudgetExceeded, PipelinePolicy, StageRoutingHint

    policy = PipelinePolicy.from_toml(FIXTURES_DIR / "policy.toml")

    assert policy.budget == 0.50
    assert policy.budget_mode == BudgetMode.REROUTE
    assert policy.on_budget_exceeded == OnBudgetExceeded.RETURN_PARTIAL
    assert policy.reroute_threshold == 0.75

    assert policy.stage_routing["retrieval"] == StageRoutingHint.CLOUD_STANDARD
    assert policy.stage_routing["summarization"] == StageRoutingHint.LOCAL
    assert policy.stage_routing["reasoning"] == StageRoutingHint.CLOUD_FRONTIER
    assert policy.stage_routing["formatting"] == StageRoutingHint.LOCAL

    assert policy.stage_overrides["final_reasoning"] == BudgetMode.HALT


def test_pipeline_policy_from_toml_defaults_when_sections_absent(tmp_path: Path) -> None:
    """A minimal TOML with only budget set should load with all defaults."""
    from l6e._types import BudgetMode, OnBudgetExceeded, PipelinePolicy

    toml_file = tmp_path / "minimal.toml"
    toml_file.write_text("[policy]\nbudget = 2.00\n")

    policy = PipelinePolicy.from_toml(toml_file)
    assert policy.budget == 2.00
    assert policy.budget_mode == BudgetMode.HALT
    assert policy.on_budget_exceeded == OnBudgetExceeded.RAISE
    assert policy.stage_routing == {}
    assert policy.stage_overrides == {}


def test_gate_decision_is_frozen() -> None:
    from l6e._types import GateDecision

    d = GateDecision(action="allow", target_model="gpt-4o", reason="ok")
    with pytest.raises((AttributeError, TypeError)):
        d.action = "halt"  # type: ignore[misc]


def test_budget_status_fields() -> None:
    from l6e._types import BudgetStatus

    status = BudgetStatus(
        run_id="r1",
        spent_usd=0.10,
        remaining_usd=0.40,
        budget_usd=0.50,
        calls_made=3,
        reroutes=1,
        pct_used=0.20,
        budget_pressure="low",
    )
    assert status.quality_safe_to_reroute is None
    assert status.budget_pressure == "low"


def test_call_record_is_frozen() -> None:
    from l6e._types import CallRecord

    record = CallRecord(
        call_index=0,
        model_requested="gpt-4o",
        model_used="gpt-4o-mini",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.001,
        rerouted=True,
        elapsed_ms=230.5,
    )
    with pytest.raises((AttributeError, TypeError)):
        record.cost_usd = 999.0  # type: ignore[misc]


def test_run_summary_records_are_tuple() -> None:
    from l6e._types import CallRecord, PipelinePolicy, RunSummary

    policy = PipelinePolicy(budget=1.00)
    record = CallRecord(
        call_index=0,
        model_requested="gpt-4o",
        model_used="gpt-4o",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=0.0001,
        rerouted=False,
        elapsed_ms=100.0,
    )
    summary = RunSummary(
        run_id="r1",
        policy=policy,
        total_cost=0.0001,
        calls_made=1,
        reroutes=0,
        savings_usd=0.0,
        records=(record,),
    )
    assert isinstance(summary.records, tuple)
    assert len(summary.records) == 1


def test_prompt_complexity_importable_from_public_namespace() -> None:
    """PromptComplexity must be in __all__ and accessible from `import l6e`
    since it appears in public method signatures (ctx.call, ctx.record)."""
    import l6e

    assert "PromptComplexity" in l6e.__all__
    assert hasattr(l6e, "PromptComplexity")
    from l6e import PromptComplexity

    assert PromptComplexity.LOW == "low"
    assert PromptComplexity.MEDIUM == "medium"
    assert PromptComplexity.HIGH == "high"


def test_latency_sla_field_documents_not_enforced() -> None:
    """latency_sla field must carry an inline note that it is not enforced in v0.1."""
    import inspect

    from l6e._types import PipelinePolicy

    src = inspect.getsource(PipelinePolicy)
    assert "not enforced" in src or "v0.2" in src
