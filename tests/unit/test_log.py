"""Unit tests for _log.py — LocalRunLog."""
from __future__ import annotations

from pathlib import Path

import pytest

from l6e._types import (
    BudgetMode,
    CallRecord,
    OnBudgetExceeded,
    PipelinePolicy,
    PromptComplexity,
    RunSummary,
    StageRoutingHint,
)


def make_summary(
    *,
    run_id: str = "r1",
    budget: float = 0.50,
    total_cost: float = 0.10,
    reroutes: int = 1,
    records: tuple[CallRecord, ...] = (),
) -> RunSummary:
    policy = PipelinePolicy(
        budget=budget,
        budget_mode=BudgetMode.REROUTE,
        on_budget_exceeded=OnBudgetExceeded.RETURN_PARTIAL,
        reroute_threshold=0.75,
        stage_routing={"retrieval": StageRoutingHint.CLOUD_STANDARD},
        stage_overrides={"final_reasoning": BudgetMode.HALT},
    )
    return RunSummary(
        run_id=run_id,
        policy=policy,
        total_cost=total_cost,
        calls_made=len(records),
        reroutes=reroutes,
        savings_usd=0.05,
        records=records,
        overhead_usd=0.01,
        overhead_calls=3,
        net_savings_usd=0.04,
    )


def make_record(call_index: int = 0) -> CallRecord:
    return CallRecord(
        call_index=call_index,
        model_requested="gpt-4o",
        model_used="ollama/qwen2.5:7b",
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=0.001,
        rerouted=True,
        elapsed_ms=200.0,
        stage="retrieval",
        prompt_complexity=PromptComplexity.LOW,
    )


# ---------------------------------------------------------------------------
# append creates file / directory
# ---------------------------------------------------------------------------


def test_append_creates_directory_and_file(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    log = LocalRunLog(path=log_path)
    log.append(make_summary())
    assert log_path.exists()


def test_append_to_existing_file_adds_line(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    log = LocalRunLog(path=log_path)
    log.append(make_summary(run_id="r1"))
    log.append(make_summary(run_id="r2"))
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# read_recent
# ---------------------------------------------------------------------------


def test_read_recent_missing_file_returns_empty(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log = LocalRunLog(path=tmp_path / "nonexistent" / "runs.jsonl")
    assert log.read_recent() == []


def test_read_recent_returns_last_n(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    log = LocalRunLog(path=log_path)
    for i in range(5):
        log.append(make_summary(run_id=f"r{i}"))

    recent = log.read_recent(n=2)
    assert len(recent) == 2
    assert recent[0].run_id == "r3"
    assert recent[1].run_id == "r4"


def test_read_recent_default_returns_all_when_fewer_than_100(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log_path = tmp_path / ".l6e" / "runs.jsonl"
    log = LocalRunLog(path=log_path)
    for i in range(3):
        log.append(make_summary(run_id=f"r{i}"))

    recent = log.read_recent()
    assert len(recent) == 3


# ---------------------------------------------------------------------------
# Round-trip: RunSummary → JSONL → RunSummary
# ---------------------------------------------------------------------------


def test_round_trip_basic_fields(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log = LocalRunLog(path=tmp_path / ".l6e" / "runs.jsonl")
    original = make_summary(run_id="round-trip-1", total_cost=0.123, reroutes=2)
    log.append(original)

    restored = log.read_recent(1)[0]
    assert restored.run_id == original.run_id
    assert restored.total_cost == pytest.approx(original.total_cost)
    assert restored.reroutes == original.reroutes
    assert restored.savings_usd == pytest.approx(original.savings_usd)
    assert restored.overhead_usd == pytest.approx(original.overhead_usd)
    assert restored.overhead_calls == original.overhead_calls
    assert restored.net_savings_usd == pytest.approx(original.net_savings_usd)
    assert restored.calls_made == original.calls_made


def test_round_trip_enum_fields(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log = LocalRunLog(path=tmp_path / ".l6e" / "runs.jsonl")
    original = make_summary()
    log.append(original)

    restored = log.read_recent(1)[0]
    assert restored.policy.budget_mode == BudgetMode.REROUTE
    assert restored.policy.on_budget_exceeded == OnBudgetExceeded.RETURN_PARTIAL
    assert restored.policy.stage_routing["retrieval"] == StageRoutingHint.CLOUD_STANDARD
    assert restored.policy.stage_overrides["final_reasoning"] == BudgetMode.HALT


def test_round_trip_nested_policy(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log = LocalRunLog(path=tmp_path / ".l6e" / "runs.jsonl")
    original = make_summary(budget=0.75)
    log.append(original)

    restored = log.read_recent(1)[0]
    assert restored.policy.budget == pytest.approx(0.75)
    assert restored.policy.reroute_threshold == pytest.approx(0.75)


def test_round_trip_call_records(tmp_path: Path) -> None:
    from l6e._log import LocalRunLog

    log = LocalRunLog(path=tmp_path / ".l6e" / "runs.jsonl")
    record = make_record(call_index=0)
    original = make_summary(records=(record,))
    log.append(original)

    restored = log.read_recent(1)[0]
    assert isinstance(restored.records, tuple)
    assert len(restored.records) == 1
    r = restored.records[0]
    assert r.model_requested == "gpt-4o"
    assert r.model_used == "ollama/qwen2.5:7b"
    assert r.rerouted is True
    assert r.prompt_complexity == PromptComplexity.LOW
    assert r.stage == "retrieval"
