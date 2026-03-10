"""Local run log — Foundation 1.

Appends every RunSummary to .l6e/runs.jsonl after pipeline exit.
When a developer upgrades to Pro, the profiler reads this file directly —
every run since day one is immediately available. Zero retroactive instrumentation.
"""
from __future__ import annotations

import dataclasses
import json
from collections import deque
from pathlib import Path

from l6e._types import (
    BudgetMode,
    CallRecord,
    OnBudgetExceeded,
    PipelinePolicy,
    PromptComplexity,
    RunSummary,
    StageRoutingHint,
)

_DEFAULT_PATH = Path(".l6e/runs.jsonl")


class LocalRunLog:
    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self._path = path

    def append(self, summary: RunSummary) -> None:
        """Append one RunSummary as a JSON line. Creates directory if absent."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dataclasses.asdict(summary), default=str)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_recent(self, n: int = 100) -> list[RunSummary]:
        """Return the last n RunSummary entries. Returns [] if file does not exist."""
        if not self._path.exists():
            return []

        with self._path.open("r", encoding="utf-8") as f:
            tail: deque[str] = deque(f, maxlen=n)

        summaries: list[RunSummary] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                summaries.append(_summary_from_dict(json.loads(line)))
            except (KeyError, ValueError, TypeError):
                continue
        return summaries


# ---------------------------------------------------------------------------
# Deserialisation helpers
# ---------------------------------------------------------------------------


def _summary_from_dict(d: dict) -> RunSummary:  # type: ignore[type-arg]
    policy = _policy_from_dict(d["policy"])
    records = tuple(_record_from_dict(r) for r in d.get("records", []))
    return RunSummary(
        run_id=d["run_id"],
        policy=policy,
        total_cost=float(d["total_cost"]),
        calls_made=int(d["calls_made"]),
        reroutes=int(d["reroutes"]),
        savings_usd=float(d["savings_usd"]),
        records=records,
    )


def _policy_from_dict(d: dict) -> PipelinePolicy:  # type: ignore[type-arg]
    stage_routing = {
        k: StageRoutingHint(v) for k, v in d.get("stage_routing", {}).items()
    }
    stage_overrides = {
        k: BudgetMode(v) for k, v in d.get("stage_overrides", {}).items()
    }
    fallback_result: str | None = d.get("fallback_result")
    latency_sla_raw = d.get("latency_sla")
    latency_sla: float | None = float(latency_sla_raw) if latency_sla_raw is not None else None
    return PipelinePolicy(
        budget=float(d["budget"]),
        budget_mode=BudgetMode(d.get("budget_mode", BudgetMode.HALT)),
        on_budget_exceeded=OnBudgetExceeded(
            d.get("on_budget_exceeded", OnBudgetExceeded.RAISE)
        ),
        fallback_result=fallback_result,
        latency_sla=latency_sla,
        reroute_threshold=float(d.get("reroute_threshold", 0.8)),
        stage_routing=stage_routing,
        stage_overrides=stage_overrides,
    )


def _record_from_dict(d: dict) -> CallRecord:  # type: ignore[type-arg]
    complexity_raw = d.get("prompt_complexity")
    prompt_complexity = PromptComplexity(complexity_raw) if complexity_raw is not None else None
    stage: str | None = d.get("stage")
    return CallRecord(
        call_index=int(d["call_index"]),
        model_requested=str(d["model_requested"]),
        model_used=str(d["model_used"]),
        prompt_tokens=int(d["prompt_tokens"]),
        completion_tokens=int(d["completion_tokens"]),
        cost_usd=float(d["cost_usd"]),
        rerouted=bool(d["rerouted"]),
        elapsed_ms=float(d["elapsed_ms"]),
        stage=stage,
        prompt_complexity=prompt_complexity,
        is_multi_turn=bool(d.get("is_multi_turn", False)),
    )
