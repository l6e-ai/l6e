"""Parity test for ``ConstraintGate`` against the shared golden matrix (L6E-40).

This matrix is the single source of truth for cloud/core parity. The
JSON ships at ``tests/fixtures/gate_parity_matrix.json`` inside the ``l6e``
package. ``hosted-edge/tests/test_gate_parity.py`` reads the same file from
the sibling ``l6e`` tree in the monorepo and exercises ``POST /v1/authorize``.
Both sides must agree on every case.

If you add a case to the matrix, run both test suites before merging.
If you change the gate decision ladder, regenerate the matrix (and
update ``l6e._gate_core`` + ``hosted-edge.enforcement.gate_core`` in
lockstep).
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from l6e._types import BudgetMode, PipelinePolicy, PromptComplexity, StageRoutingHint
from l6e.gate import ConstraintGate
from tests.conftest import FakeRouter, FakeStore

# Fixture ships with the l6e package (standalone sub-repo friendly).
_MATRIX_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "gate_parity_matrix.json"


def _load_matrix() -> list[dict]:
    with open(_MATRIX_PATH) as f:
        doc = json.load(f)
    return doc["cases"]


_CASES = _load_matrix()


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_core_gate_matches_golden_matrix(case: dict) -> None:
    """Every matrix case must produce the documented action + reason.

    ``FakeRouter`` always returns a local model so reroute decisions
    materialize cleanly (the ``:no_local_model`` fallback is covered
    by ``test_gate.py`` — here we care about the pure decision, not
    the router fallback).
    """
    policy = PipelinePolicy(
        budget=float(case["budget"]),
        budget_mode=BudgetMode(case["budget_mode"]),
        reroute_threshold=float(case["reroute_threshold"]),
        stage_overrides={
            k: BudgetMode(v) for k, v in (case.get("stage_overrides") or {}).items()
        },
        stage_routing={
            k: StageRoutingHint(v) for k, v in (case.get("stage_routing") or {}).items()
        },
    )
    complexity = (
        PromptComplexity(case["complexity"]) if case.get("complexity") else None
    )

    gate = ConstraintGate(policy=policy, router=FakeRouter(model="ollama/parity:test"))
    store = FakeStore(budget=float(case["budget"]), spent_amount=Decimal(case["spent"]))

    decision = gate.check(
        store,
        model="gpt-4o",
        estimated_cost=Decimal(case["estimated_cost"]),
        stage=case.get("stage"),
        complexity=complexity,
    )

    assert decision.action == case["expected_action"], (
        f"[{case['name']}] core action={decision.action} reason={decision.reason}"
    )
    assert decision.reason == case["expected_reason"], (
        f"[{case['name']}] core reason={decision.reason}"
    )


def test_matrix_has_minimum_coverage() -> None:
    """Acceptance: the matrix covers at least 20 cases (L6E-40)."""
    assert len(_CASES) >= 20, f"matrix has {len(_CASES)} cases; ticket requires ≥20"


def test_matrix_covers_every_priority_lane() -> None:
    """At least one case per priority lane in the ladder.

    Enforces that adding new lanes to ``_gate_core`` requires matrix
    updates; catches accidental deletion of an entire category.
    """
    reasons = {c["expected_reason"] for c in _CASES}
    required = {
        "allow",
        "budget_pressure:halt",
        "budget_pressure:reroute",
        "warn:budget_pressure",
        "stage_override:halt",
        "stage_override:reroute",
        "stage_override:warn",
        "stage_routing:local",
        "stage_routing:cloud_standard",
        "allow:frontier_protected",
    }
    missing = required - reasons
    assert not missing, f"matrix missing reason coverage: {sorted(missing)}"
