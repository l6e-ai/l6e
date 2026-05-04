"""Parity test for ``_validate_embedding`` against the shared golden matrix (L6E-101).

This matrix is the single source of truth for the embedding wire-shape
contract that the SDK and the cloud `/v1/authorize` endpoint must agree
on. The JSON ships at ``tests/fixtures/embedding_validation_parity.json``
inside the ``l6e`` package. ``hosted-edge/tests/test_embedding_parity.py``
reads the same file from the sibling ``l6e`` tree in the monorepo and
exercises ``POST /v1/authorize``. Both sides must agree on every case.

If you add a case to the matrix, run both test suites before merging.
If you tighten or loosen one validator, update both
(``l6e.cloud._validate_embedding`` and
``hosted-edge.relay.routers.authorize._require_embedding``) in lockstep
and regenerate this matrix.

The fail-surface contract this protects: a bring-your-own embedder must
not produce something the SDK accepts locally but the server then 400s
on (which would split the operator fail-surface across
``cloud_embedding_failed`` and ``cloud_authorize_5xx``). Drift between
the two validators is a silent observability bug; this test makes it a
loud CI failure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from l6e.cloud import _validate_embedding

# Fixture ships with the l6e package (standalone sub-repo friendly).
_MATRIX_PATH = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "embedding_validation_parity.json"
)


def _inflate_input(raw: Any) -> Any:
    """Expand fixture sentinels to real Python values.

    The fixture stays strict-JSON-portable by representing NaN / inf as
    string sentinels and large repeated payloads as a ``__repeat__``
    object. This helper mirrors the equivalent helper in
    ``hosted-edge/tests/test_embedding_parity.py``; if you change one,
    change the other.

    - ``"__nan__"`` / ``"__inf__"`` / ``"__neg_inf__"`` (as a list
      element) → the corresponding ``float``.
    - ``{"__repeat__": {"value": V, "count": N}}`` (as the input
      itself) → ``[V] * N``.
    - Any other value passes through unchanged.
    """
    if isinstance(raw, dict) and "__repeat__" in raw:
        spec = raw["__repeat__"]
        return [spec["value"]] * int(spec["count"])
    if not isinstance(raw, list):
        return raw
    inflated: list[Any] = []
    for item in raw:
        if item == "__nan__":
            inflated.append(math.nan)
        elif item == "__inf__":
            inflated.append(math.inf)
        elif item == "__neg_inf__":
            inflated.append(-math.inf)
        else:
            inflated.append(item)
    return inflated


def _load_matrix() -> list[dict]:
    with open(_MATRIX_PATH) as f:
        doc = json.load(f)
    return doc["cases"]


_CASES = _load_matrix()


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_client_validator_matches_golden_matrix(case: dict) -> None:
    """Every matrix case must produce the documented client-side outcome.

    Validity contract:
      - ``expected == "valid"``  → ``_validate_embedding`` returns a
        non-None ``list[float]``. If ``expected_coerced`` is provided
        (typically for int→float coercion cases), the returned list
        must equal it exactly.
      - ``expected == "invalid"`` → ``_validate_embedding`` returns
        ``None``. Never raises into caller code; the iron-rule fail-open
        path consumes ``None`` to omit ``request_embedding`` from the
        cloud body and degrade to metadata-tier behavior.
    """
    inflated = _inflate_input(case["input"])
    result = _validate_embedding(inflated)

    if case["expected"] == "valid":
        assert result is not None, (
            f"[{case['name']}] expected valid; client returned None"
        )
        # Length must always be preserved on the valid path.
        assert isinstance(result, list)
        if "expected_coerced" in case:
            assert result == case["expected_coerced"], (
                f"[{case['name']}] coerced output mismatch: "
                f"got {result!r}, want {case['expected_coerced']!r}"
            )
    elif case["expected"] == "invalid":
        assert result is None, (
            f"[{case['name']}] expected invalid; client returned "
            f"{result!r}"
        )
    else:
        pytest.fail(
            f"[{case['name']}] fixture has unknown expected value "
            f"{case['expected']!r}"
        )


def test_matrix_has_minimum_coverage() -> None:
    """Acceptance: the matrix covers the full set of failure modes plus
    at least a few valid shapes. Concrete floor mirrors the L6E-40
    coverage check pattern: enough rows that adding a new validation
    rule pressures someone into adding a fixture row alongside it."""
    assert len(_CASES) >= 15, (
        f"matrix has {len(_CASES)} cases; ticket floor is \u226515"
    )


def test_matrix_covers_every_required_failure_mode() -> None:
    """Each documented rejection rule must have at least one fixture
    case. Catches accidental deletion of a category and forces new
    validation rules to land with explicit coverage rows.
    """
    required_invalid_names = {
        "rejects_above_dim_cap_4097",
        "rejects_empty_list",
        "rejects_nan_element",
        "rejects_pos_inf_element",
        "rejects_neg_inf_element",
        "rejects_bool_true_element",
        "rejects_string_element",
        "rejects_null_element",
        "rejects_string_input",
        "rejects_dict_input",
    }
    present = {c["name"] for c in _CASES}
    missing = required_invalid_names - present
    assert not missing, (
        f"matrix missing required failure-mode coverage: {sorted(missing)}"
    )


def test_matrix_covers_at_least_one_valid_case() -> None:
    """A matrix with only invalid cases would silently pass even if the
    validator flat-out rejected everything. Force a positive-path
    fixture row."""
    valid_count = sum(1 for c in _CASES if c["expected"] == "valid")
    assert valid_count >= 3, (
        f"matrix has {valid_count} valid cases; need \u22653 (typical, at-cap, coerced)"
    )


def test_matrix_case_names_are_unique() -> None:
    """Duplicate names produce indistinguishable parametrize IDs and
    mask coverage drift."""
    names = [c["name"] for c in _CASES]
    assert len(names) == len(set(names)), (
        f"duplicate fixture case names: {sorted(set(n for n in names if names.count(n) > 1))}"
    )
