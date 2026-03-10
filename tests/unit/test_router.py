"""Unit tests for router.py — LocalRouter smoke test."""
from __future__ import annotations


def test_best_local_model_returns_string_or_none() -> None:
    """LocalRouter.best_local_model() returns an 'ollama/...' string or None — never raises."""
    from l6e.router import LocalRouter

    result = LocalRouter().best_local_model()
    assert result is None or (isinstance(result, str) and result.startswith("ollama/"))
