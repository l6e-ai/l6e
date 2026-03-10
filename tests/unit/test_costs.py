"""Unit tests for costs.py — LiteLLMCostEstimator."""
from __future__ import annotations

from unittest.mock import patch


def test_known_model_returns_nonzero_cost() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    cost = estimator.estimate(
        model="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost > 0.0


def test_unknown_model_returns_zero() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    cost = estimator.estimate(
        model="not-a-real-model-xyz-999",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost == 0.0


def test_zero_tokens_returns_zero() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    cost = estimator.estimate(model="gpt-4o-mini", prompt_tokens=0, completion_tokens=0)
    assert cost == 0.0


def test_cost_scales_with_tokens() -> None:
    """More tokens → higher cost for a known model."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    small = estimator.estimate(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    large = estimator.estimate(model="gpt-4o-mini", prompt_tokens=10000, completion_tokens=5000)
    assert large > small


def test_estimator_handles_litellm_exception() -> None:
    """If litellm raises unexpectedly, cost falls back to 0.0."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    with patch("litellm.cost_per_token", side_effect=Exception("boom")):
        cost = estimator.estimate(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    assert cost == 0.0
