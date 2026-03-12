"""Unit tests for costs.py — LiteLLMCostEstimator."""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest


def test_known_model_returns_nonzero_cost() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    cost = estimator.estimate(
        model="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost > 0.0


def test_unknown_model_returns_zero_when_fallback_disabled() -> None:
    """With fallback explicitly disabled (0.0), unknown model costs nothing."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cost = estimator.estimate(
            model="not-a-real-model-xyz-999",
            prompt_tokens=1000,
            completion_tokens=500,
        )
    assert cost == 0.0


def test_unknown_model_uses_default_fallback_rate() -> None:
    """Default estimator applies the conservative fallback rate for unknown models."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()  # default fallback_cost_per_1k = 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cost = estimator.estimate(
            model="not-a-real-model-xyz-999",
            prompt_tokens=800,
            completion_tokens=200,
        )
    # (800 + 200) / 1000 * 0.01 = 0.01
    assert cost == 0.01


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


def test_unknown_model_warns() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.estimate(
            model="not-a-real-model-xyz-999",
            prompt_tokens=1000,
            completion_tokens=500,
        )
    assert len(w) == 1
    assert "not-a-real-model-xyz-999" in str(w[0].message)
    assert "unknown model" in str(w[0].message).lower()


def test_estimator_handles_litellm_exception_with_fallback() -> None:
    """If litellm raises unexpectedly, the fallback rate is applied."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()  # default fallback = 0.01/1k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch("litellm.cost_per_token", side_effect=Exception("boom")):
            cost = estimator.estimate(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    # (100 + 50) / 1000 * 0.01 = 0.0015
    assert cost == pytest.approx(0.0015)


def test_estimator_handles_litellm_exception_fallback_disabled() -> None:
    """With fallback disabled, litellm exception returns 0.0."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch("litellm.cost_per_token", side_effect=Exception("boom")):
            cost = estimator.estimate(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
    assert cost == 0.0


def test_estimate_with_metadata_known_model() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    meta = estimator.estimate_with_metadata(
        model="gpt-4o-mini",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert meta.cost_usd > 0.0
    assert meta.model_pricing_known is True
    assert meta.pricing_confidence == "high"
    assert meta.pricing_source == "litellm_table"
    assert meta.warning is None


def test_estimate_with_metadata_unknown_model_marks_low_confidence() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model="not-a-real-model-xyz-999",
            prompt_tokens=100,
            completion_tokens=50,
        )
    assert meta.model_pricing_known is False
    assert meta.pricing_confidence == "low"
    assert meta.pricing_source == "fallback_rate"
    assert meta.warning is not None
