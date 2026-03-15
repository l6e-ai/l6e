"""Unit tests for costs.py — LiteLLMCostEstimator and resolve_model_id."""
from __future__ import annotations

import warnings
from decimal import Decimal
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
    assert cost == Decimal("0.01")


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
    assert cost == Decimal("0.0015")


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
    assert meta.resolved_model is None


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
    assert meta.resolved_model is None


# ---------------------------------------------------------------------------
# resolve_model_id — unit tests
# ---------------------------------------------------------------------------


def test_resolve_model_id_claude_sonnet_46_variants() -> None:
    """Vendor-internal Cursor suffixes resolve to the bare LiteLLM key."""
    from l6e.costs import resolve_model_id

    assert resolve_model_id("claude-4.6-sonnet-medium-thinking") == "claude-sonnet-4-6"
    assert resolve_model_id("claude-4.6-sonnet-large") == "claude-sonnet-4-6"
    assert resolve_model_id("claude-4.6-sonnet") == "claude-sonnet-4-6"


def test_resolve_model_id_claude_sonnet_45() -> None:
    from l6e.costs import resolve_model_id

    assert resolve_model_id("claude-4.5-sonnet-medium") == "claude-sonnet-4-5"


def test_resolve_model_id_claude_opus_46() -> None:
    from l6e.costs import resolve_model_id

    assert resolve_model_id("claude-opus-4-6-thinking") == "claude-opus-4-6"


def test_resolve_model_id_gpt4o_variants() -> None:
    from l6e.costs import resolve_model_id

    assert resolve_model_id("gpt-4o-medium") == "gpt-4o"
    assert resolve_model_id("gpt-4o-large") == "gpt-4o"


def test_resolve_model_id_returns_none_for_unrecognisable() -> None:
    from l6e.costs import resolve_model_id

    assert resolve_model_id("totally-unknown-model-xyz-999") is None


def test_estimator_resolves_vendor_model_id() -> None:
    """Estimator silently resolves a vendor ID and returns high-confidence cost."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        meta = estimator.estimate_with_metadata(
            model="claude-4.6-sonnet-medium-thinking",
            prompt_tokens=1000,
            completion_tokens=500,
        )
    assert meta.model_pricing_known is True
    assert meta.pricing_confidence == "high"
    assert meta.pricing_source == "litellm_table_resolved"
    assert meta.resolved_model == "claude-sonnet-4-6"
    assert meta.warning is None
    assert len(w) == 0, "no warning should be emitted for a successfully resolved model"


def test_estimator_resolved_cost_is_nonzero() -> None:
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator()
    cost = estimator.estimate(
        model="claude-4.6-sonnet-medium-thinking",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    assert cost > 0.0
