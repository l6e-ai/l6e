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


def test_family_version_fallback_prices_future_claude_release() -> None:
    """A brand-new Claude release prices from the newest known family member.

    This is the "real issue surfaced and fixed" from L6E-45: the blanket
    $0.01/1k fallback under-prices frontier models by 5-20x. Family-version
    fallback uses actual Anthropic rates for the newest known Opus/Sonnet/
    Haiku we've seen, while ``pricing_source='family_version_fallback'``
    remains distinct so the data-quality audit excludes these sessions from
    calibration and Layer 2 training.
    """
    from l6e.costs import LiteLLMCostEstimator, _build_bare_key_cache

    # Pick a version far enough in the future that the subset matcher can't
    # resolve it. Tokens must not be a superset of any exact litellm key.
    future_model = "claude-opus-9.99"
    exact_tokens = {"claude", "opus", "9", "99"}
    bare_keys = _build_bare_key_cache()
    for key_tokens, _key in bare_keys:
        assert not key_tokens.issubset(exact_tokens), (
            f"precondition: no known key should be a subset of {future_model}"
        )

    estimator = LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model=future_model,
            prompt_tokens=1000,
            completion_tokens=1000,
        )
    assert meta.model_pricing_known is False
    assert meta.pricing_confidence == "low"
    assert meta.pricing_source == "family_version_fallback"
    assert meta.resolved_model is not None
    assert meta.resolved_model.startswith("claude-opus-")
    assert meta.cost_usd > Decimal("0.02"), (
        "family fallback should exceed the $0.01/1k blanket rate for Opus"
    )
    assert meta.warning is not None and "low-confidence" in meta.warning


def test_family_version_fallback_prefers_newest_known_version() -> None:
    """Fallback picks the highest-version known family member, not any member."""
    from l6e.costs import _build_bare_key_cache, _resolve_family_fallback

    bare_keys = _build_bare_key_cache()
    import re as _re
    tokens = frozenset(
        t for t in _re.split(r"[-./: ]", "claude-opus-9.99") if t
    )
    resolved = _resolve_family_fallback(tokens, bare_keys)
    assert resolved is not None
    assert resolved.startswith("claude-opus-")
    # Must be the newest opus key among candidates whose non-version tokens
    # are a subset of the input's non-version tokens — mirroring the
    # resolver's own eligibility filter.
    input_non_version = frozenset({"claude", "opus"})
    max_version = (-1, -1)
    max_key = None
    for key_tokens, key in bare_keys:
        key_non_version = frozenset(
            t for t in key_tokens if not t.isdigit()
        )
        if not key_non_version or not key_non_version.issubset(input_non_version):
            continue
        nums = sorted(int(t) for t in key_tokens if t.isdigit())
        ver = (nums[0], nums[1]) if len(nums) >= 2 else (nums[0], 0) if nums else (0, 0)
        if ver > max_version:
            max_version = ver
            max_key = key
    assert resolved == max_key, f"expected newest opus {max_key}, got {resolved}"


def test_family_version_fallback_not_triggered_for_unrelated_model() -> None:
    """Inputs with no non-version token in common with any key still fall through."""
    from l6e.costs import LiteLLMCostEstimator

    estimator = LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model="totally-unknown-model-xyz-999",
            prompt_tokens=100,
            completion_tokens=50,
        )
    assert meta.pricing_source == "fallback_rate"


def test_family_version_fallback_rebuilds_cache_when_invalidated() -> None:
    """After cache invalidation, the next estimate repopulates transparently.

    The bare-key cache is module-global; the refresh hook sets it to
    ``None`` after a snapshot refresh. ``resolve_model_id`` rebuilds it
    lazily on the next call, which is the single init path
    ``estimate_with_metadata`` depends on.
    """
    from l6e import costs as costs_mod

    costs_mod._LITELLM_BARE_KEYS = None

    estimator = costs_mod.LiteLLMCostEstimator(fallback_cost_per_1k_tokens=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model="claude-opus-9.99",
            prompt_tokens=1000,
            completion_tokens=1000,
        )
    assert meta.pricing_source == "family_version_fallback"
    assert costs_mod._LITELLM_BARE_KEYS is not None


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


# ---------------------------------------------------------------------------
# refresh_model_cost_map_async — unit tests
# ---------------------------------------------------------------------------


def test_refresh_updates_model_cost_dict(monkeypatch: object) -> None:
    """After a successful background refresh, litellm.model_cost contains the new entries."""
    import threading

    import litellm

    import l6e.costs as costs_mod

    # Reset module state so the refresh can run again.
    costs_mod._refresh_started = False
    costs_mod._LITELLM_BARE_KEYS = None

    fake_remote = {"fake-test-model-xyz": {"input_cost_per_token": 0.001}}

    # Patch the fetch to return our fake data and validation to pass.
    with (
        patch(
            "l6e.costs.GetModelCostMap.fetch_remote_model_cost_map",
            return_value=fake_remote,
        ),
        patch(
            "l6e.costs.GetModelCostMap.validate_model_cost_map",
            return_value=True,
        ),
    ):
        costs_mod.refresh_model_cost_map_async()
        # Wait for the background thread to finish.
        for t in threading.enumerate():
            if t.name == "l6e-cost-map-refresh":
                t.join(timeout=5)

    assert "fake-test-model-xyz" in litellm.model_cost
    # Clean up.
    litellm.model_cost.pop("fake-test-model-xyz", None)


def test_refresh_invalidates_bare_key_cache() -> None:
    """After refresh, the fuzzy resolver cache is cleared so new models are found."""
    import threading

    import l6e.costs as costs_mod

    # Prime the cache.
    costs_mod._LITELLM_BARE_KEYS = [("dummy",)]
    costs_mod._refresh_started = False

    fake_remote = {"new-model": {"input_cost_per_token": 0.001}}

    with (
        patch(
            "l6e.costs.GetModelCostMap.fetch_remote_model_cost_map",
            return_value=fake_remote,
        ),
        patch(
            "l6e.costs.GetModelCostMap.validate_model_cost_map",
            return_value=True,
        ),
    ):
        costs_mod.refresh_model_cost_map_async()
        for t in threading.enumerate():
            if t.name == "l6e-cost-map-refresh":
                t.join(timeout=5)

    assert costs_mod._LITELLM_BARE_KEYS is None


def test_refresh_skips_on_validation_failure() -> None:
    """If the fetched map fails validation, model_cost is not modified."""
    import threading

    import litellm

    import l6e.costs as costs_mod

    costs_mod._refresh_started = False
    original_keys = set(litellm.model_cost.keys())

    with (
        patch(
            "l6e.costs.GetModelCostMap.fetch_remote_model_cost_map",
            return_value={"bad": {}},
        ),
        patch(
            "l6e.costs.GetModelCostMap.validate_model_cost_map",
            return_value=False,
        ),
    ):
        costs_mod.refresh_model_cost_map_async()
        for t in threading.enumerate():
            if t.name == "l6e-cost-map-refresh":
                t.join(timeout=5)

    assert set(litellm.model_cost.keys()) == original_keys


def test_refresh_only_runs_once() -> None:
    """Calling refresh_model_cost_map_async multiple times only spawns one thread."""
    import threading

    import l6e.costs as costs_mod

    costs_mod._refresh_started = False

    call_count = 0

    def counting_fetch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {}

    with (
        patch(
            "l6e.costs.GetModelCostMap.fetch_remote_model_cost_map",
            side_effect=counting_fetch,
        ),
        patch(
            "l6e.costs.GetModelCostMap.validate_model_cost_map",
            return_value=False,
        ),
    ):
        costs_mod.refresh_model_cost_map_async()
        costs_mod.refresh_model_cost_map_async()
        costs_mod.refresh_model_cost_map_async()
        for t in threading.enumerate():
            if t.name == "l6e-cost-map-refresh":
                t.join(timeout=5)

    assert call_count == 1


def test_refresh_swallows_fetch_exception() -> None:
    """Network errors in the background thread don't propagate."""
    import threading

    import l6e.costs as costs_mod

    costs_mod._refresh_started = False

    with patch(
        "l6e.costs.GetModelCostMap.fetch_remote_model_cost_map",
        side_effect=Exception("network down"),
    ):
        costs_mod.refresh_model_cost_map_async()
        for t in threading.enumerate():
            if t.name == "l6e-cost-map-refresh":
                t.join(timeout=5)
    # No exception raised — test passes if we get here.
