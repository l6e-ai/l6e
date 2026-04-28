from __future__ import annotations

import warnings
from typing import Any

import litellm
import pytest

from l6e import costs as costs_mod


@pytest.fixture(autouse=True)
def reset_litellm_bare_key_cache() -> None:
    costs_mod._LITELLM_BARE_KEYS = None
    yield
    costs_mod._LITELLM_BARE_KEYS = None


def test_family_fallback_skips_self_when_cost_per_token_raises_for_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the diagnostic-distinguishability contract from L6E-86 / L6E-87."""
    fake = "l6e-regression-family-99-2"
    older = "l6e-regression-family-99-1"

    monkeypatch.setitem(
        litellm.model_cost,
        fake,
        {
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 2.5e-5,
            "litellm_provider": "l6e-test",
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        older,
        {
            "input_cost_per_token": 4e-6,
            "output_cost_per_token": 2e-5,
            "litellm_provider": "l6e-test",
        },
    )

    def stub_cost_per_token(
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **_: Any,
    ) -> tuple[float, float]:
        if model == fake:
            raise litellm.BadRequestError(
                "simulated registration drift",
                model=fake,
                llm_provider="",
            )
        if model == older:
            return (prompt_tokens * 4e-6, completion_tokens * 2e-5)
        raise AssertionError(f"unexpected model lookup: {model}")

    monkeypatch.setattr(litellm, "cost_per_token", stub_cost_per_token)

    estimator = costs_mod.LiteLLMCostEstimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model=fake,
            prompt_tokens=1000,
            completion_tokens=500,
        )

    assert meta.pricing_source == "family_version_fallback"
    assert meta.resolved_model == older
    assert meta.resolved_model != fake


def test_resolve_model_id_does_not_subset_match_different_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = "l6e-versioned-family-99-7"
    wrong_version = "l6e-versioned-family-99"

    monkeypatch.setitem(
        litellm.model_cost,
        wrong_version,
        {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 5e-6,
            "litellm_provider": "l6e-test",
        },
    )

    assert costs_mod.resolve_model_id(fake) is None


def test_family_fallback_returns_none_when_only_self_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = "l6e-nosibling-regression-99-2"

    monkeypatch.setitem(
        litellm.model_cost,
        fake,
        {
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 2.5e-5,
            "litellm_provider": "l6e-test",
        },
    )

    def stub_cost_per_token(
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        **_: Any,
    ) -> tuple[float, float]:
        if model == fake:
            raise litellm.BadRequestError(
                "simulated registration drift",
                model=fake,
                llm_provider="",
            )
        raise AssertionError(f"unexpected model lookup: {model}")

    monkeypatch.setattr(litellm, "cost_per_token", stub_cost_per_token)

    estimator = costs_mod.LiteLLMCostEstimator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = estimator.estimate_with_metadata(
            model=fake,
            prompt_tokens=1000,
            completion_tokens=500,
        )

    assert meta.pricing_source == "fallback_rate"
    assert meta.resolved_model is None


def test_family_fallback_returns_none_when_input_is_below_known_family() -> None:
    bare_keys = [
        (frozenset({"l6e", "bounded", "family", "4", "5"}), "l6e-bounded-family-4-5"),
        (frozenset({"l6e", "bounded", "family", "4", "7"}), "l6e-bounded-family-4-7"),
    ]

    resolved = costs_mod._resolve_family_fallback(
        frozenset({"l6e", "bounded", "family", "4", "2"}),
        bare_keys,
    )

    assert resolved is None


def test_family_fallback_chooses_newest_known_version_below_input() -> None:
    bare_keys = [
        (frozenset({"l6e", "bounded", "family", "4", "1"}), "l6e-bounded-family-4-1"),
        (frozenset({"l6e", "bounded", "family", "4", "7"}), "l6e-bounded-family-4-7"),
    ]

    resolved = costs_mod._resolve_family_fallback(
        frozenset({"l6e", "bounded", "family", "4", "5"}),
        bare_keys,
    )

    assert resolved == "l6e-bounded-family-4-1"


def test_family_fallback_chooses_table_max_when_input_is_above_known_family() -> None:
    bare_keys = [
        (frozenset({"l6e", "bounded", "family", "4", "5"}), "l6e-bounded-family-4-5"),
        (frozenset({"l6e", "bounded", "family", "4", "7"}), "l6e-bounded-family-4-7"),
    ]

    resolved = costs_mod._resolve_family_fallback(
        frozenset({"l6e", "bounded", "family", "4", "9"}),
        bare_keys,
    )

    assert resolved == "l6e-bounded-family-4-7"
