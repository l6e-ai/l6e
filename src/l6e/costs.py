"""LiteLLM-backed cost estimator."""
from __future__ import annotations

import warnings

import litellm


class LiteLLMCostEstimator:
    """Estimates LLM call cost using litellm's model cost table.

    Callers supply token counts directly — this class does not tokenize.

    When a model is not in litellm's table, a warning is always emitted so
    operators can detect misconfigured model strings in logs. If
    ``fallback_cost_per_1k_tokens`` is set to a non-zero value, that rate is
    used instead of returning 0.0 — ensuring the gate can still fire.
    """

    def __init__(self, fallback_cost_per_1k_tokens: float = 0.01) -> None:
        self._fallback_cost_per_1k = fallback_cost_per_1k_tokens

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD.

        Returns 0.0 (or the configured fallback rate) for unknown models.
        Always emits a warning when the model is not recognised by litellm.
        """
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return prompt_cost + completion_cost
        except Exception:
            warnings.warn(
                f"l6e: unknown model '{model}' — cost cannot be estimated from litellm's "
                f"table. Falling back to {self._fallback_cost_per_1k:.4f} USD/1k tokens. "
                "Set PipelinePolicy.unknown_model_cost_per_1k_tokens (or "
                "'unknown_model_cost_per_1k_tokens' in [policy] of your l6e TOML) to "
                "the correct rate for this model, or to 0.0 to disable enforcement for "
                "unknown models.",
                stacklevel=2,
            )
            if self._fallback_cost_per_1k > 0:
                total_tokens = prompt_tokens + completion_tokens
                return total_tokens / 1000.0 * self._fallback_cost_per_1k
            return 0.0
