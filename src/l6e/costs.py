"""LiteLLM-backed cost estimator."""
from __future__ import annotations

import litellm


class LiteLLMCostEstimator:
    """Estimates LLM call cost using litellm's model cost table.

    Callers supply token counts directly — this class does not tokenize.
    Falls back to 0.0 for unknown models or any litellm exception.
    """

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Return estimated cost in USD, or 0.0 if model is unknown."""
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return prompt_cost + completion_cost
        except Exception:
            return 0.0
