"""LocalRouter — wraps forge hardware detection behind ILocalRouter.

Lazy-initialised: hardware probing happens on first call to best_local_model(),
not at import time. Result is cached — hardware doesn't change mid-run.
"""
from __future__ import annotations


class LocalRouter:
    """Returns the best locally-runnable Ollama model tag, or None.

    Uses forge's get_system_profile() and suggest_models() to probe hardware
    and rank models. Filters to fits_local=True, provider="ollama", and
    returns the top result prefixed as "ollama/{tag}".
    """

    def __init__(self) -> None:
        self._cached: str | None | bool = False  # False = not yet computed

    def best_local_model(self) -> str | None:
        if self._cached is not False:
            return self._cached  # type: ignore[return-value]

        result = self._probe()
        self._cached = result
        return result

    def _probe(self) -> str | None:
        try:
            from l6e_forge.models.auto import (  # type: ignore[import-not-found]
                AutoHintQuality,
                AutoHintQuantization,
                AutoHints,
                AutoHintTask,
                get_system_profile,
                suggest_models,
            )
        except ImportError:
            return None

        try:
            profile = get_system_profile()
        except Exception:
            return None

        if not profile.has_ollama:
            return None

        try:
            hints = AutoHints(
                provider_order=["ollama"],
                task=AutoHintTask.ASSISTANT,
                quality=AutoHintQuality.BALANCED,
                quantization=AutoHintQuantization.AUTO,
            )
            suggestions = suggest_models(profile, hints)
        except Exception:
            return None

        for suggestion in suggestions:
            if (
                suggestion.fits_local
                and suggestion.provider == "ollama"
                and suggestion.provider_tag
            ):
                return f"ollama/{suggestion.provider_tag}"

        return None
