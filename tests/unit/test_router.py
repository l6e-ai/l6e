"""Unit tests for router.py — LocalRouter smoke test."""
from __future__ import annotations

import sys
import types


def test_best_local_model_returns_string_or_none() -> None:
    """LocalRouter.best_local_model() returns an 'ollama/...' string or None — never raises."""
    from l6e.router import LocalRouter

    result = LocalRouter().best_local_model()
    assert result is None or (isinstance(result, str) and result.startswith("ollama/"))


# ---------------------------------------------------------------------------
# Cache hit path (line 22)
# ---------------------------------------------------------------------------


def test_best_local_model_caches_result(monkeypatch) -> None:
    """Second call returns cached value without calling _probe again."""
    from l6e.router import LocalRouter

    router = LocalRouter()
    probe_calls: list[int] = []

    original_probe = router._probe

    def counting_probe():
        probe_calls.append(1)
        return original_probe()

    monkeypatch.setattr(router, "_probe", counting_probe)

    result1 = router.best_local_model()
    result2 = router.best_local_model()

    assert result1 == result2
    assert len(probe_calls) == 1  # second call hit cache, not _probe


def test_best_local_model_cache_hit_when_none(monkeypatch) -> None:
    """Cache stores None correctly — second call must not re-probe."""
    from l6e.router import LocalRouter

    router = LocalRouter()
    probe_calls: list[int] = []

    def probe_returning_none():
        probe_calls.append(1)
        return None

    monkeypatch.setattr(router, "_probe", probe_returning_none)

    r1 = router.best_local_model()
    r2 = router.best_local_model()
    assert r1 is None
    assert r2 is None
    assert len(probe_calls) == 1


# ---------------------------------------------------------------------------
# _probe — ImportError path (lines 38-39)
# ---------------------------------------------------------------------------


def test_probe_returns_none_when_forge_not_installed(monkeypatch) -> None:
    """If l6e_forge is not importable, _probe returns None without raising."""
    # Remove any cached forge module so the import inside _probe fails
    forge_keys = [k for k in sys.modules if k.startswith("l6e_forge")]
    for k in forge_keys:
        monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setitem(sys.modules, "l6e_forge", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "l6e_forge.models", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", None)  # type: ignore[arg-type]

    from l6e.router import LocalRouter

    router = LocalRouter()
    result = router._probe()
    assert result is None


# ---------------------------------------------------------------------------
# _probe — get_system_profile() raises (lines 41-44)
# ---------------------------------------------------------------------------


def _make_fake_forge(*, has_ollama: bool = True, profile_raises: bool = False,
                     suggestions_raises: bool = False, suggestions=None):
    """Build a minimal fake l6e_forge.models.auto module."""
    class AutoHintQuality:
        BALANCED = "balanced"

    class AutoHintQuantization:
        AUTO = "auto"

    class AutoHintTask:
        ASSISTANT = "assistant"

    class AutoHints:
        def __init__(self, **kwargs):
            pass

    class _Profile:
        def __init__(self):
            self.has_ollama = has_ollama

    def get_system_profile():
        if profile_raises:
            raise RuntimeError("hardware probe failed")
        return _Profile()

    def suggest_models(profile, hints):
        if suggestions_raises:
            raise RuntimeError("suggest failed")
        return suggestions or []

    mod = types.ModuleType("l6e_forge.models.auto")
    mod.AutoHintQuality = AutoHintQuality
    mod.AutoHintQuantization = AutoHintQuantization
    mod.AutoHintTask = AutoHintTask
    mod.AutoHints = AutoHints
    mod.get_system_profile = get_system_profile
    mod.suggest_models = suggest_models
    return mod


def test_probe_returns_none_when_get_system_profile_raises(monkeypatch) -> None:
    fake_mod = _make_fake_forge(profile_raises=True)
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    assert LocalRouter()._probe() is None


# ---------------------------------------------------------------------------
# _probe — has_ollama=False (lines 46-47)
# ---------------------------------------------------------------------------


def test_probe_returns_none_when_ollama_not_present(monkeypatch) -> None:
    fake_mod = _make_fake_forge(has_ollama=False)
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    assert LocalRouter()._probe() is None


# ---------------------------------------------------------------------------
# _probe — suggest_models raises (lines 49-58)
# ---------------------------------------------------------------------------


def test_probe_returns_none_when_suggest_models_raises(monkeypatch) -> None:
    fake_mod = _make_fake_forge(has_ollama=True, suggestions_raises=True)
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    assert LocalRouter()._probe() is None


# ---------------------------------------------------------------------------
# _probe — no matching suggestion (lines 60-68)
# ---------------------------------------------------------------------------


def test_probe_returns_none_when_no_matching_suggestions(monkeypatch) -> None:
    class _NoMatchSuggestion:
        fits_local = False
        provider = "openai"
        provider_tag = "gpt-4o"

    fake_mod = _make_fake_forge(has_ollama=True, suggestions=[_NoMatchSuggestion()])
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    assert LocalRouter()._probe() is None


def test_probe_returns_none_when_suggestions_empty(monkeypatch) -> None:
    fake_mod = _make_fake_forge(has_ollama=True, suggestions=[])
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    assert LocalRouter()._probe() is None


# ---------------------------------------------------------------------------
# _probe — happy path (line 66)
# ---------------------------------------------------------------------------


def test_probe_returns_ollama_prefixed_tag_on_match(monkeypatch) -> None:
    class _GoodSuggestion:
        fits_local = True
        provider = "ollama"
        provider_tag = "qwen2.5:7b"

    fake_mod = _make_fake_forge(has_ollama=True, suggestions=[_GoodSuggestion()])
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    result = LocalRouter()._probe()
    assert result == "ollama/qwen2.5:7b"


def test_probe_skips_non_ollama_and_picks_first_ollama(monkeypatch) -> None:
    class _OpenAISuggestion:
        fits_local = False
        provider = "openai"
        provider_tag = "gpt-4o"

    class _OllamaSuggestion:
        fits_local = True
        provider = "ollama"
        provider_tag = "llama3:8b"

    fake_mod = _make_fake_forge(
        has_ollama=True,
        suggestions=[_OpenAISuggestion(), _OllamaSuggestion()],
    )
    monkeypatch.setitem(sys.modules, "l6e_forge.models.auto", fake_mod)

    from l6e.router import LocalRouter
    result = LocalRouter()._probe()
    assert result == "ollama/llama3:8b"
