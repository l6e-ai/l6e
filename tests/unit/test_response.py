"""Unit tests for _response.py — extract_token_usage()."""
from __future__ import annotations

from l6e._response import extract_token_usage

# ---------------------------------------------------------------------------
# OpenAI / LiteLLM style object
# ---------------------------------------------------------------------------


class _FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50


class _FakeResponse:
    usage = _FakeUsage()


def test_openai_style_object() -> None:
    assert extract_token_usage(_FakeResponse()) == (100, 50)


# ---------------------------------------------------------------------------
# Anthropic style object (input_tokens / output_tokens)
# ---------------------------------------------------------------------------


class _FakeAnthropicUsage:
    input_tokens = 80
    output_tokens = 40


class _FakeAnthropicResponse:
    usage = _FakeAnthropicUsage()


def test_anthropic_style_object() -> None:
    assert extract_token_usage(_FakeAnthropicResponse()) == (80, 40)


# ---------------------------------------------------------------------------
# Dict responses
# ---------------------------------------------------------------------------


def test_dict_openai_style() -> None:
    result = extract_token_usage({"usage": {"prompt_tokens": 200, "completion_tokens": 100}})
    assert result == (200, 100)


def test_dict_anthropic_style() -> None:
    assert extract_token_usage({"usage": {"input_tokens": 60, "output_tokens": 30}}) == (60, 30)


# ---------------------------------------------------------------------------
# Tiktoken fallback for plain string
# ---------------------------------------------------------------------------


def test_plain_string_returns_nonnegative_tuple() -> None:
    result = extract_token_usage("hello world response text")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] >= 0 and result[1] >= 0


# ---------------------------------------------------------------------------
# Completely unknown object → (0, 0)
# ---------------------------------------------------------------------------


def test_unknown_object_returns_zero_tuple() -> None:
    assert extract_token_usage(object()) == (0, 0)


# ---------------------------------------------------------------------------
# LangChain LLMResult style (path 2.5)
# ---------------------------------------------------------------------------


class _FakeLLMResult:
    llm_output = {"token_usage": {"prompt_tokens": 120, "completion_tokens": 60}}


def test_langchain_llm_result_token_usage() -> None:
    assert extract_token_usage(_FakeLLMResult()) == (120, 60)


class _FakeLLMResultUsageKey:
    """LLMResult with 'usage' key instead of 'token_usage'."""
    llm_output = {"usage": {"prompt_tokens": 70, "completion_tokens": 30}}


def test_langchain_llm_result_usage_key_fallback() -> None:
    assert extract_token_usage(_FakeLLMResultUsageKey()) == (70, 30)


class _FakeLLMResultZeroTokens:
    """Zero token counts must not be skipped."""
    llm_output = {"token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def test_langchain_llm_result_zero_token_counts() -> None:
    assert extract_token_usage(_FakeLLMResultZeroTokens()) == (0, 0)


# ---------------------------------------------------------------------------
# LangChain LLMResult — input_tokens/output_tokens fallback keys (lines 39, 42)
# ---------------------------------------------------------------------------


class _FakeLLMResultAnthropicKeys:
    """LLMResult with Anthropic-style input_tokens/output_tokens inside token_usage."""
    llm_output = {"token_usage": {"input_tokens": 55, "output_tokens": 25}}


def test_langchain_llm_result_input_output_tokens_fallback() -> None:
    """Lines 39 + 42: prompt_tokens absent → falls back to input_tokens/output_tokens."""
    assert extract_token_usage(_FakeLLMResultAnthropicKeys()) == (55, 25)


class _FakeLLMResultUsageAnthropicKeys:
    """LLMResult using 'usage' key with Anthropic-style token names."""
    llm_output = {"usage": {"input_tokens": 33, "output_tokens": 11}}


def test_langchain_llm_result_usage_key_anthropic_fallback() -> None:
    assert extract_token_usage(_FakeLLMResultUsageAnthropicKeys()) == (33, 11)


# ---------------------------------------------------------------------------
# tiktoken failure fallback (lines 70-71)
# ---------------------------------------------------------------------------


def test_plain_string_returns_zero_tuple_when_tiktoken_fails(monkeypatch) -> None:
    """Lines 70-71: if tiktoken raises during string fallback, return (0, 0)."""
    import l6e._response as response_mod

    def failing_get_encoding(name: str):
        raise RuntimeError("tiktoken unavailable")

    # Patch tiktoken inside the module's namespace
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", failing_get_encoding)

    result = response_mod.extract_token_usage("some text that triggers tiktoken path")
    assert result == (0, 0)
