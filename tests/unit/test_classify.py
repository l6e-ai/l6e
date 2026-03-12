"""Unit tests for _classify.py — PromptComplexityClassifier."""
from __future__ import annotations

import pytest

from l6e._types import PromptComplexity


@pytest.fixture
def classifier():
    from l6e._classify import PromptComplexityClassifier

    return PromptComplexityClassifier()


# ---------------------------------------------------------------------------
# Stage short-circuit — stage name overrides content analysis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["formatting", "extraction", "retrieval", "translation"])
def test_known_low_stage_returns_low(classifier, stage: str) -> None:
    prompt = "Analyze this carefully: step 1 compare and synthesize deeply across all dimensions."
    assert classifier.classify(prompt, stage=stage) == PromptComplexity.LOW


@pytest.mark.parametrize("stage", ["reasoning", "final_reasoning", "synthesis", "analysis"])
def test_known_high_stage_returns_high(classifier, stage: str) -> None:
    prompt = "Format this as a bullet list."
    assert classifier.classify(prompt, stage=stage) == PromptComplexity.HIGH


# ---------------------------------------------------------------------------
# Content-only classification (stage=None)
# ---------------------------------------------------------------------------


def test_single_verb_short_prompt_is_low(classifier) -> None:
    assert classifier.classify("Summarize the following text:", stage=None) == PromptComplexity.LOW


def test_extract_verb_is_low(classifier) -> None:
    result = classifier.classify("Extract all email addresses from this document.", stage=None)
    assert result == PromptComplexity.LOW


def test_format_verb_is_low(classifier) -> None:
    result = classifier.classify("Format the JSON output as a markdown table.", stage=None)
    assert result == PromptComplexity.LOW


def test_translate_verb_is_low(classifier) -> None:
    result = classifier.classify("Translate the following paragraph to French.", stage=None)
    assert result == PromptComplexity.LOW


def test_multi_step_reasoning_is_high(classifier) -> None:
    prompt = (
        "Step 1: analyze the market conditions. "
        "Step 2: compare the two strategies. "
        "Step 3: synthesize your findings and evaluate the risks."
    )
    assert classifier.classify(prompt, stage=None) == PromptComplexity.HIGH


def test_reasoning_keywords_push_high(classifier) -> None:
    # Multiple reasoning keywords + long enough to avoid length penalty
    prompt = (
        "Please analyze and critique the following argument carefully. "
        "Evaluate its logical soundness, compare it with alternative approaches, "
        "and synthesize your findings into a clear recommendation. "
        "Consider edge cases and assess the risks of each option before concluding."
    )
    assert classifier.classify(prompt, stage=None) == PromptComplexity.HIGH


def test_multiple_questions_push_high(classifier) -> None:
    # Many questions + reasoning keywords pushes well past threshold
    prompt = (
        "What are the root causes of this problem? How do they interact with each other? "
        "What would happen if we changed approach X? Why does outcome Y follow from decision Z? "
        "Please analyze the trade-offs and evaluate which path makes most sense."
    )
    assert classifier.classify(prompt, stage=None) == PromptComplexity.HIGH


def test_neutral_paragraph_is_medium(classifier) -> None:
    prompt = (
        "The quarterly report shows revenue increased by 12 percent. "
        "The main drivers were new customer acquisition and expansion in existing accounts."
    )
    assert classifier.classify(prompt, stage=None) == PromptComplexity.MEDIUM


def test_very_long_prompt_without_signals_leans_high(classifier) -> None:
    # Long prompts with no low-signals lean HIGH due to length alone
    prompt = "Consider the following information. " + ("The data shows various trends. " * 80)
    result = classifier.classify(prompt, stage=None)
    assert result in (PromptComplexity.MEDIUM, PromptComplexity.HIGH)


def test_short_empty_like_prompt_is_low(classifier) -> None:
    assert classifier.classify("List the items.", stage=None) == PromptComplexity.LOW


# ---------------------------------------------------------------------------
# Stage=None with unknown stage name falls back to content
# ---------------------------------------------------------------------------


def test_unknown_stage_falls_back_to_content(classifier) -> None:
    # "my_custom_stage" is not in any short-circuit set → uses content
    prompt = "Summarize the following text:"
    result = classifier.classify(prompt, stage="my_custom_stage")
    assert result == PromptComplexity.LOW


def test_unknown_stage_high_content_returns_high(classifier) -> None:
    prompt = "Step 1: analyze. Step 2: compare. Step 3: synthesize and evaluate critically."
    result = classifier.classify(prompt, stage="my_custom_stage")
    assert result == PromptComplexity.HIGH


# ---------------------------------------------------------------------------
# High placeholder density branch (line 81) — score -= 1
# ---------------------------------------------------------------------------


def test_high_placeholder_density_decrements_score(classifier) -> None:
    """Placeholder ratio > 0.15 should reduce score by 1.

    Prompt: 4 placeholders across ~8 words → ratio = 0.5, well above 0.15.
    Without the density penalty this neutral prompt would be MEDIUM;
    with it the score tips to LOW (short prompt -1, density -1, starts with
    non-low verb so no extra -2 → score = -2 → LOW).
    """
    # 8 words + 4 {placeholder} tokens → ratio 4/8 = 0.5 > 0.15
    prompt = "Do {action} and {target} for <user> then <result>"
    result = classifier.classify(prompt, stage=None)
    # Score: -1 (short) + -1 (density) = -2 → LOW
    assert result == PromptComplexity.LOW


def test_placeholder_density_below_threshold_does_not_decrement(classifier) -> None:
    """A prompt with sparse placeholders should NOT trigger the density penalty."""
    # Long enough to avoid length penalty, only 1 placeholder in ~50 words
    base = "Please carefully review the following document and provide a detailed assessment. "
    prompt = base * 3 + "Focus on the {key_aspect} most relevant to the task."
    result = classifier.classify(prompt, stage=None)
    # Without density penalty, neutral/medium content — should not be artificially LOW
    assert result in (PromptComplexity.MEDIUM, PromptComplexity.HIGH)
