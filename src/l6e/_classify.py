"""Structural prompt complexity classifier.

No LLM, no network. ~2ms. Feeds CallRecord.prompt_complexity, which the
Pro profiler's Task Classification Tuner uses to learn which stages tolerate
local models.
"""
from __future__ import annotations

import re

from l6e._types import PromptComplexity

# Stages whose names directly imply complexity — short-circuit content analysis.
_LOW_STAGES: frozenset[str] = frozenset[str]({
    "formatting", "format", "extraction", "extract",
    "retrieval", "retrieve", "translation", "translate", "listing",
})
_HIGH_STAGES: frozenset[str] = frozenset[str]({
    "reasoning", "final_reasoning", "synthesis", "synthesize",
    "analysis", "analyze", "evaluation", "evaluate",
})

# Content signals
_LOW_VERBS: frozenset[str] = frozenset[str]({
    "summarize", "extract", "format", "translate",
    "list", "transcribe", "convert", "reformat", "enumerate",
})
_HIGH_KEYWORDS: frozenset[str] = frozenset[str]({
    "analyze", "analyse", "compare", "evaluate", "synthesize",
    "critique", "assess", "contrast", "argue", "reason",
})
_STEP_PATTERN: re.Pattern[str] = re.compile(
    r"\bstep\s+\d+\b|\bfirst[,\s].{0,30}then\b|\bfirstly\b|\b\d+\.\s+\w",
    re.IGNORECASE,
)


class PromptComplexityClassifier:
    """Classify a prompt as LOW / MEDIUM / HIGH using structural heuristics only."""

    def classify(self, prompt: str, stage: str | None) -> PromptComplexity:
        # 1. Stage short-circuit
        if stage is not None:
            stage_lower = stage.lower()
            if stage_lower in _LOW_STAGES:
                return PromptComplexity.LOW
            if stage_lower in _HIGH_STAGES:
                return PromptComplexity.HIGH

        # 2. Score-based content analysis
        score = self._content_score(prompt)

        if score >= 2:
            return PromptComplexity.HIGH
        if score <= -2:
            return PromptComplexity.LOW
        return PromptComplexity.MEDIUM

    def _content_score(self, prompt: str) -> int:
        """Return a signed integer score. Positive → HIGH, negative → LOW."""
        score = 0
        lower = prompt.lower().strip()

        # --- Length signals ---
        if len(prompt) < 200:
            score -= 1
        elif len(prompt) > 2000:
            score += 1

        # --- Low-complexity signals ---

        # Prompt starts with a single low-complexity verb
        first_word = lower.split()[0].rstrip(".,!?") if lower.split() else ""
        if first_word in _LOW_VERBS:
            score -= 2

        # High template density: lots of {placeholders} or <tags>
        placeholder_count = len(re.findall(r"\{[^}]+\}|<[^>]+>", prompt))
        word_count = max(len(lower.split()), 1)
        if placeholder_count / word_count > 0.15:
            score -= 1

        # --- High-complexity signals ---

        # Reasoning / analytical keywords — each unique match contributes, capped at +2
        keyword_hits = sum(
            1 for kw in _HIGH_KEYWORDS if re.search(r"\b" + kw + r"\b", lower)
        )
        score += min(keyword_hits, 2)

        # Multi-step instructions
        if _STEP_PATTERN.search(prompt):
            score += 2

        # Multiple questions (>2 question marks)
        if prompt.count("?") > 2:
            score += 1

        return score
