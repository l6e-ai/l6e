"""Universal adapter — re-export anchor for raw LiteLLM / OpenAI SDK users.

The primary integration surface is ``ctx.call()`` on ``PipelineContext``.
This module exists as the documentation and import anchor so Day 5 adapters
follow the same package pattern.

Usage::

    from l6e.pipeline import pipeline

    with pipeline("my-run", policy) as ctx:
        response = ctx.call(
            fn=litellm.completion,
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stage="greeting",
        )
"""
from __future__ import annotations

from l6e.pipeline import PipelineContext, pipeline

__all__ = ["PipelineContext", "pipeline"]
