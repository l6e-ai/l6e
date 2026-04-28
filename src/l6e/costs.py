"""LiteLLM-backed cost estimator."""
from __future__ import annotations

import logging
import re
import threading
import warnings
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

_logger = logging.getLogger(__name__)

# Tokens stripped from LiteLLM key strings before subset matching.
# Date tokens (≥5 consecutive digits) and "latest" are noise that would
# never appear in a caller-supplied model ID like "claude-4.6-sonnet-large".
_DATE_OR_LATEST_RE = re.compile(r"^\d{5,}$|^latest$")

# Recognise a numeric version token like ``4`` or ``4.6``/``4-6`` inside a key.
_VERSION_TOKEN_RE = re.compile(r"^\d+$")

# Cache of (normalized-key-tokens, original-key) pairs built once.
_LITELLM_BARE_KEYS: list[tuple[frozenset[str], str]] | None = None


def _build_bare_key_cache() -> list[tuple[frozenset[str], str]]:
    """Return (token_set, key) for every unqualified key in litellm's table.

    "Unqualified" means no provider prefix (no ``/``, ``@``, or ``:``).
    Token sets have date-noise and "latest" stripped so they can be matched
    as subsets against caller-supplied model IDs that carry tier suffixes
    (e.g. ``-medium-thinking``, ``-large``).
    """
    result: list[tuple[frozenset[str], str]] = []
    for key in litellm.model_cost:
        if "/" in key or "@" in key or ":" in key:
            continue
        tokens = frozenset(
            t
            for t in re.split(r"[-./: ]", key.lower())
            if t and not _DATE_OR_LATEST_RE.match(t)
        )
        if tokens:
            result.append((tokens, key))
    return result


def _extract_version(tokens: frozenset[str]) -> tuple[int, int]:
    """Extract a ``(major, minor)`` version from a token set.

    Tokens like ``4``, ``6`` or ``5``, ``3`` are interpreted as
    ``major.minor`` in the order they would appear in the original key.
    Missing minor defaults to 0 so single-version families still sort.
    Returns ``(0, 0)`` when no version tokens are present.
    """
    versions = sorted(int(t) for t in tokens if _VERSION_TOKEN_RE.match(t))
    if not versions:
        return (0, 0)
    if len(versions) == 1:
        return (versions[0], 0)
    return (versions[0], versions[1])


def _resolve_family_fallback(
    input_tokens: frozenset[str],
    bare_keys: list[tuple[frozenset[str], str]],
) -> str | None:
    """Find the newest strictly older same-family key after lookup failure.

    "Same-family" means every non-version token of the candidate appears in
    the input (so ``claude-opus-4-6`` matches input ``claude-opus-4.7``
    because ``{claude, opus}`` ⊆ ``{claude, opus, 4, 7}``). Among those
    candidates we pick the highest ``(major, minor)`` version that is still
    strictly lower than the caller's input version. This avoids returning the
    input model itself when LiteLLM has a price row but provider registration
    is broken, while still pricing a brand-new release by the most recent
    predecessor rather than punting to the blanket ``$0.01/1k`` fallback.
    """
    best_version: tuple[int, int] = (-1, -1)
    best_key: str | None = None
    input_version = _extract_version(input_tokens)
    input_non_version = frozenset(
        t for t in input_tokens if not _VERSION_TOKEN_RE.match(t)
    )
    for key_tokens, orig_key in bare_keys:
        key_non_version = frozenset(
            t for t in key_tokens if not _VERSION_TOKEN_RE.match(t)
        )
        if not key_non_version or not key_non_version.issubset(input_non_version):
            continue
        version = _extract_version(key_tokens)
        if version >= input_version:
            continue
        if version > best_version:
            best_version = version
            best_key = orig_key
    return best_key


def resolve_model_id(model_id: str) -> str | None:
    """Return the best matching LiteLLM model key for *model_id*, or ``None``.

    AI coding assistants often report verbose, vendor-internal model IDs that
    include tier or capability suffixes not present in LiteLLM's cost table
    (e.g. ``claude-4.6-sonnet-medium-thinking`` vs ``claude-sonnet-4-6``).
    This function normalises both sides to token sets and finds the longest
    LiteLLM key whose tokens are a complete subset of the input's tokens,
    preferring keys with fewer tokens on a tie (more specific matches win).

    The search is limited to unqualified keys (no provider prefix) to avoid
    accidentally resolving a vendor-specific price entry.
    """
    global _LITELLM_BARE_KEYS  # noqa: PLW0603
    if _LITELLM_BARE_KEYS is None:
        _LITELLM_BARE_KEYS = _build_bare_key_cache()

    input_tokens = frozenset(
        t for t in re.split(r"[-./: ]", model_id.lower()) if t
    )
    input_version_tokens = frozenset(
        t for t in input_tokens if _VERSION_TOKEN_RE.match(t)
    )

    best_score = -1
    best_n_tokens = 9999
    best_key: str | None = None

    for key_tokens, orig_key in _LITELLM_BARE_KEYS:
        if not key_tokens.issubset(input_tokens):
            continue
        key_version_tokens = frozenset(
            t for t in key_tokens if _VERSION_TOKEN_RE.match(t)
        )
        if key_version_tokens and input_version_tokens and \
            key_version_tokens != input_version_tokens:
            continue
        score = len(key_tokens)
        if score > best_score or (score == best_score and len(key_tokens) < best_n_tokens):
            best_score = score
            best_n_tokens = len(key_tokens)
            best_key = orig_key

    return best_key


@dataclass(frozen=True)
class CostEstimateMetadata:
    cost_usd: Decimal
    pricing_confidence: Literal["high", "low"]
    pricing_source: str
    warning: str | None
    model_pricing_known: bool
    resolved_model: str | None = None


class LiteLLMCostEstimator:
    """Estimates LLM call cost using litellm's model cost table.

    Callers supply token counts directly — this class does not tokenize.

    When a model is not in litellm's table, a fuzzy resolver attempts to find
    the closest unqualified LiteLLM key by token-subset matching — this handles
    vendor-internal model IDs that carry tier/capability suffixes unknown to
    LiteLLM (e.g. ``claude-4.6-sonnet-medium-thinking`` → ``claude-sonnet-4-6``).

    If resolution also fails, a warning is always emitted and, if
    ``fallback_cost_per_1k_tokens`` is non-zero, that rate is applied to
    ensure the gate can still fire.
    """

    def __init__(self, fallback_cost_per_1k_tokens: float = 0.01) -> None:
        self._fallback_cost_per_1k = fallback_cost_per_1k_tokens

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """Return estimated cost in USD.

        Returns 0.0 (or the configured fallback rate) for unknown models.
        Always emits a warning when the model is not recognised by litellm.
        """
        return self.estimate_with_metadata(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ).cost_usd

    def estimate_with_metadata(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        emit_warning: bool = True,
    ) -> CostEstimateMetadata:
        """Return estimated cost and confidence metadata."""
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return CostEstimateMetadata(
                cost_usd=Decimal(str(prompt_cost + completion_cost)),
                pricing_confidence="high",
                pricing_source="litellm_table",
                warning=None,
                model_pricing_known=True,
                resolved_model=None,
            )
        except Exception:
            pass

        resolved = resolve_model_id(model)
        if resolved is not None:
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=resolved,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                return CostEstimateMetadata(
                    cost_usd=Decimal(str(prompt_cost + completion_cost)),
                    pricing_confidence="high",
                    pricing_source="litellm_table_resolved",
                    warning=None,
                    model_pricing_known=True,
                    resolved_model=resolved,
                )
            except Exception:
                pass

        # Last-resort before the blanket fallback: price by the newest
        # known model of the same family. ``claude-opus-4.7`` → the
        # newest ``claude-opus-*`` we do know about. Estimates are
        # marked low-confidence with a distinct pricing_source so the
        # data-quality audit can exclude these sessions from calibration
        # and Layer 2 training until the real pricing lands.
        #
        # ``resolve_model_id`` above always initialises ``_LITELLM_BARE_KEYS``
        # before we reach this point, so the cache is guaranteed populated.
        assert _LITELLM_BARE_KEYS is not None
        family_resolved = _resolve_family_fallback(
            frozenset(
                t for t in re.split(r"[-./: ]", model.lower()) if t
            ),
            _LITELLM_BARE_KEYS,
        )
        if family_resolved is not None:
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=family_resolved,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                family_warning = (
                    f"l6e: unknown model '{model}' — priced from nearest known "
                    f"family member '{family_resolved}'. Estimate flagged "
                    "low-confidence; session is excluded from calibration and "
                    "Layer 2 training (see data-quality audit)."
                )
                if emit_warning:
                    warnings.warn(family_warning, stacklevel=2)
                return CostEstimateMetadata(
                    cost_usd=Decimal(str(prompt_cost + completion_cost)),
                    pricing_confidence="low",
                    pricing_source="family_version_fallback",
                    warning=family_warning,
                    model_pricing_known=False,
                    resolved_model=family_resolved,
                )
            except Exception:
                pass

        warning = (
            f"l6e: unknown model '{model}' — cost cannot be estimated from litellm's "
            f"table. Falling back to {self._fallback_cost_per_1k:.4f} USD/1k tokens. "
            "Set PipelinePolicy.unknown_model_cost_per_1k_tokens (or "
            "'unknown_model_cost_per_1k_tokens' in [policy] of your l6e TOML) to "
            "the correct rate for this model, or to 0.0 to disable enforcement for "
            "unknown models."
        )
        if emit_warning:
            warnings.warn(warning, stacklevel=2)
        if self._fallback_cost_per_1k > 0:
            total_tokens = prompt_tokens + completion_tokens
            return CostEstimateMetadata(
                cost_usd=Decimal(str(total_tokens / 1000.0 * self._fallback_cost_per_1k)),
                pricing_confidence="low",
                pricing_source="fallback_rate",
                warning=warning,
                model_pricing_known=False,
                resolved_model=None,
            )
        return CostEstimateMetadata(
            cost_usd=Decimal("0"),
            pricing_confidence="low",
            pricing_source="fallback_disabled",
            warning=warning,
            model_pricing_known=False,
            resolved_model=None,
        )


# ---------------------------------------------------------------------------
# Background refresh of litellm's model cost map
# ---------------------------------------------------------------------------

_refresh_started = False


def _merge_fetched_cost_map(fetched: dict[str, Any]) -> None:
    """Merge a freshly-fetched cost map into litellm's globals.

    Three updates happen together; missing any one of them leaves litellm
    in an inconsistent state where a model has a price entry but
    ``cost_per_token`` still raises:

    1. ``litellm.model_cost`` — the price dict consulted by ``cost_per_token``.
    2. ``litellm.<provider>_models`` (e.g. ``anthropic_models``) — provider-
       grouped sets that ``get_llm_provider`` consults to infer a provider
       when the caller passes a bare model id like ``claude-opus-4-7`` (no
       ``anthropic/`` prefix). litellm only populates these sets at import
       time via ``add_known_models``; updating ``model_cost`` post-import
       does not flow through to them. The symptom of forgetting this step
       is ``BadRequestError: LLM Provider NOT provided`` for a model that
       is in ``model_cost`` — see the L6E-86 diagnostic on 2026-04-25.
    3. ``_LITELLM_BARE_KEYS`` — our own fuzzy-resolver cache; invalidated
       so the next ``resolve_model_id`` call rebuilds it from the merged
       ``litellm.model_cost``.
    """
    global _LITELLM_BARE_KEYS  # noqa: PLW0603
    litellm.model_cost.update(fetched)
    try:
        litellm.add_known_models(fetched)
    except Exception:
        _logger.debug("l6e: add_known_models failed post-refresh", exc_info=True)
    _LITELLM_BARE_KEYS = None


def refresh_model_cost_map_async() -> None:
    """Fetch the remote model cost map in a background thread and merge it
    into ``litellm.model_cost``.

    Safe to call multiple times — only the first invocation spawns a thread.
    On any failure the local (bundled) map remains in effect.
    """
    global _refresh_started  # noqa: PLW0603
    if _refresh_started:
        return
    _refresh_started = True

    def _fetch() -> None:
        try:
            url = litellm.model_cost_map_url
            fetched = GetModelCostMap.fetch_remote_model_cost_map(url, timeout=10)
            backup_count = GetModelCostMap._get_backup_model_count()
            if GetModelCostMap.validate_model_cost_map(fetched, backup_count):
                _merge_fetched_cost_map(fetched)
                _logger.debug("l6e: refreshed litellm model cost map from %s", url)
            else:
                _logger.debug("l6e: remote model cost map failed validation, keeping local")
        except Exception:
            _logger.debug("l6e: background model cost map refresh failed", exc_info=True)

    threading.Thread(target=_fetch, daemon=True, name="l6e-cost-map-refresh").start()
