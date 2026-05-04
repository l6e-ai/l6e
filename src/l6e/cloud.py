"""Cloud-sync configuration and ``/v1/authorize`` client for the SDK.

Opt-in surface for the cloud-sync integration tier. A customer wires it
via the ``cloud=`` kwarg on ``pipeline()``; absent that kwarg, the SDK
behaves exactly as it did before this module existed (purely local
enforcement, no network calls).

This module owns three responsibilities:

1. ``CloudConfig`` — the user-facing configuration surface.
2. ``_post_authorize`` — the synchronous HTTP client for
   ``POST /v1/authorize``. Returns ``None`` on every failure path
   (timeout, non-2xx, malformed JSON, network exception, missing API
   key, garbage envelope) so the caller can fall back to local
   enforcement. The gate fails open, always — that's the iron rule.
3. ``_sanitize_authorize_response`` — defensive validation of the server
   envelope before any field is used to drive a ``GateDecision``. NaN /
   negative / missing-action / unknown-pressure-label all collapse to
   ``None`` so a poisoned response can't quietly corrupt downstream state.

The current latency posture is synchronous block with a tight
``latency_deadline_ms`` and fail-open on exceed: ``_post_authorize``
honors the deadline as a local timeout floor and never raises into
caller code. A future fire-and-forget allow-path mode would wrap this
module's helpers rather than replacing them.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

PrivacyTier = Literal["metadata", "embeddings", "hashed_prompts"]

# Customer-supplied client-side embedder. Receives the SDK-extracted user
# prompts (list of strings, in message order) and returns a single
# ``list[float]`` to attach as ``request_embedding`` on the cloud
# authorize body. Bring-your-own — l6e does not prescribe a model. The
# customer owns pooling (mean / first-only / concat-then-embed) so they
# can match whatever their privacy-reviewed embedder produces.
Embedder = Callable[[list[str]], list[float]]

# Privacy tiers wired up today. ``hashed_prompts`` is reserved on the
# public type so opt-in flags don't break compatibility when L6E-98
# ships, but accepting it silently now would be a bug surface (operator
# thinks they have hashed prompts when they don't). ``CloudConfig``
# therefore rejects it at construction.
_SUPPORTED_PRIVACY_TIERS: frozenset[str] = frozenset({"metadata", "embeddings"})
_PRIVACY_TIER_FOLLOWUP_HINT = (
    "hashed_prompts privacy tier is not yet implemented and will raise "
    "at construction. Use privacy_tier='metadata' or "
    "privacy_tier='embeddings' until LSH/MinHash support ships (L6E-98)."
)

# Server-side cap on embedding dimensionality (mirrored from
# ``hosted-edge/src/relay/routers/authorize.py:_MAX_EMBEDDING_DIM``).
# Validating client-side keeps a single ``cloud_embedding_failed``
# fail-surface — the alternative is letting the server 400 and
# fail-opening through the ``cloud_authorize_5xx`` path, which conflates
# "embedder produced garbage" with "server is unhealthy" in operator
# dashboards. Iron-rule extension: client-side validation is the right
# place to catch bring-your-own-embedder misconfiguration.
_MAX_EMBEDDING_DIM = 4096

# Server response envelope contracts. The MCP client consumes the same
# wire schema, so the two clients must agree on what "valid envelope"
# means or one of them will corrupt local state when the server drifts.
_VALID_SERVER_ACTIONS: frozenset[str] = frozenset({"allow", "reroute", "halt"})
_VALID_BUDGET_PRESSURE: frozenset[str] = frozenset(
    {"low", "moderate", "high", "critical"}
)
_VALID_COLDSTART_SOURCES: frozenset[str] = frozenset(
    {"prior", "shrunk", "warm", "prior_unavailable"}
)


@dataclass(frozen=True)
class CloudConfig:
    """Opt-in cloud-sync configuration for ``pipeline()``.

    Construction is the first place a customer could misconfigure
    cloud-sync, so we validate aggressively here:

    - ``timeout_s`` and ``latency_deadline_ms`` must be finite and > 0.
    - ``privacy_tier="hashed_prompts"`` raises ``NotImplementedError``
      pending L6E-98.
    - ``privacy_tier="embeddings"`` requires ``embedder`` to be set;
      missing ``embedder`` raises ``ValueError`` at construction. This
      is a misconfig, not a runtime fail-open: a customer who declared
      "I want embeddings tier" but forgot the embedder would otherwise
      silently degrade to metadata on every call, defeating the purpose
      of the opt-in. Catch it loudly at load time.
    - ``api_key`` resolves at construction time: explicit kwarg wins,
      else ``os.environ["L6E_API_KEY"]``, else ``None``. A ``None`` key
      after resolution does *not* raise — it's logged at WARNING and
      the gate fails open immediately on every ``check()`` (so customer
      pipelines that briefly forget the env var degrade to local-only
      enforcement, not crash).
    """

    base_url: str = "https://relay.l6e.ai"
    api_key: str | None = None
    timeout_s: float = 0.250
    latency_deadline_ms: int = 250
    privacy_tier: PrivacyTier = "metadata"
    # Customer-supplied client-side embedder. Required when
    # ``privacy_tier="embeddings"``; ignored otherwise. Setting it under
    # ``metadata`` tier is allowed (and silent) so customers can flip
    # tiers via env without conditionally removing the embedder.
    embedder: Embedder | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("CloudConfig.base_url must be a non-empty string")
        timeout = float(self.timeout_s)
        if math.isnan(timeout) or math.isinf(timeout) or timeout <= 0:
            raise ValueError(
                "CloudConfig.timeout_s must be a finite positive number "
                f"(got {self.timeout_s!r})"
            )
        if not isinstance(self.latency_deadline_ms, int) or isinstance(
            self.latency_deadline_ms, bool,
        ) or self.latency_deadline_ms <= 0:
            raise ValueError(
                "CloudConfig.latency_deadline_ms must be a positive int "
                f"(got {self.latency_deadline_ms!r})"
            )
        if self.privacy_tier not in _SUPPORTED_PRIVACY_TIERS:
            raise NotImplementedError(
                f"CloudConfig.privacy_tier={self.privacy_tier!r} is not yet supported. "
                + _PRIVACY_TIER_FOLLOWUP_HINT
            )
        if self.privacy_tier == "embeddings" and self.embedder is None:
            raise ValueError(
                "CloudConfig.privacy_tier='embeddings' requires an "
                "embedder. Pass embedder=<callable[[list[str]], list[float]]> "
                "or set privacy_tier='metadata'. Bring-your-own; l6e does "
                "not prescribe an embedding model."
            )
        if self.embedder is not None and not callable(self.embedder):
            raise ValueError(
                "CloudConfig.embedder must be callable "
                f"(got {type(self.embedder).__name__})"
            )
        # ``api_key`` resolution: explicit > env > None. We mutate via
        # ``object.__setattr__`` because the dataclass is frozen.
        if self.api_key is None:
            env_key = os.environ.get("L6E_API_KEY")
            if env_key:
                object.__setattr__(self, "api_key", env_key)
        if not self.api_key:
            logger.warning(
                "cloud_config_missing_api_key",
                extra={"base_url": self.base_url},
            )

    @property
    def effective_timeout_s(self) -> float:
        """The HTTP timeout used per call: tighter of ``timeout_s`` and
        ``latency_deadline_ms``. Mirrors the MCP client's deadline-honored
        path so server-slow degradation has a hard local ceiling."""
        deadline_seconds = self.latency_deadline_ms / 1000.0
        return min(float(self.timeout_s), deadline_seconds)


# ---------------------------------------------------------------------------
# Shared sync httpx.Client — module-singleton with atexit cleanup. Reusing
# a single client across calls eliminates per-request DNS and TLS overhead.
# ---------------------------------------------------------------------------

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_sync_client(timeout: float) -> httpx.Client:
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        _client = httpx.Client(timeout=timeout)
        return _client


def _shutdown_client() -> None:
    global _client  # noqa: PLW0603
    with _client_lock:
        to_close = _client
        _client = None
    if to_close is not None:
        # atexit must not raise; a broken client tear-down is strictly
        # preferable to crashing the interpreter on exit.
        with contextlib.suppress(Exception):
            to_close.close()


atexit.register(_shutdown_client)


def _reset_client() -> None:
    """Clear the cached client. Used by tests for isolation."""
    global _client  # noqa: PLW0603
    with _client_lock:
        if _client is not None:
            with contextlib.suppress(Exception):
                _client.close()
        _client = None


# ---------------------------------------------------------------------------
# Embedder invocation — fail-open to metadata-tier behavior on any failure.
# ---------------------------------------------------------------------------


def _validate_embedding(value: object) -> list[float] | None:
    """Defensive shape check matching the server-side
    ``_require_embedding`` contract (``hosted-edge/.../authorize.py``).

    Returns the coerced ``list[float]`` on success or ``None`` on any
    structural problem. Catching client-side keeps a single
    ``cloud_embedding_failed`` fail-surface — the alternative is letting
    the server 400 and routing through ``cloud_authorize_5xx``, which
    conflates "embedder produced garbage" with "server is unhealthy" in
    operator dashboards.

    Rejects: non-list, empty list, > ``_MAX_EMBEDDING_DIM`` dims,
    elements that aren't ``Real`` (or are ``bool``, since ``True/False``
    silently coerce to 1/0 and mask wire bugs), NaN, inf.
    """
    if not isinstance(value, list) or not value:
        return None
    if len(value) > _MAX_EMBEDDING_DIM:
        return None
    coerced: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            return None
        fitem = float(item)
        if math.isnan(fitem) or math.isinf(fitem):
            return None
        coerced.append(fitem)
    return coerced


def _safe_embed(
    embedder: Embedder, prompts: list[str],
) -> list[float] | None:
    """Invoke a customer-supplied embedder and validate the result.

    Returns the validated ``list[float]`` on success, or ``None`` on any
    failure mode (embedder raised, returned non-list, returned empty,
    too many dims, NaN / inf, non-numeric elements). Never raises into
    caller code — the iron-rule extension for the embeddings tier:
    any embedder failure collapses to metadata-tier behavior (the
    ``request_embedding`` field is omitted from the cloud body), the
    cloud call still drives the decision.

    A single stable log key — ``cloud_embedding_failed`` — fires on every
    failure. Operators pivot dashboards on this one tag rather than on a
    family of subkeys, mirroring how ``_post_authorize`` collapses many
    network failure modes onto a small set of greppable strings.
    """
    try:
        result = embedder(prompts)
    except Exception:
        logger.warning("cloud_embedding_failed", exc_info=True)
        return None
    validated = _validate_embedding(result)
    if validated is None:
        # Distinguish "embedder ran but returned garbage" from "embedder
        # raised" via the structured ``reason`` extra without a separate
        # log key. The two failure modes route through the same
        # fail-open path so they share the same operator surface.
        logger.warning(
            "cloud_embedding_failed",
            extra={"reason": "invalid_embedding_shape"},
        )
        return None
    return validated


# ---------------------------------------------------------------------------
# Response sanitizer — defensive, returns None on any malformed envelope.
# ---------------------------------------------------------------------------


def _finite_non_negative(value: object) -> float | None:
    """Coerce a server-supplied numeric to a sane non-negative float.

    Returns ``None`` for NaN, inf, negative, non-numeric, or ``bool``
    (since ``bool`` is an ``int`` subclass and silent True→1 promotion
    would mask a wire bug).
    """
    if isinstance(value, bool) or value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    fvalue = float(value)
    if math.isnan(fvalue) or math.isinf(fvalue) or fvalue < 0:
        return None
    return fvalue


def _sanitize_authorize_response(resp: object) -> dict[str, Any] | None:
    """Validate a ``/v1/authorize`` response envelope.

    Returns the dict on success or ``None`` on any garbage that should
    trigger fail-open. The validation rules match the MCP client's
    sanitizer field-for-field: the two clients must agree on what
    "valid envelope" means or one of them will corrupt local state when
    the server drifts.
    """
    if not isinstance(resp, dict):
        return None
    if resp.get("action") not in _VALID_SERVER_ACTIONS:
        return None
    if _finite_non_negative(resp.get("calibrated_cost_usd")) is None:
        return None
    if _finite_non_negative(resp.get("remaining_usd")) is None:
        return None
    if resp.get("budget_pressure") not in _VALID_BUDGET_PRESSURE:
        return None
    factor_raw = resp.get("calibration_factor")
    if factor_raw is not None and _finite_non_negative(factor_raw) is None:
        return None
    coldstart_source = resp.get("coldstart_source")
    if (
        coldstart_source is not None
        and coldstart_source not in _VALID_COLDSTART_SOURCES
    ):
        return None
    # Margin-tier prediction fields are optional but if present must be sane.
    for key in ("predicted_cost_mean_usd", "predicted_cost_p95_usd"):
        v = resp.get(key)
        if v is not None and _finite_non_negative(v) is None:
            return None
    return resp


# ---------------------------------------------------------------------------
# The HTTP call.
# ---------------------------------------------------------------------------


def _post_authorize(
    cfg: CloudConfig,
    body: dict[str, Any],
) -> dict[str, Any] | None:
    """POST to ``{cfg.base_url}/v1/authorize``. Returns sanitized dict or None.

    Failure modes mapped to ``None`` (caller falls back to local gate):

    - missing API key (config-time) — logged at WARNING by ``CloudConfig``,
      we just bail here with a stable structured log.
    - ``httpx.TimeoutException`` (deadline-honored or server-slow).
    - non-200 HTTP status.
    - malformed JSON body on a 200 response.
    - any other ``Exception`` (network, DNS, TLS, etc.).
    - server response that fails ``_sanitize_authorize_response``.

    Each path emits a stable, operator-greppable log key:
    ``cloud_authorize_no_api_key``, ``cloud_authorize_timeout``,
    ``cloud_authorize_5xx``, ``cloud_authorize_bad_json``,
    ``cloud_authorize_invalid_envelope``, ``cloud_authorize_failed``.

    Never raises into caller code. The iron rule is uncompromising here.
    """
    if not cfg.api_key:
        logger.warning(
            "cloud_authorize_no_api_key",
            extra={"base_url": cfg.base_url},
        )
        return None

    url = f"{cfg.base_url.rstrip('/')}/v1/authorize"
    effective_timeout = cfg.effective_timeout_s

    try:
        client = _get_sync_client(effective_timeout)
    except Exception:
        # Defensive: if even client construction fails (broken httpx
        # install, OOM, etc.), fail-open. There's no recovery path here
        # except local enforcement.
        logger.warning("cloud_authorize_failed", exc_info=True)
        return None

    try:
        resp = client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=effective_timeout,
        )
    except httpx.TimeoutException:
        logger.warning(
            "cloud_authorize_timeout",
            extra={
                "url": url,
                "effective_timeout": effective_timeout,
            },
        )
        return None
    except Exception:
        logger.warning("cloud_authorize_failed", exc_info=True)
        return None

    if resp.status_code != 200:
        logger.warning(
            "cloud_authorize_5xx",
            extra={"status": resp.status_code, "body": resp.text[:200]},
        )
        return None

    try:
        parsed = resp.json()
    except Exception:
        logger.warning("cloud_authorize_bad_json", exc_info=True)
        return None

    sanitized = _sanitize_authorize_response(parsed)
    if sanitized is None:
        logger.warning(
            "cloud_authorize_invalid_envelope",
            extra={"keys": sorted(parsed.keys()) if isinstance(parsed, dict) else None},
        )
        return None
    return sanitized
