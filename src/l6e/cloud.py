"""Cloud-sync configuration and ``/v1/authorize`` client for the SDK.

Opt-in surface for **Tier 3 (SDK cloud-sync)** in
``pivot-docs/cost-benchmark-margin-thesis/05-integration-architecture.md``.
A customer wires it via the ``cloud=`` kwarg on ``pipeline()`` (see L6E-73);
absent that kwarg, the SDK behaves exactly as it did before this module
existed.

This module owns three responsibilities:

1. ``CloudConfig`` — the user-facing configuration surface.
2. ``_post_authorize`` — the synchronous HTTP client for
   ``POST /v1/authorize``. Sibling to (and modeled directly on)
   ``mcp/src/l6e_mcp/core/remote_authorize.py``: returns ``None`` on every
   failure path so the caller can fall back to local enforcement, per
   L6E-41's iron rule.
3. ``_sanitize_authorize_response`` — defensive validation of the server
   envelope before any field is used to drive a ``GateDecision``. NaN /
   negative / missing-action / unknown-pressure-label all collapse to
   ``None`` so a poisoned response can't quietly corrupt downstream state.

Q2(a) decision (synchronous block with tight ``latency_deadline_ms``,
fail-open on exceed) is realized here: ``_post_authorize`` honors the
deadline as a local timeout floor and never raises into caller code.
A future fire-and-forget allow-path mode (Q2(b), tracked as a follow-up
ticket) would wrap this module's helpers, not replace them.
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

PrivacyTier = Literal["metadata", "embeddings", "hashed_prompts"]

# The only privacy tier wired up in M0. ``embeddings`` and
# ``hashed_prompts`` block on L6E-64 / L6E-68 — the customer-side
# embedder choice and default-tier decision. Until those land, accepting
# the strings silently would be a bug surface (operator thinks they have
# embeddings when they don't), so ``CloudConfig`` rejects them at
# construction with a pointer to the follow-up issues.
_SUPPORTED_PRIVACY_TIERS: frozenset[str] = frozenset({"metadata"})
_PRIVACY_TIER_FOLLOWUP_HINT = (
    "embeddings / hashed_prompts privacy tiers are not yet implemented; "
    "see L6E-64 (privacy default decision) and L6E-68 (client-side "
    "embedder design) for the blocking work."
)

# Server response envelope contracts. Kept in sync with the matching
# constants in ``mcp/src/l6e_mcp/server.py`` (``_VALID_SERVER_ACTIONS``
# etc.) — the two clients consume the same wire schema.
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
    - ``privacy_tier`` outside ``metadata`` raises ``NotImplementedError``
      pointing at L6E-64 / L6E-68.
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
# Shared sync httpx.Client — module-singleton with atexit cleanup. Mirrors
# the async-client pattern in mcp/src/l6e_mcp/core/remote_authorize.py.
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
    trigger fail-open. Mirrors ``mcp/src/l6e_mcp/server.py
    ._sanitize_server_authorize_response`` field-for-field — the two
    clients must agree on what "valid envelope" means or one of them is
    going to corrupt local state when the server drifts.
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
