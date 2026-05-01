"""Tests for ``RemoteConstraintGate`` and the cloud-sync wiring (L6E-73).

Covers six surfaces:

1. ``CloudConfig`` validation — privacy_tier scope (Q3a), api_key
   resolution, finite/positive timeouts, missing key fail-open posture.
2. ``_sanitize_authorize_response`` — the sibling-of-MCP envelope guard
   that rejects NaN / negative / missing-action server responses.
3. ``_post_authorize`` — every fail-open path (timeout, 5xx, bad JSON,
   network exception, no api_key, garbage envelope) collapses to
   ``None``, never raises, logs at WARNING.
4. ``_apply_cloud_response`` — pure mapping function. Allow / reroute
   / halt land on ``GateDecision`` with the additive optional fields
   populated. Reroute target falls back to local router when the
   server omits ``routed_model_suggestion``.
5. ``RemoteConstraintGate.check()`` — end-to-end. Cloud allow/reroute/halt
   round-trip into ``GateDecision``; every cloud failure mode falls
   through to the inner local gate decorated with
   ``calibration_source='local_fallback'`` and a ``fail_open:cloud_*``
   reason. Identity kwargs (``user_id`` / ``tenant_id`` / ``cohort_hint``)
   reach the request body verbatim.
6. ``pipeline()`` factory — passing ``cloud=`` wires
   ``RemoteConstraintGate``; absent ``cloud=`` wires plain
   ``ConstraintGate``. Q1(a) factor caching: server-supplied
   ``calibration_factor`` is cached on ``GateDecision`` and re-applied
   to ``CallRecord.cost_usd`` at record-time.

Mirrors the iron-rule tests in ``mcp/tests/test_fail_open.py``. If a
new failure mode appears here, add it there too — and vice versa.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from l6e._types import (
    PipelinePolicy,
)
from l6e.cloud import (
    CloudConfig,
    _post_authorize,
    _reset_client,
    _sanitize_authorize_response,
)
from l6e.gate import (
    ConstraintGate,
    RemoteConstraintGate,
    _apply_cloud_response,
)
from l6e.pipeline import pipeline
from tests.conftest import FakeCostEstimator, FakeRouter, FakeStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cloud_client() -> Any:
    _reset_client()
    yield
    _reset_client()


@pytest.fixture
def _cloud_cfg(monkeypatch: pytest.MonkeyPatch) -> CloudConfig:
    monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")
    return CloudConfig(base_url="https://relay.l6e.ai")


_VALID_RESP = {
    "action": "allow",
    "calibrated_cost_usd": 0.05,
    "raw_cost_usd": 0.02,
    "calibration_factor": 2.5,
    "calibration_source": "personal",
    "remaining_usd": 4.95,
    "budget_pressure": "low",
    "gate_reason": "allow",
}


def _mock_client_returning(resp_json: dict, status: int = 200) -> Any:
    """Build a patch context that hands back a fake httpx.Client whose
    ``.post`` returns a response with the given JSON / status."""
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = resp_json
    mock_resp.text = ""
    mock_client = MagicMock()
    mock_client.post = MagicMock(return_value=mock_resp)
    return patch("l6e.cloud._get_sync_client", return_value=mock_client)


def _mock_client_raising(exc: Exception) -> Any:
    mock_client = MagicMock()
    mock_client.post = MagicMock(side_effect=exc)
    return patch("l6e.cloud._get_sync_client", return_value=mock_client)


# ===========================================================================
# 1. CloudConfig validation
# ===========================================================================


class TestCloudConfigValidation:
    def test_minimum_config_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")
        cfg = CloudConfig(base_url="https://relay.l6e.ai")
        assert cfg.api_key == "sk-l6e-test"
        assert cfg.timeout_s == 0.250
        assert cfg.latency_deadline_ms == 250
        assert cfg.privacy_tier == "metadata"

    def test_explicit_api_key_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("L6E_API_KEY", "sk-from-env")
        cfg = CloudConfig(api_key="sk-explicit")
        assert cfg.api_key == "sk-explicit"

    def test_missing_api_key_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Iron rule: a missing API key must not raise at construction.
        A pipeline that briefly forgets the env var should degrade to
        local-only enforcement, not crash."""
        monkeypatch.delenv("L6E_API_KEY", raising=False)
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        cfg = CloudConfig(base_url="https://relay.l6e.ai")
        assert cfg.api_key is None
        assert any(
            r.getMessage() == "cloud_config_missing_api_key"
            for r in caplog.records
        )

    def test_embeddings_tier_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="L6E-64"):
            CloudConfig(api_key="sk", privacy_tier="embeddings")  # type: ignore[arg-type]

    def test_hashed_prompts_tier_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="L6E-64"):
            CloudConfig(api_key="sk", privacy_tier="hashed_prompts")  # type: ignore[arg-type]

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            CloudConfig(api_key="sk", timeout_s=0)

    def test_negative_latency_deadline_rejected(self) -> None:
        with pytest.raises(ValueError, match="latency_deadline_ms"):
            CloudConfig(api_key="sk", latency_deadline_ms=-1)

    def test_empty_base_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            CloudConfig(api_key="sk", base_url="")

    def test_effective_timeout_honors_tighter_of_two(self) -> None:
        cfg = CloudConfig(api_key="sk", timeout_s=1.0, latency_deadline_ms=50)
        assert cfg.effective_timeout_s == pytest.approx(0.05)

        cfg2 = CloudConfig(api_key="sk", timeout_s=0.1, latency_deadline_ms=5000)
        assert cfg2.effective_timeout_s == pytest.approx(0.1)


# ===========================================================================
# 2. _sanitize_authorize_response
# ===========================================================================


class TestSanitizeAuthorizeResponse:
    def test_accepts_valid(self) -> None:
        assert _sanitize_authorize_response(_VALID_RESP) is not None

    def test_rejects_missing_action(self) -> None:
        bad = {**_VALID_RESP}
        del bad["action"]
        assert _sanitize_authorize_response(bad) is None

    def test_rejects_invalid_action(self) -> None:
        assert _sanitize_authorize_response({**_VALID_RESP, "action": "maybe"}) is None

    def test_rejects_nan_calibrated_cost(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "calibrated_cost_usd": float("nan")},
            )
            is None
        )

    def test_rejects_inf_calibrated_cost(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "calibrated_cost_usd": float("inf")},
            )
            is None
        )

    def test_rejects_negative_calibrated_cost(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "calibrated_cost_usd": -0.5},
            )
            is None
        )

    def test_rejects_invalid_pressure_label(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "budget_pressure": "nuclear"},
            )
            is None
        )

    def test_rejects_nan_calibration_factor(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "calibration_factor": float("nan")},
            )
            is None
        )

    def test_allows_missing_calibration_factor(self) -> None:
        resp = {**_VALID_RESP}
        del resp["calibration_factor"]
        assert _sanitize_authorize_response(resp) is not None

    def test_rejects_unknown_coldstart_source(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "coldstart_source": "lukewarm"},
            )
            is None
        )

    def test_rejects_negative_predicted_cost(self) -> None:
        assert (
            _sanitize_authorize_response(
                {**_VALID_RESP, "predicted_cost_mean_usd": -1.0},
            )
            is None
        )

    def test_rejects_non_dict(self) -> None:
        assert _sanitize_authorize_response([1, 2, 3]) is None  # type: ignore[arg-type]
        assert _sanitize_authorize_response("nope") is None  # type: ignore[arg-type]


# ===========================================================================
# 3. _post_authorize fail-open matrix
# ===========================================================================


class TestPostAuthorizeFailOpen:
    def test_no_api_key_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("L6E_API_KEY", raising=False)
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        cfg = CloudConfig(base_url="https://relay.l6e.ai")
        assert _post_authorize(cfg, {"session_id": "s"}) is None
        assert any(
            r.getMessage() == "cloud_authorize_no_api_key" for r in caplog.records
        )

    def test_timeout_returns_none(
        self, _cloud_cfg: CloudConfig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        with _mock_client_raising(httpx.TimeoutException("deadline exceeded")):
            assert _post_authorize(_cloud_cfg, {"session_id": "s"}) is None
        rec = next(
            r for r in caplog.records if r.getMessage() == "cloud_authorize_timeout"
        )
        assert rec.levelno == logging.WARNING

    def test_5xx_returns_none(
        self, _cloud_cfg: CloudConfig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        with _mock_client_returning(_VALID_RESP, status=503) as _:
            assert _post_authorize(_cloud_cfg, {"session_id": "s"}) is None
        assert any(
            r.getMessage() == "cloud_authorize_5xx" for r in caplog.records
        )

    def test_bad_json_returns_none(
        self, _cloud_cfg: CloudConfig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("Expecting value")
        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_resp)

        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        with patch("l6e.cloud._get_sync_client", return_value=mock_client):
            assert _post_authorize(_cloud_cfg, {"session_id": "s"}) is None
        assert any(
            r.getMessage() == "cloud_authorize_bad_json" for r in caplog.records
        )

    def test_network_exception_returns_none(
        self, _cloud_cfg: CloudConfig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        with _mock_client_raising(RuntimeError("dns_blackhole")):
            assert _post_authorize(_cloud_cfg, {"session_id": "s"}) is None
        assert any(
            r.getMessage() == "cloud_authorize_failed" for r in caplog.records
        )

    def test_garbage_envelope_returns_none(
        self, _cloud_cfg: CloudConfig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        garbage = {**_VALID_RESP, "calibrated_cost_usd": float("nan")}
        caplog.set_level(logging.WARNING, logger="l6e.cloud")
        with _mock_client_returning(garbage):
            assert _post_authorize(_cloud_cfg, {"session_id": "s"}) is None
        assert any(
            r.getMessage() == "cloud_authorize_invalid_envelope"
            for r in caplog.records
        )

    def test_happy_path_returns_sanitized_dict(
        self, _cloud_cfg: CloudConfig,
    ) -> None:
        with _mock_client_returning(_VALID_RESP):
            result = _post_authorize(_cloud_cfg, {"session_id": "s"})
        assert result is not None
        assert result["action"] == "allow"

    def test_latency_deadline_tightens_request_timeout(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """L6E-41 scenario 2 (cloud slow): the local timeout floor must
        match ``latency_deadline_ms`` regardless of how high
        ``timeout_s`` is set."""
        monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")
        cfg = CloudConfig(timeout_s=10.0, latency_deadline_ms=50)
        with _mock_client_returning(_VALID_RESP) as patched:
            _post_authorize(cfg, {"session_id": "s"})
        # _get_sync_client(50ms) called once; client.post received timeout=0.05
        post_kwargs = patched.return_value.post.call_args.kwargs
        assert post_kwargs["timeout"] == pytest.approx(0.05)


# ===========================================================================
# 4. _apply_cloud_response — pure mapping
# ===========================================================================


class _StaticRouter:
    def __init__(self, model: str | None = "ollama/qwen2.5:7b") -> None:
        self._m = model

    def best_local_model(self) -> str | None:
        return self._m


class TestApplyCloudResponse:
    def test_allow_populates_optional_fields(self) -> None:
        resp = {
            **_VALID_RESP,
            "action": "allow",
            "predicted_cost_mean_usd": 0.04,
            "predicted_cost_p95_usd": 0.10,
            "policy_id_applied": "margin-default-v0",
        }
        d = _apply_cloud_response(
            response=resp, model="claude-sonnet-4", local_router=_StaticRouter(),
        )
        assert d.action == "allow"
        assert d.target_model == "claude-sonnet-4"
        assert d.calibration_source == "personal"
        assert d.calibration_factor == Decimal("2.5")
        assert d.predicted_cost_mean_usd == Decimal("0.04")
        assert d.predicted_cost_p95_usd == Decimal("0.10")
        assert d.policy_id_applied == "margin-default-v0"

    def test_reroute_uses_server_suggestion_when_present(self) -> None:
        resp = {
            **_VALID_RESP,
            "action": "reroute",
            "routed_model_suggestion": "ollama/qwen2.5:7b-cheap",
        }
        d = _apply_cloud_response(
            response=resp, model="claude-sonnet-4", local_router=_StaticRouter("OOPS"),
        )
        assert d.action == "reroute"
        assert d.target_model == "ollama/qwen2.5:7b-cheap"

    def test_reroute_falls_back_to_local_router(self) -> None:
        resp = {**_VALID_RESP, "action": "reroute"}
        d = _apply_cloud_response(
            response=resp,
            model="claude-sonnet-4",
            local_router=_StaticRouter("ollama/llama3:8b"),
        )
        assert d.action == "reroute"
        assert d.target_model == "ollama/llama3:8b"

    def test_reroute_with_no_target_halts_cleanly(self) -> None:
        resp = {**_VALID_RESP, "action": "reroute"}
        d = _apply_cloud_response(
            response=resp,
            model="claude-sonnet-4",
            local_router=_StaticRouter(None),
        )
        assert d.action == "halt"
        assert d.target_model == "claude-sonnet-4"
        assert "no_reroute_target" in d.reason

    def test_halt_keeps_requested_model_in_target(self) -> None:
        resp = {**_VALID_RESP, "action": "halt"}
        d = _apply_cloud_response(
            response=resp, model="claude-sonnet-4", local_router=_StaticRouter(),
        )
        assert d.action == "halt"
        assert d.target_model == "claude-sonnet-4"


# ===========================================================================
# 5. RemoteConstraintGate end-to-end + identity plumbing
# ===========================================================================


def _make_remote_gate(
    cloud_cfg: CloudConfig, *, budget: float = 5.0,
) -> RemoteConstraintGate:
    policy = PipelinePolicy(budget=budget)
    return RemoteConstraintGate(
        policy=policy, router=FakeRouter(), cloud=cloud_cfg,
    )


class TestRemoteConstraintGateHappyPath:
    def test_allow_round_trips(self, _cloud_cfg: CloudConfig) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning(_VALID_RESP):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "allow"
        assert d.calibration_source == "personal"
        assert d.calibration_factor == Decimal("2.5")

    def test_reroute_round_trips(self, _cloud_cfg: CloudConfig) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        resp = {
            **_VALID_RESP,
            "action": "reroute",
            "routed_model_suggestion": "ollama/qwen2.5:7b",
        }
        with _mock_client_returning(resp):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "reroute"
        assert d.target_model == "ollama/qwen2.5:7b"

    def test_halt_round_trips(self, _cloud_cfg: CloudConfig) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning({**_VALID_RESP, "action": "halt"}):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "halt"


class TestRemoteConstraintGateFailOpen:
    """Every cloud failure path must fall through to the local gate
    decorated with ``calibration_source='local_fallback'`` and a
    ``fail_open:cloud_*`` reason. Iron rule, L6E-41."""

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.TimeoutException("deadline"),
            RuntimeError("network_dead"),
        ],
    )
    def test_network_failure_falls_through(
        self, _cloud_cfg: CloudConfig, exc: Exception,
    ) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_raising(exc):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "allow"  # local gate, budget healthy
        assert d.calibration_source == "local_fallback"
        assert d.reason.startswith("fail_open:cloud_")

    def test_5xx_falls_through(self, _cloud_cfg: CloudConfig) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning(_VALID_RESP, status=503):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "allow"
        assert d.calibration_source == "local_fallback"

    def test_garbage_envelope_falls_through(
        self, _cloud_cfg: CloudConfig,
    ) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning(
            {**_VALID_RESP, "calibrated_cost_usd": -1.0},
        ):
            d = gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        assert d.action == "allow"
        assert d.calibration_source == "local_fallback"

    def test_no_api_key_falls_through(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("L6E_API_KEY", raising=False)
        cfg = CloudConfig()
        gate = _make_remote_gate(cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))

        # No HTTP mock — _post_authorize bails before the request when
        # api_key is missing. Iron rule: must not raise.
        d = gate.check(
            store,
            model="claude-sonnet-4",
            estimated_cost=Decimal("0.02"),
            stage="planning",
            complexity=None,
        )
        assert d.action == "allow"
        assert d.calibration_source == "local_fallback"


class TestIdentityKwargsRoundTripIntoBody:
    """L6E-73 acceptance #2: ``ctx.call(user_id='u_7')`` must produce a
    request body with ``user_id='u_7'`` so the server marks the row as
    ``is_margin_request=True``. Verified at the gate-body layer; the
    server-side flip is verified in the hosted-edge test suite."""

    def test_user_id_reaches_body_verbatim(
        self, _cloud_cfg: CloudConfig,
    ) -> None:
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning(_VALID_RESP) as patched:
            gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
                user_id="u_7",
                tenant_id="acme",
                cohort_hint="paid",
            )
        body = patched.return_value.post.call_args.kwargs["json"]
        assert body["user_id"] == "u_7"
        assert body["tenant_id"] == "acme"
        assert body["cohort_hint"] == "paid"
        assert body["model"] == "claude-sonnet-4"
        assert body["session_id"] == "fake-run-id"
        assert body["client"] == "l6e_sdk"
        # latency_deadline_ms forwarded so the server can honor the
        # iron-rule "cloud slow → treat as down" path.
        assert body["latency_deadline_ms"] == _cloud_cfg.latency_deadline_ms

    def test_no_identity_kwargs_keeps_them_off_the_wire(
        self, _cloud_cfg: CloudConfig,
    ) -> None:
        """Non-Margin SDK callers must continue to land in
        ``authorize_events`` as non-Margin rows (no identity fields →
        ``is_margin_request=False`` server-side)."""
        gate = _make_remote_gate(_cloud_cfg)
        store = FakeStore(budget=5.0, spent_amount=Decimal("0.0"))
        with _mock_client_returning(_VALID_RESP) as patched:
            gate.check(
                store,
                model="claude-sonnet-4",
                estimated_cost=Decimal("0.02"),
                stage="planning",
                complexity=None,
            )
        body = patched.return_value.post.call_args.kwargs["json"]
        assert "user_id" not in body
        assert "tenant_id" not in body
        assert "cohort_hint" not in body


# ===========================================================================
# 6. pipeline() factory + Q1(a) calibration_factor flow into CallRecord
# ===========================================================================


class _StubResponse:
    """A canned LLM response that ``extract_token_usage`` can read."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage = MagicMock()
        self.usage.prompt_tokens = prompt_tokens
        self.usage.completion_tokens = completion_tokens


class TestPipelineFactoryWiring:
    def test_no_cloud_uses_local_constraint_gate(self, tmp_path: Any) -> None:
        ctx = pipeline(
            policy=PipelinePolicy(budget=5.0),
            log_path=tmp_path / "runs.jsonl",
            router=FakeRouter(),
        )
        # Whitebox: the gate attribute should be the OSS class, not Remote.
        assert isinstance(ctx._gate, ConstraintGate)
        assert not isinstance(ctx._gate, RemoteConstraintGate)

    def test_cloud_kwarg_wires_remote_constraint_gate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")
        ctx = pipeline(
            policy=PipelinePolicy(budget=5.0),
            cloud=CloudConfig(),
            log_path=tmp_path / "runs.jsonl",
            router=FakeRouter(),
        )
        assert isinstance(ctx._gate, RemoteConstraintGate)


class TestCalibrationFactorFlowQ1a:
    """L6E-73 Q1(a): the cloud's ``calibration_factor`` is cached on the
    advise-time ``GateDecision`` and re-applied at record-time so
    ``CallRecord.cost_usd`` is "calibrated actual" cost — local
    estimator output × per-user multiplier."""

    def test_factor_multiplies_local_cost_at_record_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")
        # Stub the local cost estimator to a known value.
        from l6e.pipeline import PipelineContext

        ctx = pipeline(
            policy=PipelinePolicy(budget=5.0),
            cloud=CloudConfig(),
            log_path=tmp_path / "runs.jsonl",
            router=FakeRouter(),
        )
        # Replace the estimator with a deterministic fake. This is the
        # post-call estimator — what ``record()`` uses to convert
        # actual token counts into cost.
        ctx._estimator = FakeCostEstimator(cost=Decimal("0.04"))  # type: ignore[assignment]

        # Simulate ``call()`` invoking ``record()`` with the
        # calibration_factor that ``advise()`` cached on the decision.
        # We sidestep the full call path so we can pin the exact
        # behavior of ``record()``.
        record = ctx.record(
            model_requested="claude-sonnet-4",
            model_used="claude-sonnet-4",
            response=_StubResponse(prompt_tokens=1000, completion_tokens=500),
            elapsed_ms=42.0,
            calibration_factor=Decimal("2.5"),
        )
        # 0.04 (local estimate) × 2.5 (server factor) = 0.10
        assert record.cost_usd == Decimal("0.04") * Decimal("2.5")
        assert isinstance(ctx, PipelineContext)

    def test_no_factor_uses_local_cost_unchanged(
        self, tmp_path: Any,
    ) -> None:
        """OSS local-only path: no factor cached, cost_usd is bare local
        estimate. Backward compatibility check."""
        ctx = pipeline(
            policy=PipelinePolicy(budget=5.0),
            log_path=tmp_path / "runs.jsonl",
            router=FakeRouter(),
        )
        ctx._estimator = FakeCostEstimator(cost=Decimal("0.04"))  # type: ignore[assignment]
        record = ctx.record(
            model_requested="claude-sonnet-4",
            model_used="claude-sonnet-4",
            response=_StubResponse(prompt_tokens=1000, completion_tokens=500),
            elapsed_ms=42.0,
        )
        assert record.cost_usd == Decimal("0.04")

    def test_invalid_factor_silently_falls_back(
        self, tmp_path: Any, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A negative or zero calibration factor must not crash record()
        — fail-open extends to accounting accuracy. The local estimate
        is used unchanged."""
        ctx = pipeline(
            policy=PipelinePolicy(budget=5.0),
            log_path=tmp_path / "runs.jsonl",
            router=FakeRouter(),
        )
        ctx._estimator = FakeCostEstimator(cost=Decimal("0.04"))  # type: ignore[assignment]
        record = ctx.record(
            model_requested="claude-sonnet-4",
            model_used="claude-sonnet-4",
            response=_StubResponse(prompt_tokens=1000, completion_tokens=500),
            elapsed_ms=42.0,
            calibration_factor=Decimal("-1.5"),
        )
        # Negative factor ignored; cost_usd unchanged from local estimate.
        assert record.cost_usd == Decimal("0.04")


class TestEndToEndAcceptanceCriterion2:
    """L6E-73 Acceptance #2: `pipeline(policy, cloud=CloudConfig(...))`
    + `ctx.call(user_id="u_7")` produces a request body with
    `user_id="u_7"` reaching ``/v1/authorize``. Server-side flip to
    ``is_margin_request=True`` is verified in the hosted-edge suite."""

    def test_call_with_user_id_sends_margin_body(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
    ) -> None:
        monkeypatch.setenv("L6E_API_KEY", "sk-l6e-test")

        captured: dict = {}

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _VALID_RESP
        mock_resp.text = ""
        mock_client = MagicMock()

        def _capture_post(*args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs.get("json", {}))
            return mock_resp

        mock_client.post = MagicMock(side_effect=_capture_post)

        with patch("l6e.cloud._get_sync_client", return_value=mock_client):
            ctx = pipeline(
                policy=PipelinePolicy(budget=5.0),
                cloud=CloudConfig(),
                log_path=tmp_path / "runs.jsonl",
                router=FakeRouter(),
            )
            ctx.call(
                lambda model, messages: _StubResponse(100, 50),
                model="claude-sonnet-4",
                messages=[{"role": "user", "content": "hi"}],
                stage="planning",
                user_id="u_7",
                cohort_hint="paid",
            )

        assert captured["user_id"] == "u_7"
        assert captured["cohort_hint"] == "paid"
        # session_id is the ctx.run_id — verifies it lands as the
        # session correlation key for ``authorize_events``.
        assert captured["session_id"] == ctx.run_id
