"""Unit tests for exceptions.py — BudgetExceeded and LatencySLAExceeded."""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# LatencySLAExceeded — lines 21-23 (never tested before)
# ---------------------------------------------------------------------------


def test_latency_sla_exceeded_stores_attributes() -> None:
    from l6e.exceptions import LatencySLAExceeded

    exc = LatencySLAExceeded(elapsed_ms=1500.0, sla_ms=1000.0)
    assert exc.elapsed_ms == pytest.approx(1500.0)
    assert exc.sla_ms == pytest.approx(1000.0)


def test_latency_sla_exceeded_message_contains_both_values() -> None:
    from l6e.exceptions import LatencySLAExceeded

    exc = LatencySLAExceeded(elapsed_ms=2345.6, sla_ms=999.9)
    msg = str(exc)
    assert "2345.6" in msg
    assert "999.9" in msg


def test_latency_sla_exceeded_is_exception_subclass() -> None:
    from l6e.exceptions import LatencySLAExceeded

    exc = LatencySLAExceeded(elapsed_ms=100.0, sla_ms=50.0)
    assert isinstance(exc, Exception)


def test_latency_sla_exceeded_can_be_raised_and_caught() -> None:
    from l6e.exceptions import LatencySLAExceeded

    with pytest.raises(LatencySLAExceeded) as exc_info:
        raise LatencySLAExceeded(elapsed_ms=300.0, sla_ms=200.0)

    assert exc_info.value.elapsed_ms == pytest.approx(300.0)
    assert exc_info.value.sla_ms == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# BudgetExceeded — reason="" default (pre-existing but edge case worth covering)
# ---------------------------------------------------------------------------


def test_budget_exceeded_empty_reason_default() -> None:
    from decimal import Decimal

    from l6e.exceptions import BudgetExceeded

    exc = BudgetExceeded(spent=Decimal("0.10"), budget=0.05)
    assert exc.reason == ""
    msg = str(exc)
    assert "0.10" in msg
    assert "0.050000" in msg


def test_latency_sla_exceeded_not_in_public_all() -> None:
    """LatencySLAExceeded is not raised in v0.1; it must not be advertised in __all__."""
    import l6e

    assert "LatencySLAExceeded" not in l6e.__all__
