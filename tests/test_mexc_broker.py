"""MexcBroker dry-run + mocked CCXT param construction."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from moonshot.brokers import MexcBroker
from moonshot.config import Config
from moonshot.types import Direction, ExecutionError


def test_dry_run_logs_without_network(cfg: Config) -> None:
    b = MexcBroker(cfg, dry_run=True)
    b.set_leverage("BTC/USDT:USDT", 5)
    fill = b.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    assert fill.order_id.startswith("dry-")
    b.set_stop("BTC/USDT:USDT", Direction.LONG, 0.01, 95.0)
    b.cancel_stop("BTC/USDT:USDT")
    assert b.get_equity() == cfg.starting_equity
    ops = [c["op"] for c in b._call_log]
    assert ops == [
        "set_leverage", "create_order", "set_stop", "cancel_stop", "fetch_balance",
    ]
    # Opening order params include isolated openType + leverage
    create = next(c for c in b._call_log if c["op"] == "create_order")
    assert create["params"]["openType"] == 1
    assert create["params"]["leverage"] == 5
    assert create["params"]["reduceOnly"] is False


def test_dry_run_reduce_only_omits_leverage(cfg: Config) -> None:
    b = MexcBroker(cfg, dry_run=True)
    b.market_order("ETH/USDT:USDT", "sell", 0.1, 50.0, reduce_only=True)
    create = b._call_log[-1]
    assert create["params"]["reduceOnly"] is True
    assert "leverage" not in create["params"]
    assert "openType" not in create["params"]


def test_mocked_exchange_create_order_params(cfg: Config) -> None:
    ex = MagicMock()
    ex.market.return_value = {"contractSize": 1.0}
    ex.amount_to_precision.side_effect = lambda _s, q: q
    ex.price_to_precision.side_effect = lambda _s, p: p
    ex.create_order.return_value = {
        "id": "oid-1", "average": 100.5, "filled": 0.01, "fee": {"cost": 0.01},
    }
    b = MexcBroker(cfg, dry_run=False, exchange=ex)
    b.set_leverage("BTC/USDT:USDT", 7)
    fill = b.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    assert fill.order_id == "oid-1"
    args, kwargs = ex.create_order.call_args
    # symbol, type, side, amount, price, params
    assert args[0] == "BTC/USDT:USDT"
    assert args[1] == "market"
    assert args[2] == "buy"
    params = args[5]
    assert params["openType"] == 1
    assert params["leverage"] == 7


def test_api_circuit_breaker(cfg: Config) -> None:
    ex = MagicMock()
    ex.market.return_value = {"contractSize": 1.0}
    ex.amount_to_precision.side_effect = lambda _s, q: q
    ex.create_order.side_effect = RuntimeError("network")
    b = MexcBroker(cfg, dry_run=False, exchange=ex, api_fail_halt=2)
    with pytest.raises(ExecutionError):
        b.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    with pytest.raises(ExecutionError):
        b.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    assert b._halted
    with pytest.raises(ExecutionError, match="halted"):
        b.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
