"""LIVE_VERIFIED gate on the live CLI path."""

from __future__ import annotations

import sys

import pytest

from moonshot.cli import main


def test_live_refuses_without_live_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "1")
    monkeypatch.setenv("LIVE_VERIFIED", "0")
    monkeypatch.setattr(sys, "argv", ["moonshot", "live"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1


def test_live_refuses_without_trading_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "0")
    monkeypatch.setenv("LIVE_VERIFIED", "1")
    monkeypatch.setattr(sys, "argv", ["moonshot", "live"])
    with pytest.raises(SystemExit) as ei:
        main()
    assert ei.value.code == 1
