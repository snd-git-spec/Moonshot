"""HUNT_LEGACY_SCORING is comparison-only — week-first rank, never default."""

from __future__ import annotations

from types import SimpleNamespace

from moonshot.hunt.scoring import _rank_key


def _br(**kw):
    defaults = dict(
        total_return=0.1,
        win_rate=0.5,
        expectancy_r=0.1,
        trades=20,
        max_drawdown=0.1,
        final_equity=110.0,
        halted="",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_rank_key_fold_primary_by_default(monkeypatch):
    monkeypatch.delenv("HUNT_LEGACY_SCORING", raising=False)
    a = _rank_key(
        _br(),
        _br(total_return=0.05),
        week={"ret": 0.9},
        fold={"p25_sortino": 0.1, "roi": 0.05},
        cost_ok=True,
    )
    b = _rank_key(
        _br(),
        _br(total_return=0.2),
        week={"ret": 0.01},
        fold={"p25_sortino": 2.0, "roi": 0.2},
        cost_ok=True,
    )
    assert b > a  # higher p25 wins despite lower week


def test_rank_key_legacy_week_primary(monkeypatch):
    monkeypatch.setenv("HUNT_LEGACY_SCORING", "1")
    a = _rank_key(
        _br(),
        _br(total_return=0.05),
        week={"ret": 0.9},
        fold={"p25_sortino": 0.1, "roi": 0.05},
        cost_ok=True,
    )
    b = _rank_key(
        _br(),
        _br(total_return=0.2),
        week={"ret": 0.01},
        fold={"p25_sortino": 2.0, "roi": 0.2},
        cost_ok=True,
    )
    assert a > b  # week-first when legacy flag on
