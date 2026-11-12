"""Phase 2 — fold-ranked elite archive / islands."""

from __future__ import annotations

from moonshot.hunt.scoring import (
    _best_island_elite_cfg,
    _elite_rank_tuple,
    _raise_elite_archive,
    _select_polish_base,
)
from moonshot.config import Config


def test_elite_archive_orders_by_p25_not_week():
    state: dict = {}
    _raise_elite_archive(
        state,
        knobs={"strategy": "momentum_15", "timeframe": "5m", "leverage": 10},
        p25_sortino=0.2,
        roi=0.1,
        cost_ok=True,
        week_ret=0.9,
        generation=1,
        ts="2026-01-01T00:00:00Z",
    )
    _raise_elite_archive(
        state,
        knobs={"strategy": "trend_mtf", "timeframe": "1h", "leverage": 5},
        p25_sortino=1.5,
        roi=0.05,
        cost_ok=True,
        week_ret=0.0,
        generation=2,
        ts="2026-01-02T00:00:00Z",
    )
    arch = state["elite_archive"]
    assert arch[0]["strategy"] == "trend_mtf"
    assert float(arch[0]["p25_sortino"]) == 1.5
    assert "momentum_15|5m" in state["islands"]
    assert "trend_mtf|1h" in state["islands"]


def test_select_polish_prefers_island_over_live():
    live = Config(strategy="momentum_15", timeframe="5m", leverage=10)
    island = Config(strategy="trend_mtf", timeframe="1h", leverage=5)
    out = _select_polish_base(
        live_base=live,
        matrix_winner_cfg=None,
        best_by_tf_cfg=None,
        champion_cfg=None,
        island_elite_cfg=island,
    )
    assert out.strategy == "trend_mtf"
    assert out.timeframe == "1h"


def test_best_island_skips_live_fingerprint():
    state: dict = {}
    _raise_elite_archive(
        state,
        knobs={"strategy": "momentum_15", "timeframe": "1h", "leverage": 5},
        p25_sortino=0.5,
        roi=0.1,
        cost_ok=True,
        week_ret=-0.02,
        generation=1,
        ts="2026-01-01T00:00:00Z",
        fingerprint="momentum_15|1h|5",
    )
    _raise_elite_archive(
        state,
        knobs={"strategy": "combo_or", "timeframe": "15m", "leverage": 7},
        p25_sortino=0.4,
        roi=0.08,
        cost_ok=True,
        week_ret=0.01,
        generation=2,
        ts="2026-01-02T00:00:00Z",
        fingerprint="combo_or|15m|7",
    )
    base = Config(strategy="momentum_15", timeframe="1h", leverage=5)

    def _k2c(kn):
        return Config(
            strategy=kn["strategy"],
            timeframe=kn["timeframe"],
            leverage=int(kn["leverage"]),
        )

    cfg = _best_island_elite_cfg(state, base=base, knobs_to_cfg=_k2c)
    assert cfg is not None
    assert cfg.strategy == "combo_or"


def test_elite_rank_cost_beats_fail():
    a = {"p25_sortino": 1.0, "cost_ok": True, "roi": 0.1, "week_ret": 0.0}
    b = {"p25_sortino": 1.0, "cost_ok": False, "roi": 0.9, "week_ret": 0.9}
    assert _elite_rank_tuple(a) > _elite_rank_tuple(b)
