"""Phase 3 — sealed holdout selection boundary + promote gate."""

from __future__ import annotations

import pandas as pd
import pytest

from moonshot.hunt.promote_gates import (
    assert_selection_before_sealed,
    sealed_holdout_gate,
    selection_end_for_sealed,
)


def test_selection_end_before_data_end():
    end = pd.Timestamp("2026-11-12", tz="UTC")
    sel = selection_end_for_sealed(end)
    assert sel < end
    assert (end - sel).days == 14


def test_assert_selection_rejects_past_sealed():
    end = pd.Timestamp("2026-11-12", tz="UTC")
    sel = selection_end_for_sealed(end)
    assert_selection_before_sealed(sel, end)
    with pytest.raises(RuntimeError, match="sealed"):
        assert_selection_before_sealed(end, end)


def test_sealed_fails_when_red():
    ok, why, meta = sealed_holdout_gate(
        sealed_week_ret=-0.05,
        live_sealed_week_ret=-0.10,
    )
    assert ok is False
    assert meta["evaluated"] is True


def test_sealed_requires_edge_over_live():
    ok, why, meta = sealed_holdout_gate(
        sealed_week_ret=0.01,
        live_sealed_week_ret=0.01,
        research_min_week_edge=0.008,
    )
    assert ok is False
