"""A.3c — cost stress gate + sealed holdout not-read + audit fields."""

from __future__ import annotations

import pytest

from moonshot.hunt.cost_holdout import (
    SealedHoldoutGuard,
    cost_stress_gate,
    sealed_holdout_audit_fields,
    sealed_holdout_slice,
)


def test_cost_stress_gate():
    ok = cost_stress_gate(0.5, 0.1)
    assert ok.eligible
    assert ok.pass_1x and ok.pass_2x
    fail2 = cost_stress_gate(0.5, -0.1)
    assert not fail2.eligible
    assert fail2.pass_1x and not fail2.pass_2x
    fail1 = cost_stress_gate(-0.2, 0.3)
    assert not fail1.eligible


def test_sealed_holdout_not_read():
    sel, hold = sealed_holdout_slice(100, holdout_frac=0.22)
    assert sel.stop == hold.start
    assert hold.stop == 100
    guard = SealedHoldoutGuard(hold)
    # Selection range OK
    guard.assert_range_not_read(sel.start, sel.stop)
    with pytest.raises(RuntimeError, match="sealed holdout"):
        guard.assert_not_read(hold.start)
    assert guard.was_read


def test_sealed_holdout_audit_fields():
    fields = sealed_holdout_audit_fields(
        git_sha="abc123",
        data_slice_hash="deadbeef",
        holdout_start_idx=80,
        holdout_end_idx=100,
        metric=-0.1,
        passed=False,
    )
    assert fields["git_sha"] == "abc123"
    assert fields["data_slice_hash"] == "deadbeef"
    assert fields["retired_on_fail"] is True
    assert set(fields) >= {
        "git_sha",
        "data_slice_hash",
        "holdout_start_idx",
        "holdout_end_idx",
        "metric",
        "passed",
        "retired_on_fail",
    }
