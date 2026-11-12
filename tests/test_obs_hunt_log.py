"""Hunt pass experiment_log packing."""

from __future__ import annotations

from pathlib import Path

from moonshot.experiment_log import get_runs
from moonshot.obs import enrich_ladder_rows, log_hunt_pass_experiment


def test_log_hunt_pass_experiment_shape(tmp_path: Path):
    db = tmp_path / "experiments.db"
    log_hunt_pass_experiment(
        generation=42,
        config_hash="abc123",
        data_hash=None,
        funnel={
            "scout_in": 80,
            "scout_keep": 16,
            "confirm_soft": 4,
            "confirm_b": 10,
            "confirm_dual_fail": 1,
            "matrix_scored": 12,
            "finalist": 1,
            "gates_pass": 0,
            "promote": 0,
        },
        kill_hist={"sparse": 3, "period_robust": 2},
        ladder=enrich_ladder_rows(
            [{"leverage": 10, "week_ret": 0.2, "tf": "5m"}],
            book_exp=0.15,
            book_dd=0.12,
            cost_2x_ok=True,
        ),
        timing={"stage_a_s": 1.2, "total_s": 9.0},
        cache={"hits": 10, "misses": 2, "prepare_calls": 12},
        meta={"hunt_source": "binance", "generation": 42, "promote_reason": "hold"},
        db_path=db,
    )
    rows = get_runs({"kind": "hunt_pass"}, db_path=db)
    assert len(rows) == 1
    import json

    metrics = json.loads(rows[0]["metrics_json"])
    assert metrics["funnel"]["scout_in"] == 80
    assert metrics["kill_histogram"]["sparse"] == 3
    assert metrics["leverage_ladder"][0]["exp"] == 0.15
    assert metrics["leverage_ladder"][0]["cost_2x_ok"] is True
    assert metrics["sealed_holdout"]["evaluated"] is False
    assert metrics["sealed_holdout"].get("why") in (None, "not_provided")
    assert metrics["timing"]["total_s"] == 9.0
