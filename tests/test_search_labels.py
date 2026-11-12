"""Recent searches labels: show hunt activity, not Live reprints."""

from __future__ import annotations

from moonshot.hunt.search_labels import (
    coverage_label,
    funnel_label,
    holdout_column_name,
    holdout_week_note,
    live_holdout_label,
    matrix_scored_label,
    nearest_miss_label,
    oos_week_note,
    robust_flag_note,
    what_happened_label,
)

_ESCAPE = "Escape deploy — left a worse Live book; did not pass period-robust"


def test_silence_is_not_better_setup():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "silence_density",
            "promote_tier": "silence_density",
            "promote_provenance": "escape",
            "week_return": 0.30,
            "tested": 79,
        }
    )
    assert msg == _ESCAPE
    assert "silence" not in msg.lower()
    assert "champion" not in msg.lower()
    assert "rotated" not in msg.lower()


def test_silence_rotate_same_plain_line():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "silence_rotate",
            "promote_tier": "silence_rotate",
            "promote_provenance": "escape",
        }
    )
    assert msg == _ESCAPE


def test_matrix_escape_improved_not_better():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "matrix_escape",
            "promote_tier": "matrix_escape",
            "promote_provenance": "escape",
        }
    )
    assert msg == _ESCAPE


def test_research_is_better_found():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "research",
            "promote_tier": "research",
            "promote_provenance": "robust",
        }
    )
    assert msg == "Better setup found · deployed to paper"


def test_champion_rescue_plain():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "champion_rescue",
            "promote_tier": "champion_rescue",
            "promote_provenance": "robust",
        }
    )
    assert msg == "Best known setup put back on paper"


def test_champion_rescue_escape_not_better():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "champion_rescue",
            "promote_tier": "champion_rescue",
            "promote_provenance": "escape",
        }
    )
    assert msg == _ESCAPE


def test_hold_shows_mode_and_soft():
    msg = what_happened_label(
        {
            "improved": False,
            "promote_reason": "baseline",
            "search_mode": "explore",
            "tested": 25,
            "soft_runs": 14,
            "funnel_counts": {"A_keep": 16, "B": 14},
            "material_why": "identical knobs",
        }
    )
    assert "explore" in msg
    assert "soft-checked 14" in msg
    assert "tried 25" in msg
    assert "identical" not in msg.lower()


def test_hold_unfinished_when_soft_zero():
    msg = what_happened_label(
        {
            "improved": False,
            "search_mode": "breakout",
            "tested": 38,
            "soft_runs": 0,
            "funnel_counts": {"A_keep": 16, "B": 0},
        }
    )
    assert "unfinished" in msg
    assert "breakout" in msg


def test_hold_zero_tested_with_pool_is_live_tf_empty():
    msg = what_happened_label(
        {
            "improved": False,
            "search_mode": "breakout",
            "tested": 0,
            "soft_runs": 0,
            "n_cand": 50,
            "funnel_counts": {"A_in": 0, "A_keep": 0, "B": 0},
        }
    )
    assert "funnel empty" in msg
    assert "Live TF" in msg
    assert "checked 0 ideas" not in msg


def test_funnel_and_coverage_labels():
    h = {
        "soft_runs": 14,
        "tested": 25,
        "n_unique_strat": 3,
        "n_unique_tf": 4,
        "n_unique_lev": 5,
        "funnel_counts": {"A_keep": 16, "B": 14},
        "tf_lev_matrix": [
            {"tf": "5m", "leverage": 5},
            {"tf": "1h", "leverage": 10},
        ],
    }
    assert funnel_label(h) == "A16→B14→soft14"
    assert "3 strat" in coverage_label(h)
    assert "unfinished" in funnel_label(
        {"soft_runs": 0, "funnel_counts": {"A_keep": 16, "B": 0}}
    )


def test_nearest_miss_and_live_holdout():
    h = {
        "improved": False,
        "week_return": -0.017,
        "nearest_miss": {
            "week_ret": -0.08,
            "why": "dual week fail",
            "book": {"strategy": "momentum_15", "timeframe": "1h", "leverage": 7},
        },
    }
    miss = nearest_miss_label(h)
    assert "momentum_15" in miss and "7×@1h" in miss
    assert "week -8%" in miss
    assert live_holdout_label(h) == "Live -2%"
    assert holdout_week_note(h) == "-8%"


def test_holdout_week_is_just_percent_on_promote():
    note = holdout_week_note(
        {"improved": True, "promote_reason": "silence_density", "week_return": 0.298}
    )
    assert note == "+30%"


def test_oos_and_robust_flag():
    assert oos_week_note({"test_return": -0.34}) == "-34%"
    assert robust_flag_note({"promote_provenance": "escape", "improved": True}) == "Escape"
    assert robust_flag_note({"promote_provenance": "robust", "improved": True}) == "Robust"


def test_holdout_column_never_says_champion():
    assert "champion" not in holdout_column_name("holdout 14d").lower()


def test_matrix_escape_improved_not_better_legacy_tier():
    """Older rows may lack provenance — reason still marks escape."""
    msg = what_happened_label(
        {"improved": True, "promote_reason": "matrix_escape", "promote_tier": "matrix_escape"}
    )
    assert msg == _ESCAPE


def test_matrix_label_from_cells():
    h = {
        "tf_lev_matrix": [
            {"tf": "5m", "leverage": 10},
            {"tf": "15m", "leverage": 10},
            {"tf": "1h", "leverage": 7},
            {"tf": "4h", "leverage": 5},
            {"tf": "5m", "leverage": 5},
        ],
        "leverage_ladder": [
            {"timeframe": "5m", "leverage": 5},
            {"timeframe": "5m", "leverage": 10},
        ],
    }
    label = matrix_scored_label(h)
    assert "15m" in label and "1h" in label and "4h" in label
    assert "ladder" not in label


def test_matrix_label_legacy_marked_old_log():
    h = {
        "tested": 79,
        "leverage_ladder": [
            {"timeframe": "5m", "leverage": 5},
            {"timeframe": "5m", "leverage": 10},
            {"timeframe": "5m", "leverage": 15},
        ],
    }
    label = matrix_scored_label(h)
    assert "old log" in label
