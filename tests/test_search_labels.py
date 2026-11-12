"""Recent searches labels: better found, escape, or not — no jargon lies."""

from __future__ import annotations

from moonshot.hunt.search_labels import (
    holdout_column_name,
    holdout_week_note,
    matrix_scored_label,
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


def test_hold_is_no_better():
    msg = what_happened_label(
        {
            "improved": False,
            "promote_reason": "matrix_escape",
            "tested": 46,
            "material_why": "identical knobs",
        }
    )
    assert msg == "No better setup found · checked 46 ideas"
    assert "champion" not in msg.lower()
    assert "identical" not in msg.lower()


def test_holdout_week_is_just_percent():
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
