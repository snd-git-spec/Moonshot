"""Recent searches labels: better found, or not — no jargon lies."""

from __future__ import annotations

from moonshot.hunt.search_labels import (
    holdout_column_name,
    holdout_week_note,
    matrix_scored_label,
    what_happened_label,
)


def test_silence_is_not_better_setup():
    msg = what_happened_label(
        {
            "improved": True,
            "promote_reason": "silence_density",
            "promote_tier": "silence_density",
            "week_return": 0.30,
            "tested": 79,
        }
    )
    assert msg == "No better setup found · paper settings tweaked while waiting"
    assert "silence" not in msg.lower()
    assert "champion" not in msg.lower()
    assert "rotated" not in msg.lower()


def test_silence_rotate_same_plain_line():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "silence_rotate", "promote_tier": "silence_rotate"}
    )
    assert msg == "No better setup found · paper settings tweaked while waiting"


def test_matrix_escape_improved_not_better():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "matrix_escape", "promote_tier": "matrix_escape"}
    )
    assert msg == "No better setup found · paper settings tweaked while waiting"


def test_research_is_better_found():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "research", "promote_tier": "research"}
    )
    assert msg == "Better setup found · deployed to paper"


def test_champion_rescue_plain():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "champion_rescue", "promote_tier": "champion_rescue"}
    )
    assert msg == "Best known setup put back on paper"


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


def test_holdout_column_never_says_champion():
    assert "champion" not in holdout_column_name("holdout 14d").lower()


def test_matrix_label_uses_full_tf_lev_matrix():
    h = {
        "tested": 46,
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
