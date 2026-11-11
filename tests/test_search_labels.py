"""Recent searches labels must not lie about champion vs Live/paper."""

from __future__ import annotations

from moonshot.hunt.search_labels import (
    holdout_column_name,
    holdout_week_note,
    matrix_scored_label,
    what_happened_label,
)


def test_silence_density_never_says_new_champion():
    h = {
        "improved": True,
        "promote_reason": "silence_density",
        "promote_tier": "silence_density",
        "week_return": 0.30,
        "tested": 79,
    }
    msg = what_happened_label(h)
    assert "New champion" not in msg
    assert "New peak champion" not in msg
    assert "Live/paper rotated" in msg
    assert "peak champion unchanged" in msg
    assert "silence_density" in msg


def test_silence_rotate_never_says_new_champion():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "silence_rotate", "promote_tier": "silence_rotate"}
    )
    assert "New champion" not in msg
    assert "Live/paper rotated" in msg


def test_matrix_escape_improved_is_live_rotate_not_champion():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "matrix_escape", "promote_tier": "matrix_escape"}
    )
    assert "New champion" not in msg
    assert "Live/paper rotated" in msg
    assert "matrix_escape" in msg


def test_research_promote_says_peak_champion():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "research", "promote_tier": "research"}
    )
    assert msg.startswith("New peak champion deployed to paper")


def test_champion_rescue_says_restored():
    msg = what_happened_label(
        {"improved": True, "promote_reason": "champion_rescue", "promote_tier": "champion_rescue"}
    )
    assert "Peak champion restored" in msg
    assert "New champion" not in msg


def test_hold_does_not_say_still_champion():
    msg = what_happened_label(
        {
            "improved": False,
            "promote_reason": "matrix_escape",
            "tested": 46,
            "material_why": "identical knobs",
        }
    )
    assert "Still champion" not in msg
    assert "No deploy" in msg
    assert "peak unchanged" in msg


def test_holdout_week_note_not_champion_wording():
    note = holdout_week_note(
        {"improved": True, "promote_reason": "silence_density", "week_return": 0.298}
    )
    assert "champion" not in note.lower()
    assert "+30%" in note
    assert "Live" in note


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
        # Legacy collapse must be ignored when full matrix present.
        "leverage_ladder": [
            {"timeframe": "5m", "leverage": 5},
            {"timeframe": "5m", "leverage": 10},
        ],
    }
    label = matrix_scored_label(h)
    assert "15m" in label and "1h" in label and "4h" in label
    assert "ladder alias" not in label
    assert "5 cells" in label


def test_matrix_label_legacy_ladder_marked_alias():
    h = {
        "tested": 79,
        "leverage_ladder": [
            {"timeframe": "5m", "leverage": 5},
            {"timeframe": "5m", "leverage": 10},
            {"timeframe": "5m", "leverage": 15},
        ],
    }
    label = matrix_scored_label(h)
    assert "ladder alias" in label
    assert "5m" in label
