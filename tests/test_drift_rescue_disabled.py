"""A.2 — silence_rotate disabled; drift path does not family-swap."""

from __future__ import annotations

from moonshot.hunt import constants as hc


def test_drift_rescue_disabled():
    assert hc.HUNT_SILENCE_ROTATE_ENABLED is False
