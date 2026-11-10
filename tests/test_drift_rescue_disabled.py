"""Silence rotate is the idle/bleed escape hatch (Product A) — keep enabled."""

from __future__ import annotations

from moonshot.hunt import constants as hc


def test_silence_rotate_enabled():
    assert hc.HUNT_SILENCE_ROTATE_ENABLED is True
