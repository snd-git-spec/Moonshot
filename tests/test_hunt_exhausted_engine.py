"""Phase 2 — exhaustion changes search; clears only on robust."""

from __future__ import annotations

from moonshot.hunt.scoring import _discover_seed_cfgs, _hunt_search_mode
from moonshot.config import Config


def test_exhausted_forces_breakout():
    assert (
        _hunt_search_mode(10, held_streak=0, hunt=True, has_champion=True, hunt_exhausted=True)
        == "breakout"
    )


def test_stuck_still_mixes_without_exhausted():
    mode = _hunt_search_mode(0, held_streak=10, hunt=True, has_champion=True, hunt_exhausted=False)
    assert mode in ("breakout", "explore")


def test_discover_every_pass_when_exhausted(tmp_path, monkeypatch):
    import moonshot.hunt.scoring as sc

    rules = tmp_path / "rules.json"
    rules.write_text('{"top":[{"x":1},{"y":2},{"z":3}]}')
    monkeypatch.setattr(sc, "RULES_PATH", rules)
    base = Config(strategy="momentum_15", timeframe="5m", leverage=10)
    # held low + gen not multiple of 11 — normally empty
    assert _discover_seed_cfgs(base, generation=1, held=0, hunt_exhausted=False) == []
    seeds = _discover_seed_cfgs(base, generation=1, held=0, hunt_exhausted=True)
    assert len(seeds) >= 1


def test_improve_clears_exhausted_only_on_robust():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "moonshot/hunt/improve.py").read_text()
    assert 'promote_provenance == "robust"' in src
    assert "regime_reassess_needed" in src
    assert "escape promote · hunt_exhausted stays" in src
