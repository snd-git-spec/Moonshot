"""Promote gates must stay imported/called from improve — no silent stubs."""

from __future__ import annotations

import ast
from pathlib import Path

from moonshot.hunt import promote_gates
from moonshot.hunt.promote_gates import PROMOTE_GATE_SYMBOLS


def test_promote_gate_symbols_exist():
    for name in PROMOTE_GATE_SYMBOLS:
        assert hasattr(promote_gates, name), f"missing promote_gates.{name}"


def test_improve_references_promote_gates():
    """AST/source check: improve.py must call the integrity gate surface."""
    root = Path(__file__).resolve().parents[1]
    improve_path = root / "moonshot" / "hunt" / "improve.py"
    src = improve_path.read_text(encoding="utf-8")
    assert "promote_gates" in src or "from moonshot.hunt.promote_gates" in src
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    required = {
        "live_death_book",
        "sealed_holdout_gate",
        "research_dsr_gate",
        "selection_end_for_sealed",
        "escape_sealed_floor_ok",
        "record_pass_trials",
    }
    missing = required - names
    assert not missing, f"improve.py missing promote gate refs: {sorted(missing)}"


def test_live_death_book_green_week_not_death():
    ok, why = promote_gates.live_death_book(
        week_ret=0.39, oos_ret=-0.05, soft_ret=-0.05, paper_bleeding=False
    )
    assert ok is False
    assert why == "live_not_death"


def test_live_death_book_red_week():
    ok, why = promote_gates.live_death_book(
        week_ret=-0.05, oos_ret=0.1, soft_ret=0.1, paper_bleeding=False
    )
    assert ok is True
    assert "week" in why


def test_sealed_gate_beats_live():
    ok, why, meta = promote_gates.sealed_holdout_gate(
        sealed_week_ret=0.05,
        live_sealed_week_ret=0.0,
        kill_illusion=False,
    )
    assert ok is True
    assert meta["passed"] is True


def test_sealed_gate_fails_flat_vs_live():
    ok, why, meta = promote_gates.sealed_holdout_gate(
        sealed_week_ret=0.0,
        live_sealed_week_ret=0.0,
        kill_illusion=False,
        research_min_week_edge=0.008,
    )
    assert ok is False
    assert "sealed" in why


def test_research_dsr_gate_blocks_negative():
    ok, why, meta = promote_gates.research_dsr_gate(
        observed_sr=-0.5, n_obs=40, n_trials=100, threshold=0.0
    )
    assert ok is False
    assert why == "dsr_fail"
    assert meta["evaluated"] is True
