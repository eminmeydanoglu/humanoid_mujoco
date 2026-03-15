from __future__ import annotations

from pathlib import Path


def test_rl_plan_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "rl_plan.md").exists()
