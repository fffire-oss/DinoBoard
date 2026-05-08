"""Unit tests for compute_schedule_ratio (three-segment schedule).

The schedule has three regions:
  step <= hold_steps           → initial_ratio
  step >= decay_end_step       → 0.0
  hold_steps < step < end      → linear interpolation

Used by:
  - heuristic_guidance ratio
  - training_filter ratio
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.pipeline import compute_schedule_ratio


def test_disabled_when_decay_end_zero():
    for step in [0, 1, 100]:
        assert compute_schedule_ratio(step, 0, 1.0) == 0.0


def test_disabled_when_decay_end_negative():
    assert compute_schedule_ratio(5, -1, 1.0) == 0.0


def test_no_hold_linear_decay():
    # hold=0 → step 1 already starts decaying. Matches legacy behaviour.
    assert compute_schedule_ratio(0, 10, 1.0, hold_steps=0) == pytest.approx(1.0)
    assert compute_schedule_ratio(5, 10, 1.0, hold_steps=0) == pytest.approx(0.5)
    assert compute_schedule_ratio(9, 10, 1.0, hold_steps=0) == pytest.approx(0.1)
    assert compute_schedule_ratio(10, 10, 1.0, hold_steps=0) == 0.0
    assert compute_schedule_ratio(11, 10, 1.0, hold_steps=0) == 0.0


def test_hold_period_then_decay():
    # hold=2, decay_end=5, initial=1.0 → 1.0 / 1.0 / 1.0 / 0.667 / 0.333 / 0.0
    assert compute_schedule_ratio(0, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(1, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(2, 5, 1.0, hold_steps=2) == pytest.approx(1.0)
    assert compute_schedule_ratio(3, 5, 1.0, hold_steps=2) == pytest.approx(2.0 / 3.0)
    assert compute_schedule_ratio(4, 5, 1.0, hold_steps=2) == pytest.approx(1.0 / 3.0)
    assert compute_schedule_ratio(5, 5, 1.0, hold_steps=2) == 0.0


def test_initial_ratio_below_one():
    # Hold period locks at initial_ratio (not 1.0).
    assert compute_schedule_ratio(0, 4, 0.4, hold_steps=1) == pytest.approx(0.4)
    assert compute_schedule_ratio(1, 4, 0.4, hold_steps=1) == pytest.approx(0.4)
    # Step 2: decay span 4-1=3, remaining 4-2=2 → 0.4 * 2/3
    assert compute_schedule_ratio(2, 4, 0.4, hold_steps=1) == pytest.approx(0.4 * 2 / 3)


def test_misconfigured_hold_too_large_raises():
    with pytest.raises(ValueError):
        compute_schedule_ratio(0, 5, 1.0, hold_steps=5)
    with pytest.raises(ValueError):
        compute_schedule_ratio(0, 5, 1.0, hold_steps=10)


def test_strictly_decreasing_in_decay_window():
    prev = compute_schedule_ratio(2, 10, 1.0, hold_steps=2)
    for step in range(3, 10):
        cur = compute_schedule_ratio(step, 10, 1.0, hold_steps=2)
        assert cur < prev, f"non-monotonic at step {step}: {cur} >= {prev}"
        prev = cur


def test_after_decay_end_is_exactly_zero():
    for step in range(5, 20):
        assert compute_schedule_ratio(step, 5, 1.0, hold_steps=2) == 0.0
