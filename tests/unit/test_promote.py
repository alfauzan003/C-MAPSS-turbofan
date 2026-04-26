"""Tests for compare_and_promote — auto-promotion rule logic."""

from pdm.models.train import PromoteDecision, compare_and_promote_decision


def test_promotes_when_new_beats_old_by_more_than_threshold():
    decision = compare_and_promote_decision(
        new_rmse=10.0, current_production_rmse=11.0, improvement_threshold_pct=2.0
    )
    # 11 → 10 = ~9% improvement, > 2%
    assert decision is PromoteDecision.PROMOTE


def test_does_not_promote_when_improvement_below_threshold():
    decision = compare_and_promote_decision(
        new_rmse=10.9, current_production_rmse=11.0, improvement_threshold_pct=2.0
    )
    # ~0.9% improvement, < 2% threshold
    assert decision is PromoteDecision.HOLD


def test_does_not_promote_when_new_is_worse():
    decision = compare_and_promote_decision(
        new_rmse=12.0, current_production_rmse=11.0, improvement_threshold_pct=2.0
    )
    assert decision is PromoteDecision.HOLD


def test_promotes_when_no_current_production():
    decision = compare_and_promote_decision(
        new_rmse=10.0, current_production_rmse=None, improvement_threshold_pct=2.0
    )
    assert decision is PromoteDecision.PROMOTE


def test_threshold_is_inclusive_lower_bound():
    """Improvement of exactly the threshold counts as promote."""
    decision = compare_and_promote_decision(
        new_rmse=98.0, current_production_rmse=100.0, improvement_threshold_pct=2.0
    )
    assert decision is PromoteDecision.PROMOTE
