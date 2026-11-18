"""
Tests for training/metrics.py, specifically that its outputs are actually
JSON-serializable end to end — evaluate.py's --output flag dumps
MetricsResult.to_dict() + meets_targets()/target_results straight to
json.dump(), and numpy scalar types don't always cooperate with that.
"""

import json

import numpy as np
import torch

from module3_deep_learning.training.metrics import MetricsCalculator


def test_metrics_result_is_json_serializable():
    """
    Regression test: MetricsResult.meets_targets() (and evaluate.py's
    parallel _check_targets()) used to return numpy.bool_ from comparisons
    like `self.precision >= 0.90` where self.precision is a numpy.float64
    (sklearn's return type). numpy.float64 subclasses Python's float, so it
    serializes fine, but numpy.bool_ does NOT subclass Python's bool — so
    json.dump() raised "Object of type bool is not JSON serializable" the
    moment a real (non-empty) evaluation tried to write --output.
    """
    calc = MetricsCalculator()
    failure_prob = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    failure_target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    type_logits = torch.zeros(2, 2, 12)
    type_target = torch.tensor([[0, -1], [-1, 1]])
    time_pred = torch.tensor([[10.0, 0.0], [0.0, 20.0]])
    time_target = torch.tensor([[15.0, -1.0], [-1.0, 25.0]])

    calc.update(failure_prob, failure_target, type_logits, type_target, time_pred, time_target)
    result = calc.compute()

    json.dumps(result.to_dict())          # must not raise
    json.dumps(result.meets_targets())    # must not raise


def test_find_optimal_threshold_respects_fpr_constraint_and_beats_default():
    """
    Regression test: find_optimal_threshold() existed (constrained to
    FPR <= target, maximizing F1 within that) but nothing in evaluate.py
    ever called it — every real evaluation silently used the hardcoded
    default threshold=0.5 regardless of what the model's probabilities
    actually looked like. This checks the method itself does what its
    docstring claims: 18 clean negatives near 0.0-0.3, 2 negatives that
    would trip a low threshold (0.55, 0.6), and 5 positives spread
    0.35-0.75. At threshold=0.5, several positives (0.35, 0.4, 0.45) are
    missed (recall 2/5); a lower threshold recovers them but only if FPR
    stays <= 0.05 (at most 1 false positive out of 20 negatives).
    """
    calc = MetricsCalculator()
    neg_probs = [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.2,
                 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.55, 0.6]
    pos_probs = [0.35, 0.4, 0.45, 0.65, 0.75]
    probs = neg_probs + pos_probs
    targets = [0.0] * len(neg_probs) + [1.0] * len(pos_probs)

    n = len(probs)
    calc.update(
        torch.tensor(probs).view(1, n),
        torch.tensor(targets).view(1, n),
        torch.zeros(1, n, 12),
        torch.full((1, n), -1),
        torch.zeros(1, n),
        torch.full((1, n), -1.0),
    )

    default_result = calc.compute()  # threshold=0.5
    optimal_threshold, optimal_result = calc.find_optimal_threshold(target_fpr=0.05)

    assert optimal_result.false_positive_rate <= 0.05
    assert optimal_result.recall >= default_result.recall
    assert optimal_result.f1 >= default_result.f1
    # threshold restored after find_optimal_threshold (it saves/restores internally)
    assert calc.threshold == 0.5
