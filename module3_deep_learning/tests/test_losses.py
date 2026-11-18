"""Tests for MultiTaskLoss / MultiTaskFocalLoss (training/losses.py)."""

import math

import torch

from module3_deep_learning.training.losses import MultiTaskFocalLoss, MultiTaskLoss

BATCH, SERVICES, NUM_TYPES = 4, 7, 12


def _random_batch(all_targets_negative: bool = False):
    torch.manual_seed(0)
    failure_logits = torch.randn(BATCH, SERVICES)
    failure_type_logits = torch.randn(BATCH, SERVICES, NUM_TYPES)
    time_to_failure = torch.rand(BATCH, SERVICES) * 300

    if all_targets_negative:
        is_failure_target = torch.zeros(BATCH, SERVICES)
        failure_type_target = torch.full((BATCH, SERVICES), -1, dtype=torch.long)
        time_to_failure_target = torch.full((BATCH, SERVICES), -1.0)
    else:
        is_failure_target = torch.randint(0, 2, (BATCH, SERVICES)).float()
        failure_type_target = torch.where(
            is_failure_target.bool(), torch.randint(0, NUM_TYPES, (BATCH, SERVICES)), torch.full((BATCH, SERVICES), -1)
        )
        time_to_failure_target = torch.where(
            is_failure_target.bool(), torch.rand(BATCH, SERVICES) * 300, torch.full((BATCH, SERVICES), -1.0)
        )

    return (
        failure_logits, failure_type_logits, time_to_failure,
        is_failure_target, failure_type_target, time_to_failure_target,
    )


def test_loss_is_finite_on_a_normal_mixed_batch():
    loss_fn = MultiTaskLoss()
    total_loss, loss_dict = loss_fn(*_random_batch(all_targets_negative=False))
    assert math.isfinite(total_loss.item())
    assert math.isfinite(loss_dict["type_ce"])


def test_loss_is_finite_when_batch_has_no_failures():
    """Regression test: most windows have no failure at all, so a batch's
    failure_type targets can be entirely -1 (ignore_index). Before this fix,
    nn.CrossEntropyLoss(ignore_index=-1)'s default 'mean' reduction divided
    by the count of non-ignored targets (0 in this case), silently returning
    NaN and poisoning total_loss — and every gradient computed from it —
    instead of raising. Caught via a real local training dry run on the
    small dummy dataset (35 windows, 7 services): epoch 1's randomly
    shuffled batches hit this and Train Loss / Val Loss both logged as nan."""
    loss_fn = MultiTaskLoss()
    total_loss, loss_dict = loss_fn(*_random_batch(all_targets_negative=True))
    assert math.isfinite(total_loss.item()), "total_loss must not be NaN when a batch has zero failure examples"
    assert loss_dict["type_ce"] == 0.0
    assert loss_dict["time_mse"] == 0.0


def test_focal_loss_variant_is_also_finite_when_batch_has_no_failures():
    """Same NaN guard, but for MultiTaskFocalLoss's independent copy of the
    type-loss computation (it doesn't reuse MultiTaskLoss.forward)."""
    loss_fn = MultiTaskFocalLoss()
    total_loss, loss_dict = loss_fn(*_random_batch(all_targets_negative=True))
    assert math.isfinite(total_loss.item())
    assert loss_dict["type_ce"] == 0.0


def test_mlkp_term_is_zero_safe_when_absent():
    """mlkp_logits/targets default to None (auxiliary task off, eval mode,
    or a batch that happened to mask zero positions — see full_model.py's
    _compute_mlkp) — same guard shape as type_loss/time_loss, must never
    raise or produce NaN."""
    loss_fn = MultiTaskLoss()
    total_loss, loss_dict = loss_fn(*_random_batch(all_targets_negative=False))
    assert loss_dict["mlkp_ce"] == 0.0
    assert math.isfinite(total_loss.item())


def test_mlkp_term_is_finite_and_contributes_to_total_when_present():
    num_services = SERVICES
    n_masked = 5
    mlkp_logits = torch.randn(n_masked, num_services, requires_grad=True)
    mlkp_targets = torch.randint(0, num_services, (n_masked,))

    loss_fn = MultiTaskLoss(mlkp_weight=0.3)
    batch = _random_batch(all_targets_negative=False)
    total_with, loss_dict = loss_fn(*batch, mlkp_logits=mlkp_logits, mlkp_targets=mlkp_targets)
    total_without, _ = loss_fn(*batch)

    assert math.isfinite(total_with.item())
    assert loss_dict["mlkp_ce"] > 0.0
    assert not torch.isclose(total_with, total_without)

    total_with.backward()
    assert mlkp_logits.grad is not None
    assert torch.isfinite(mlkp_logits.grad).all()


def test_mlkp_term_wired_into_focal_variant_too():
    """MultiTaskFocalLoss has its own independent forward() (doesn't reuse
    MultiTaskLoss.forward) — the MLKP guard/term must be duplicated there
    too, not silently dropped for anyone using the focal-loss variant."""
    n_masked = 3
    mlkp_logits = torch.randn(n_masked, SERVICES)
    mlkp_targets = torch.randint(0, SERVICES, (n_masked,))

    loss_fn = MultiTaskFocalLoss(mlkp_weight=0.3)
    batch = _random_batch(all_targets_negative=False)
    total_loss, loss_dict = loss_fn(*batch, mlkp_logits=mlkp_logits, mlkp_targets=mlkp_targets)

    assert math.isfinite(total_loss.item())
    assert loss_dict["mlkp_ce"] > 0.0


def test_pos_weight_and_type_class_weights_are_actually_applied():
    """Regression test: MultiTaskLoss accepted pos_weight/type_class_weights
    but train.py never passed them (dataset.get_class_weights() was dead
    code) — verify that when supplied, they measurably change the loss
    versus the unweighted default, so a future accidental removal of the
    wiring in train.py would show up as a behavior change here too."""
    batch = _random_batch(all_targets_negative=False)
    unweighted = MultiTaskLoss()
    weighted = MultiTaskLoss(
        pos_weight=5.0,
        type_class_weights=torch.linspace(0.1, 2.0, NUM_TYPES),
    )
    loss_a, _ = unweighted(*batch)
    loss_b, _ = weighted(*batch)
    assert not torch.isclose(loss_a, loss_b)
