import numpy as np
import pandas as pd
import pytest
import torch

from gruxnet.deap_control import candidate_is_better, trial_diagnostics
from gruxnet.eegnet import EEGNetControl, TrainingChannelScaler, normalized_waveforms


def test_scaler_uses_training_values_and_rejects_held_out_rows(pdf_workspace):
    cache = pdf_workspace
    first = np.arange(32, dtype=np.float32).reshape(2, 2, 8)
    second = (100 + np.arange(16, dtype=np.float32)).reshape(1, 2, 8)
    np.save(cache / "train1.npy", first)
    np.save(cache / "train2.npy", second)
    rows = pd.DataFrame([dict(cache_file="train1.npy", cache_index=0, split="train"),
                         dict(cache_file="train1.npy", cache_index=1, split="train"),
                         dict(cache_file="train2.npy", cache_index=0, split="train")])
    held_out = pd.DataFrame([dict(cache_file="held_out_not_present.npy", cache_index=0, split="validation")])
    all_values = np.concatenate([first, second]).astype(np.float64)
    fitted = TrainingChannelScaler.fit(rows, cache)
    np.testing.assert_allclose(fitted.mean, all_values.mean(axis=(0, 2)), atol=1e-12)
    np.testing.assert_allclose(fitted.scale, all_values.std(axis=(0, 2)), atol=1e-12)
    assert fitted.count == 24
    with pytest.raises(ValueError, match="training rows only"):
        TrainingChannelScaler.fit(pd.concat([rows, held_out]), cache)
    fitted.save(cache / "stats.npz")
    loaded = TrainingChannelScaler.load(cache / "stats.npz")
    np.testing.assert_array_equal(fitted.mean, loaded.mean)
    np.testing.assert_array_equal(fitted.scale, loaded.scale)
    # Held-out amplitude survives a frozen affine transform; per-window RMS
    # deliberately removes it. This verifies the scientific control's difference.
    wave = torch.tensor([[[1., -1., 1., -1.]]])
    stats = (torch.zeros(1, 1, 1), torch.ones(1, 1, 1))
    torch.testing.assert_close(normalized_waveforms(2*wave, stats), 2*normalized_waveforms(wave, stats))
    torch.testing.assert_close(normalized_waveforms(2*wave, stats, "window"), normalized_waveforms(wave, stats, "window"))


def test_eegnet_classifier_is_registered_before_optimizer_and_receives_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = EEGNetControl(channels=3, samples=64, dropout=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    registered = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(model.classifier.weight) in registered and id(model.classifier.bias) in registered
    x = torch.randn(4, 3, 64)
    logits = model(x)
    assert logits.shape == (4, 2)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    assert model.classifier.weight.grad.norm() > 0
    assert model.temporal[1].weight.grad.norm() > 0
    before = model.classifier.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight)
    with torch.no_grad():
        model.spatial.weight.mul_(100)
        model.classifier.weight.mul_(100)
    model.apply_constraints()
    assert float(model.spatial.weight.flatten(1).norm(dim=1).max().detach()) <= 1.000001
    assert float(model.classifier.weight.norm(dim=1).max().detach()) <= .250001


def test_validation_selection_uses_declared_tie_loss_and_trial_aggregation():
    assert candidate_is_better(.51, .9, .50, .6)
    assert candidate_is_better(.50, .6, .50, .7)
    assert not candidate_is_better(.49, .1, .50, .7)
    assert not candidate_is_better(.50, .7, .50, .7)
    predictions = pd.DataFrame([
        dict(dataset="DEAP", subject_id="DEAP:S01", trial_id="a", label=0, positive_probability=.1),
        dict(dataset="DEAP", subject_id="DEAP:S01", trial_id="a", label=0, positive_probability=.3),
        dict(dataset="DEAP", subject_id="DEAP:S02", trial_id="b", label=1, positive_probability=.9),
    ])
    diagnostic = trial_diagnostics(predictions)
    assert diagnostic["trial_predicted_class_counts"] == [1, 1]
    assert diagnostic["balanced_trial_log_loss"] == pytest.approx(( -np.log(.8)-np.log(.9))/2)
