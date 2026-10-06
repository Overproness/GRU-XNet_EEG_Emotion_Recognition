import numpy as np
import pandas as pd
import torch
from scripts.source_bn_diagnostic import calibrate
from scripts.audit_learning_controls import independent_metrics


def test_population_moments_weight_unequal_chunks_and_preserve_parameters():
    torch.manual_seed(18)
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 1), torch.nn.BatchNorm2d(3),
                                torch.nn.ELU(), torch.nn.Dropout(.5),
                                torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3),
                                torch.nn.ELU(), torch.nn.Conv2d(3, 2, 1),
                                torch.nn.BatchNorm2d(2))
    source = torch.randn(16, 2, 3, 5); source[-4:] += 20
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    first = model[0](source).detach().double().numpy()
    stats = calibrate(model, source)
    np.testing.assert_allclose(stats['1']['mean'], first.mean((0, 2, 3)), atol=1e-6)
    np.testing.assert_allclose(stats['1']['variance'], first.var((0, 2, 3)), atol=1e-5)
    assert stats['1']['observations_per_feature'] == 16*3*5
    assert not model.training and all(not m.training for m in model.modules())
    for name, p in model.named_parameters(): torch.testing.assert_close(original[name], p, atol=0, rtol=0)


def test_independent_binary_score_conditions_on_non_neutral_probability_mass():
    rows = pd.DataFrame({'label': [0, 1, 2, 1], 'original_label': [0, 1, 3, 2],
                         'p0': [.8, .7, .7, .4], 'p1': [.1, .2, .1, .4], 'p2': [.1, .1, .2, .2]})
    result = independent_metrics(rows, 'SEEDIV')
    assert result['binary']['n'] == 3
    assert result['binary']['balanced_accuracy'] == 1.
    expected = -np.log(np.float32(2/3))
    np.testing.assert_allclose(result['binary']['balanced_log_loss'], expected, atol=1e-7)
