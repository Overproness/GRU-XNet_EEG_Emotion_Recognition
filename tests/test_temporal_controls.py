"""Check the actual scientific controls: labels, observations, order and sampling."""
import numpy as np
import torch

from gruxnet.temporal_controls import (COARSE, TemporalControl, coarse_probabilities,
                                      common_metrics, draw_batches, represent, sequence_features)


def test_relative_power_removes_common_channel_gain():
    t = np.arange(512)/128
    waveform = (np.sin(2*np.pi*6*t)+.5*np.sin(2*np.pi*10*t)+.2*np.sin(2*np.pi*20*t)+.1*np.sin(2*np.pi*35*t))
    windows = np.tile(waveform,(10,14,1)).astype(np.float32)
    powers = sequence_features(windows)
    gained = sequence_features(windows*4)
    np.testing.assert_allclose(represent(powers,"relative"),represent(gained,"relative"),atol=3e-6)
    np.testing.assert_allclose(gained-powers,2*np.log(4),atol=3e-6)


def test_native_and_coarse_have_identical_initial_grouped_probabilities():
    x = torch.randn(6,10,56)
    for architecture in ("mean_mlp","transformer"):
        torch.manual_seed(42)
        coarse = TemporalControl(architecture,"coarse3").eval()
        torch.manual_seed(42)
        native = TemporalControl(architecture,"native4").eval()
        with torch.no_grad():
            p = torch.softmax(coarse(x),1).numpy()
            q = torch.softmax(native(x),1).numpy()
        np.testing.assert_allclose(p,coarse_probabilities(q,"native4"),atol=1e-7)
        for name,parameter in coarse.named_parameters():
            if not name.startswith("head."):
                torch.testing.assert_close(parameter,dict(native.named_parameters())[name],rtol=0,atol=0)


def test_mean_mlp_is_order_invariant_but_temporal_transformer_uses_positions():
    torch.manual_seed(42)
    x = torch.randn(6,10,56)
    mean = TemporalControl("mean_mlp","coarse3").eval()
    temporal = TemporalControl("transformer","coarse3").eval()
    with torch.no_grad():
        torch.testing.assert_close(mean(x),mean(x.flip(1)),atol=1e-6,rtol=1e-6)
        assert not torch.allclose(temporal(x),temporal(x.flip(1)),atol=1e-6,rtol=1e-6)
    assert sum(p.numel() for p in mean.parameters())==19079
    assert sum(p.numel() for p in temporal.parameters())==19075


def test_sampling_holds_trials_and_coarse_classes_fixed_for_native_training():
    labels = np.tile(np.arange(4),20)
    train = np.arange(40)
    first = draw_batches(labels,train,42,updates=5)
    second = draw_batches(labels,train,42,updates=5)
    np.testing.assert_array_equal(first,second)
    assert set(first.ravel()).issubset(set(train))
    for batch in first:
        assert np.bincount(COARSE[labels[batch]],minlength=3).tolist()==[20,20,20]
    assert set(labels[first.ravel()])=={0,1,2,3}


def test_common_metrics_use_grouped_labels_and_condition_on_nonneutral():
    y = np.array([0,1,2,3])
    native = np.array([[.8,.05,.05,.1],[.1,.5,.3,.1],[.1,.2,.6,.1],[.1,.05,.05,.8]])
    fine = common_metrics(y,native,"native4")
    coarse = common_metrics(y,coarse_probabilities(native,"native4"),"coarse3")
    assert fine["coarse3"]==coarse["coarse3"]
    assert fine["binary"]==coarse["binary"]
    assert fine["coarse3"]["n"]==4 and fine["binary"]["n"]==3
    assert fine["binary"]["balanced_accuracy"]==1
