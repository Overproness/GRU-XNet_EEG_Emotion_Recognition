import numpy as np
from gruxnet.reve_probe import adapt


def test_adapter_observation_statistics_do_not_mix_trials_or_people():
    rng=np.random.default_rng(7); inputs=rng.normal(size=(2,14,5120)).astype(np.float32)
    combined=adapt(inputs)
    np.testing.assert_array_equal(combined[:10],adapt(inputs[:1]))
    np.testing.assert_array_equal(combined[10:],adapt(inputs[1:]))
    shifted=inputs.copy(); shifted[1]*=100; shifted[1]+=700
    np.testing.assert_array_equal(adapt(shifted)[:10],combined[:10])
    assert combined.shape==(20,14,800)


def test_constant_signal_and_window_order_are_defined():
    assert np.isfinite(adapt(np.zeros((1,14,5120),dtype=np.float32))).all()
    time=np.arange(5120)/128
    signal=np.stack([np.sin(2*np.pi*(5+c)*time) for c in range(14)])[None].astype(np.float32)
    result=adapt(signal).transpose(1,0,2).reshape(14,8000)
    np.testing.assert_allclose(result.mean(1),0,atol=1e-6)
    np.testing.assert_allclose(result.std(1),1,atol=1e-6)
