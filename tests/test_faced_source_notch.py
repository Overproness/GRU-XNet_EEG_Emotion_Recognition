import numpy as np
import pytest
from gruxnet.faced_source_notch import notched_late_trial_windows
from gruxnet.faced_proposal_preprocessing_v2 import late_trial_windows


@pytest.mark.parametrize('rate',[250,1000])
def test_fixed_notch_preserves_ten_hz_and_removes_fifty_hz(rate):
    t=np.arange(32*rate)/rate
    values=np.zeros((30,len(t)))
    values[0]=10*np.sin(2*np.pi*10*t)+100*np.sin(2*np.pi*50*t)
    values[1]=-values[0]
    base=late_trial_windows(values,rate);notched=notched_late_trial_windows(values,rate)
    core_t=2+np.arange(28*128)/128
    def amplitude(x,f):
        signal=x[:,0,:].reshape(-1)
        return abs(2*np.mean(signal*np.exp(-2j*np.pi*f*core_t)))
    assert notched.shape==(7,30,512)
    assert abs(amplitude(notched,10)-10)<.15
    assert amplitude(notched,50)<amplitude(base,50)*.001
    np.testing.assert_allclose(notched.sum(axis=1),0,atol=1e-4)


def test_notch_rejects_wrong_geometry_before_filtering():
    with pytest.raises(ValueError,match='declared'):
        notched_late_trial_windows(np.zeros((30,8000)),1000)
