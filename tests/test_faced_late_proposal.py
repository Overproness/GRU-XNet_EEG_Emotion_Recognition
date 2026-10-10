from gruxnet.faced_proposal_preprocessing_v2 import late_segment_bounds
import pytest


@pytest.mark.parametrize('rate',[250,1000])
def test_late_segment_never_uses_after_video_or_before_trial(rate):
    start,stop=late_segment_bounds(7.001,41.107,rate,100*rate)
    assert stop<=41.107*rate and start>=7.001*rate
    assert stop-start==32*rate
    assert start+2*rate>=7.001*rate and stop-2*rate<41.107*rate


def test_short_trial_or_recording_fails_without_borrowing_neighbor_samples():
    with pytest.raises(ValueError,match='own original video'):
        late_segment_bounds(10,41,250,100*250)
    with pytest.raises(ValueError,match='own original video'):
        late_segment_bounds(10,44,250,43*250)
