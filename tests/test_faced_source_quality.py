import io
import numpy as np
import pytest
from gruxnet.faced_source_quality import decode_signed24, read_interval, quality_metrics
from gruxnet.faced_proposal_preprocessing import trial_windows


def encoded(values):
    return b''.join(int(v).to_bytes(3, 'little', signed=True) for v in values)


def test_signed_extremes_and_byte_order():
    values = [-8388608,-65537,-256,-1,0,1,255,65536,8388607]
    np.testing.assert_array_equal(decode_signed24(encoded(values), True), values)
    with pytest.raises(ValueError, match='Truncated'):
        decode_signed24(b'\x00\x01')


class Traced(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload); self.reads = []
    def read(self, size=-1):
        self.reads.append((self.tell(), size)); return super().read(size)


def fixture():
    values = np.asarray([[i*100+j for j in range(12)] for i in range(3)], dtype=np.int32)
    payload = b'X'*8+b''.join(encoded(values[c,r*4:(r+1)*4]) for r in range(3) for c in range(3))
    h = {'sample_width':3,'continuity':'BDF','header_bytes':8,'record_count':3,
         'columns':{'samples_per_record':['4']*3}}
    return values,payload,h


def test_partial_records_skip_other_channels_and_neighbor_samples():
    values,payload,h = fixture(); stream=Traced(payload)
    result, receipt = read_interval(stream,h,3,10,[2,0],True)
    np.testing.assert_array_equal(result,values[[2,0],3:10])
    assert receipt['sample_values_decoded']==14 and receipt['extra_boundary_samples_decoded']==0
    assert stream.reads==[(r['offset'],r['bytes']) for r in receipt['ranges']]
    assert all(r['channel_index'] in (2,0) and 3<=r['start_sample']<r['stop_sample']<=10 for r in receipt['ranges'])


@pytest.mark.parametrize('start,stop,channels',[(-1,3,[0]),(0,13,[0]),(3,3,[0]),(0,2,[0,0]),(0,2,[3])])
def test_invalid_read_boundaries_do_not_read(start,stop,channels):
    _,payload,h=fixture();stream=Traced(payload)
    with pytest.raises(ValueError): read_interval(stream,h,start,stop,channels)
    assert not stream.reads


def test_truncated_sample_range_fails_closed():
    _,payload,h=fixture()
    with pytest.raises(ValueError,match='Truncated'):
        read_interval(io.BytesIO(payload[:-1]),h,10,12,[2])


def test_mixed_rate_header_rejected_before_read():
    _,payload,h=fixture();h['columns']['samples_per_record']=['4','2','4']
    stream=Traced(payload)
    with pytest.raises(ValueError,match='equal'): read_interval(stream,h,0,3,[0])
    assert not stream.reads


def test_flat_channel_flags_are_descriptive_without_dropping_windows():
    counts=np.zeros((30,8000),dtype=np.int32);values=np.zeros_like(counts,dtype=float)
    windows=trial_windows(values,250);q=quality_metrics(counts,values,windows,250)
    assert len(q['flagged_channels'])==30 and q['windows_dropped']==0 and q['valid_windows']==7
    assert all('stationary_counts' in row['review_flags'] for row in q['channels'])
    assert q['finite_processed_values']
