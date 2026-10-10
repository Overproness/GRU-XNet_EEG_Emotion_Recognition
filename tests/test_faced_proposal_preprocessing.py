import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('preprocessing',Path(__file__).resolve().parents[1]/'gruxnet/faced_proposal_preprocessing.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def header():
    labels=p.COMMON_CHANNELS+['A1','A2']
    return {'sample_rates_hz':[250.0]*32,'columns':{'labels':labels,'units':['uV']*32,
        'physical_min':['-375000']*32,'physical_max':['375000']*32,
        'digital_min':['-8388608']*32,'digital_max':['8388607']*32}}


def test_adc_endpoints_and_unit_repair_preserve_physical_scale():
    counts=np.tile(np.asarray([-8388608,0,8388607],dtype=np.int32),(32,1))
    h=header()
    values=p.calibrate_common_counts(counts,'sub-000',h)
    assert values.shape==(30,3)
    np.testing.assert_allclose(values[:,0],-375000)
    np.testing.assert_allclose(values[:,2],375000)
    h['columns']['units']=['?V']*32
    repaired=p.calibrate_common_counts(counts,'sub-004',h)
    np.testing.assert_array_equal(values,repaired)
    with pytest.raises(ValueError,match='Undocumented'):
        p.calibrate_common_counts(counts,'sub-000',h)


def test_both_channel_layouts_have_identical_scalp_order():
    modern=p.COMMON_CHANNELS+['HEOR','HEOL']
    reverse={v:k for k,v in p.ALIASES.items()}
    legacy=[reverse.get(v,v) for v in p.COMMON_CHANNELS]+['A1','A2']
    assert p.common_indices(legacy)==p.common_indices(modern)==list(range(30))
    with pytest.raises(ValueError,match='Duplicate'):
        p.common_indices(modern+['T3'])


@pytest.mark.parametrize('sampling_rate',[250,1000])
def test_common_reference_invariance_and_physical_sine_scale(sampling_rate):
    t=np.arange(32*sampling_rate)/sampling_rate
    values=np.zeros((30,len(t)))
    values[0]=10*np.sin(2*np.pi*10*t)
    values[1]=-values[0]
    original=p.trial_windows(values,sampling_rate)
    shifted=p.trial_windows(values+100*np.sin(2*np.pi*3*t)[None,:]+400,sampling_rate)
    assert original.shape==(7,30,512)
    np.testing.assert_allclose(original,shifted,atol=1e-5)
    assert abs(np.std(original[:,0])-10/np.sqrt(2))<.15
    np.testing.assert_allclose(original.sum(axis=1),0,atol=1e-4)


def test_resampling_does_not_fold_high_frequency_into_low_band():
    t=np.arange(8000)/250
    values=np.zeros((30,len(t)))
    values[0]=10*np.sin(2*np.pi*100*t);values[1]=-values[0]
    transformed=p.trial_windows(values,250)
    assert np.std(transformed[:,0])<.1


def test_wrong_geometry_and_nonfinite_data_fail_without_fitted_repair():
    with pytest.raises(ValueError,match='32-second'):
        p.trial_windows(np.zeros((30,7999)),250)
    data=np.zeros((30,8000));data[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        p.trial_windows(data,250)
