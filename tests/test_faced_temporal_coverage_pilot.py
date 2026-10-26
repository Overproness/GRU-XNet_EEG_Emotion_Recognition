"""Worker activation/path safeguards using manufactured metadata only."""
import pytest
from scripts.run_faced_temporal_coverage_pilot_2026_10_12 import (
    CAP_SECONDS, safe_raw_path, validate_activation,
)

def fixture():
    plan = {'signal_access_activated':False,'new_fits':0,'confirmation_access':False,
        'pilot':{'trial_count':8,'segment_count':24,'trials':[None]*8,'segments':[None]*24,
                 'counts_sample_values_planned':100}}
    activation = {'status':'active_source_only_quality_pilot','plan_sha256':'frozen',
        'original_rating_access':False,'cached_target_access':False,
        'validation_signal_access':False,'confirmation_access':False,'new_fits':0,
        'trial_cap':8,'segment_cap':24,'seconds_cap':CAP_SECONDS,'cpu_seconds_cap':CAP_SECONDS,'counts_sample_values_cap':100}
    return activation,plan

@pytest.mark.parametrize('field,value', [('original_rating_access',True),('cached_target_access',True),
    ('validation_signal_access',True),('confirmation_access',True),('new_fits',1),
    ('trial_cap',9),('segment_cap',25),('counts_sample_values_cap',101),('plan_sha256','tampered'),
    ('cpu_seconds_cap',CAP_SECONDS+1)])
def test_widened_activation_rejected(field,value):
    activation,plan = fixture(); activation[field]=value
    with pytest.raises(PermissionError):
        validate_activation(activation,plan,'frozen')

def test_missing_segment_rejected():
    activation,plan = fixture(); plan['pilot']['segments'].pop()
    with pytest.raises(PermissionError):
        validate_activation(activation,plan,'frozen')

@pytest.mark.parametrize('person,name',[('../sub001','data.bdf'),('sub-001/../sub099','data.bdf'),
    ('sub-001','After_remarks.mat'),('sub001','evt.bdf')])
def test_recording_path_scope_rejected(person,name):
    with pytest.raises(PermissionError):
        safe_raw_path(person,name)

def test_closed_activation_and_layout_pass():
    activation,plan = fixture(); validate_activation(activation,plan,'frozen')
    assert safe_raw_path('sub-001','data.bdf').name == 'data.bdf'
