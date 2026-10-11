"""Fixed quality gate and lazy prediction fallback; no data/target readers or fits.

This conservative trial-level policy does not reconstruct failed electrodes.
The caller must authenticate identities/files before supplying QC or predictors.
"""
from math import isfinite
from numbers import Real
from .faced_proposal_preprocessing import COMMON_CHANNELS
from .faced_source_quality import QUALITY_RULES

POLICY_ID='faced-quality-context-fallback-v1'
RULES={k:QUALITY_RULES[k] for k in (
    'raw_rail_fraction_above','raw_identical_adjacent_fraction_above',
    'processed_channel_sd_below_uv','processed_channel_sd_above_uv',
    'processed_channel_peak_to_peak_above_uv')}
RULES['processed_49_51_to_1_45_power_ratio_above']=.10


def _number(value, name, upper=None):
    if isinstance(value,bool) or not isinstance(value,Real) or not isfinite(value) or value<0 or (upper is not None and value>upper):
        raise ValueError('Invalid quality metric: '+name)
    return float(value)


def eeg_availability(quality, *, technical_passed=True, repair_capacity_exceeded=False):
    """Recompute existing QC rules, ignoring identity, ratings and saved flag text."""
    if type(technical_passed) is not bool or type(repair_capacity_exceeded) is not bool:
        raise ValueError('Require explicit technical/capacity booleans')
    reasons=[]
    if not technical_passed:
        return {'policy_id':POLICY_ID,'eeg_available':False,
                'reasons':[{'channel':'all','reason':'numerically_unavailable_eeg'}]}
    if not isinstance(quality,dict) or 'channels' not in quality:
        raise ValueError('Authenticated numerical EEG requires its QC receipt')
    channels=quality['channels']
    if [c['channel'] for c in channels]!=list(COMMON_CHANNELS) or quality.get('valid_windows')!=7 or quality.get('windows_dropped')!=0:
        raise ValueError('QC channel/window geometry changed')
    if quality.get('finite_processed_values') is not True:
        reasons.append({'channel':'all','reason':'nonfinite_processed_values'})
    if repair_capacity_exceeded:reasons.append({'channel':'all','reason':'repair_capacity_exceeded'})
    for c in channels:
        rail=_number(c['raw_rail_fraction'],'raw_rail_fraction',1)
        same=_number(c['raw_identical_adjacent_fraction'],'raw_identical_adjacent_fraction',1)
        sd=_number(c['processed_sd_uv'],'processed_sd_uv')
        ptp=_number(c['processed_peak_to_peak_uv'],'processed_peak_to_peak_uv')
        line=_number(c['processed_49_51_to_1_45_power_ratio'],'processed_line_ratio')
        flags=[]
        if rail>RULES['raw_rail_fraction_above']:flags.append('adc_rail')
        if same>RULES['raw_identical_adjacent_fraction_above']:flags.append('stationary_counts')
        if sd<RULES['processed_channel_sd_below_uv']:flags.append('low_processed_variation')
        if sd>RULES['processed_channel_sd_above_uv']:flags.append('high_processed_variation')
        if ptp>RULES['processed_channel_peak_to_peak_above_uv']:flags.append('high_processed_excursion')
        if line>RULES['processed_49_51_to_1_45_power_ratio_above']:flags.append('residual_50hz_power')
        reasons.extend({'channel':c['channel'],'reason':f} for f in flags)
    return {'policy_id':POLICY_ID,'eeg_available':not reasons,'reasons':reasons}


def predict_with_quality(decision, family, *, eeg_predictor=None, context_predictor=None, source_median=None):
    """Never invoke an EEG predictor for an unavailable trial; retain its outcome."""
    if family not in ('context_only','spectral_eeg_only','context_plus_spectral_eeg'):
        raise ValueError('Unknown declared prediction family')
    if (decision.get('policy_id')!=POLICY_ID or type(decision.get('eeg_available')) is not bool
            or not isinstance(decision.get('reasons'),list)
            or decision['eeg_available']==bool(decision['reasons'])):
        raise ValueError('Invalid frozen availability decision')
    fallback=False;eeg_used=False
    if family=='context_only':
        branch='context_model';value=context_predictor()
    elif decision['eeg_available']:
        branch='eeg_model' if family=='spectral_eeg_only' else 'joint_model'
        value=eeg_predictor();eeg_used=True
    elif family=='context_plus_spectral_eeg':
        branch='context_fallback';value=context_predictor();fallback=True
    else:
        branch='source_global_median_fallback';value=source_median;fallback=True
    value=_number(value,'prediction',7)
    return {'prediction':value,'branch':branch,'fallback':fallback,'eeg_used':eeg_used}


def require_source_quality(records, quality_loader, expected_pairs):
    """Guard a future source set before targets/fitting; inference fallback cannot bypass it."""
    expected=set(expected_pairs);actual=[(r['subject'],r['vid']) for r in records]
    if set(actual)!=expected or len(actual)!=len(expected):raise PermissionError('Source identities/coverage differ')
    failures=[]
    for r,pair in zip(records,actual):
        decision=eeg_availability(quality_loader(pair) if r['technical_passed'] else None,
            technical_passed=r['technical_passed'],repair_capacity_exceeded=r.get('repair_capacity_exceeded',False))
        if not decision['eeg_available']:failures.append({'subject':pair[0],'vid':pair[1],'reasons':decision['reasons']})
    return {'passed':not failures,'source_trials':len(records),'failed_trials':len(failures),
            'failures':failures,'fitting_held':bool(failures),'policy_id':POLICY_ID}
