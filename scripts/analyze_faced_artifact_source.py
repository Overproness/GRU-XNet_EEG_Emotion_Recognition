"""Independent cached artifact/QC replay; never opens recordings or ratings."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import numpy as np
from gruxnet.faced_classical_contract import context_features
from gruxnet.faced_source_quality import QUALITY_RULES
from gruxnet.faced_proposal_preprocessing import common_indices
from scripts.run_faced_artifact_control import PUBLIC, PRIVATE, RAW_EVIDENCE, check, load, digest, save, array_digest, BASE, PREVIOUS
from gruxnet.faced_artifact_control import RULES, COMMON_CHANNELS, qualified_weights

OUTPUT = PUBLIC/'source_analysis'


def engineering_flags(channel):
    """Recompute all six rules from saved numerical metrics, not stored flags."""
    r = QUALITY_RULES; flags = []
    if channel['raw_rail_fraction'] > r['raw_rail_fraction_above']: flags.append('adc_rail')
    if channel['raw_identical_adjacent_fraction'] > r['raw_identical_adjacent_fraction_above']: flags.append('stationary_counts')
    if channel['processed_sd_uv'] < r['processed_channel_sd_below_uv']: flags.append('low_processed_variation')
    if channel['processed_sd_uv'] > r['processed_channel_sd_above_uv']: flags.append('high_processed_variation')
    if channel['processed_peak_to_peak_uv'] > r['processed_channel_peak_to_peak_above_uv']: flags.append('high_processed_excursion')
    if channel['processed_49_51_to_1_45_power_ratio'] > .10: flags.append('residual_50hz_power')
    return flags


def check_access(row, access, header):
    start, stop = row['sample_interval']
    sizes = [int(s) for s in header['columns']['samples_per_record']]
    if len(set(sizes)) != 1: raise ValueError('Unsupported cached channel geometry')
    per_record = sizes[0]; record_bytes = sum(sizes)*3
    expected = []
    for record in range(start//per_record, (stop-1)//per_record+1):
        lo, hi = max(start, record*per_record), min(stop, (record+1)*per_record)
        for channel in row['channel_indices']:
            expected.append({'channel_index': channel, 'start_sample': lo, 'stop_sample': hi,
                'offset': header['header_bytes']+record*record_bytes+3*sum(sizes[:channel])+3*(lo-record*per_record),
                'bytes': 3*(hi-lo)})
    expected_digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    count = 30*(stop-start)
    if access['ranges'] != expected or access['range_ledger_sha256'] != expected_digest or row['range_ledger_sha256'] != expected_digest:
        raise ValueError('Selective-access offsets differ from independent reconstruction')
    if count != access['sample_values_decoded'] or count != row['samples_decoded'] or access['signal_bytes_decoded'] != 3*count:
        raise ValueError('Selected-count coverage differs')
    if access['extra_boundary_samples_decoded'] != 0 or not access['all_counts_scalar_reference_checked']:
        raise ValueError('Unexpected boundary access or missing scalar reference')
    return len(expected)


def analyze(pilot=False):
    programme, activation, expected, validation = check()
    if pilot:expected=[(r['subject'],r['vid']) for r in activation['pilot_pairs']]
    execution = load(PUBLIC/'execution.json')
    rows = load(PUBLIC/'source_records.json')['records']
    if len(rows) != len(expected) or {(r['subject'], r['vid']) for r in rows} != set(expected):
        raise ValueError('Source audit incomplete or identity geometry changed')
    headers = {r['subject']: r['data.bdf'] for r in load(RAW_EVIDENCE/'headers.json')['records']}
    joins = {r['subject']: r for r in load(RAW_EVIDENCE/'joins_numeric_projection.json')['records']}
    catalogue = {r['clip']: r for r in load(REPO/'results/development/faced_codebooks_2026-10-10/audit.json')['stimuli']}
    flags = []; trial_rows = []; channel_counts = Counter(); group_counts = {}; ledgers = 0
    total_windows = 0; max_calibration = 0.; caches = {}
    previous={(r['subject'],r['vid']):r for r in load(PREVIOUS/'source_records.json')['records']}
    repaired=[]; unchanged=0; maximum_weight_discrepancy=0.
    for row in rows:
        subject, vid = row['subject'], row['vid']; stem = f'{subject}_clip{vid:02d}'
        if not row['technical_passed']: raise ValueError('Numerical source failure needs a separate incomplete-audit diagnosis')
        header = headers[subject]; rate = int(header['sample_rates_hz'][0])
        span = [s for s in joins[subject]['spans'] if s['vid'] == vid]
        if len(span) != 1: raise ValueError('Source identity join is not unique')
        stop = math.floor(span[0]['end_seconds']*rate); start = stop-32*rate
        if row['sample_interval'] != [start,stop] or row['trial'] != span[0]['trial'] or row['native_rate_hz'] != rate:
            raise ValueError('Source interval/identity differs from marker-derived proposal')
        if row['channel_indices'] != common_indices(header['columns']['labels']):
            raise ValueError('Source channel selection differs from common-scalp contract')
        paths = {'access': PRIVATE/(stem+'_access.json'), 'quality': PRIVATE/(stem+'_quality.json'), 'features': PRIVATE/(stem+'_features.npz')}
        for kind, key in [('access','access_ledger_file_sha256'), ('quality','quality_file_sha256'), ('features','feature_cache_sha256')]:
            actual = digest(paths[kind])
            if actual != row[key]: raise ValueError('Private source cache changed')
            caches[paths[kind].name] = actual
        access = load(paths['access']); quality = load(paths['quality'])
        artifact_path=PRIVATE/(stem+'_artifact.json');artifact=load(artifact_path)
        if digest(artifact_path)!=row['artifact_file_sha256']:raise ValueError('Private repair receipt changed')
        caches[artifact_path.name]=digest(artifact_path)
        bad=[i for i,(sd,ptp) in enumerate(zip(artifact['pre_car_sd_uv'],artifact['pre_car_peak_to_peak_uv']))
            if sd>RULES['pre_car_core_sd_above_uv'] or ptp>RULES['pre_car_core_peak_to_peak_above_uv']]
        names=[COMMON_CHANNELS[i] for i in bad]
        if len(artifact['pre_car_sd_uv'])!=30 or len(artifact['pre_car_peak_to_peak_uv'])!=30 or bad!=artifact['detected_indices'] or names!=row['detected_channels']:
            raise ValueError('Repair detection differs from fixed signal thresholds')
        if row['count_array_sha256']!=previous[subject,vid]['count_array_sha256']:
            raise ValueError('Original counts changed from preserved baseline')
        if not artifact['donors_unchanged'] or artifact['trials_or_windows_dropped'] or artifact['author_code_reproduction']:
            raise ValueError('Repair scope differs')
        if not bad:
            unchanged+=1
            if row['processed_window_sha256']!=previous[subject,vid]['processed_window_sha256'] or row['spectral_feature_sha256']!=previous[subject,vid]['spectral_feature_sha256']:
                raise ValueError('No-detection feature/window bytes differ from preserved baseline')
        elif len(bad)<=3:
            weights,donors,error=qualified_weights(tuple(bad));maximum_weight_discrepancy=max(maximum_weight_discrepancy,error)
            if not np.array_equal(weights,np.asarray(artifact['interpolation_weights'])) or artifact['repaired_channels']!=names or row['repaired_channels']!=names:
                raise ValueError('Repair weights or channel identities differ')
            repaired.append({'subject':subject,'vid':vid,'repaired_channels':names,
                'before_max_sd_uv':previous[subject,vid]['processed_sd_uv_min_median_max'][2],
                'after_max_sd_uv':row['processed_sd_uv_min_median_max'][2],
                'before_max_peak_to_peak_uv':previous[subject,vid]['maximum_processed_peak_to_peak_uv'],
                'after_max_peak_to_peak_uv':row['maximum_processed_peak_to_peak_uv'],
                'previous_flagged':bool(previous[subject,vid]['review_flags'])})
        ledgers += check_access(row, access, headers[subject])
        if len(quality['channels']) != 30 or not quality['finite_processed_values'] or quality['valid_windows'] != 7 or quality['windows_dropped'] != 0:
            raise ValueError('Quality geometry changed')
        reconstructed = []
        if len(bad)>3:reconstructed.append({'channel':'multiple','flag':'repair_capacity_exceeded'})
        for c in quality['channels']:
            derived = engineering_flags(c)
            if derived != c['review_flags']: raise ValueError('Stored channel flag differs from independent threshold calculation')
            for flag in derived:
                reconstructed.append({'channel': c['channel'], 'flag': flag})
                channel_counts[c['channel']] += 1
                flags.append({'subject': subject, 'vid': vid, 'acquisition_group': row['acquisition_group'],
                    'channel': c['channel'], 'flag': flag, 'processed_sd_uv': c['processed_sd_uv'],
                    'processed_peak_to_peak_uv': c['processed_peak_to_peak_uv'],
                    'raw_rail_fraction': c['raw_rail_fraction'], 'raw_identical_adjacent_fraction': c['raw_identical_adjacent_fraction'],
                    'processed_line_ratio': c['processed_49_51_to_1_45_power_ratio']})
        if reconstructed != row['review_flags']: raise ValueError('Public trial review flags differ from cached metrics')
        with np.load(paths['features'], allow_pickle=False) as cache:
            if set(cache.files) != {'context','spectral'} or cache['context'].shape != (13,) or cache['spectral'].shape != (150,):
                raise ValueError('Feature cache schema differs')
            if not np.isfinite(cache['spectral']).all(): raise ValueError('Source feature nonfinite')
            metadata = {**catalogue[vid], 'trial': row['trial'], 'acquisition_group': row['acquisition_group']}
            if not np.array_equal(cache['context'], context_features(metadata)):
                raise ValueError('Context features differ from frozen metadata-only definition')
            if array_digest(cache['context']) != row['context_feature_sha256'] or array_digest(cache['spectral']) != row['spectral_feature_sha256']:
                raise ValueError('Source feature byte fingerprint differs')
        total_windows += 7; max_calibration = max(max_calibration, row['calibration_max_absolute_error_uv'])
        raw_sd = [c['raw_sd_uv'] for c in quality['channels']]
        strongest = quality['channels'][int(np.argmax(raw_sd))]
        trial_rows.append({'subject': subject, 'vid': vid, 'acquisition_group': row['acquisition_group'],
            'native_rate_hz': row['native_rate_hz'], 'flagged_channels': '|'.join(sorted({f['channel'] for f in reconstructed})),
            'review_flags': '|'.join(sorted({f['flag'] for f in reconstructed})),
            'channel_rule_flags': len(reconstructed),
            'processed_max_sd_uv': row['processed_sd_uv_min_median_max'][2],
            'processed_max_peak_to_peak_uv': row['maximum_processed_peak_to_peak_uv'],
            'processed_max_line_ratio': row['maximum_processed_line_ratio'],
            'raw_max_sd_channel': strongest['channel'], 'raw_max_to_median_sd_ratio': max(raw_sd)/max(float(np.median(raw_sd)),1e-30),
            'window_median_channel_sd_uv_min': min(quality['window_median_channel_sd_uv']),
            'window_median_channel_sd_uv_max': max(quality['window_median_channel_sd_uv'])})
        group = group_counts.setdefault(row['acquisition_group'], {'trials': 0, 'flagged_trials': 0, 'people': set(), 'flagged_people': set()})
        group['trials'] += 1; group['people'].add(subject)
        if reconstructed: group['flagged_trials'] += 1; group['flagged_people'].add(subject)
    flagged_rows = [r for r in trial_rows if r['review_flags']]
    per_flag = {}
    for name in sorted({f['flag'] for f in flags}):
        selected = [f for f in flags if f['flag'] == name]
        per_flag[name] = {'channel_trial_events': len(selected), 'trials': len({(f['subject'],f['vid']) for f in selected}),
                          'participants': len({f['subject'] for f in selected})}
    for group in group_counts.values():
        group['participants'] = len(group.pop('people')); group['flagged_participants'] = len(group.pop('flagged_people'))
    summary = {'source_trials': len(rows), 'source_participants':len({r['subject'] for r in rows}), 'source_clips': sorted({r['vid'] for r in rows}),
        'interpolated_trials':len(repaired),'interpolated_channel_trial_events':sum(len(r['repaired_channels']) for r in repaired),
        'unchanged_trial_count':unchanged,'interpolation_details':repaired,'maximum_independent_mne_weight_discrepancy':maximum_weight_discrepancy,
        'feature_shapes': [13,150], 'source_feature_cache_count': len(rows),
        'processed_windows': total_windows, 'windows_or_trials_dropped': 0,
        'selected_sample_values_decoded_and_scalar_checked': sum(r['samples_decoded'] for r in rows),
        'selective_range_entries_independently_reconstructed': ledgers,
        'maximum_calibration_discrepancy_uv': max_calibration,
        'flagged_trials': len(flagged_rows), 'flagged_participants': len({r['subject'] for r in flagged_rows}),
        'flagged_clips': sorted({r['vid'] for r in flagged_rows}),
        'flagged_trial_details': flagged_rows, 'rule_counts': per_flag, 'channel_rule_counts': dict(channel_counts),
        'acquisition_groups': group_counts, 'all_processed_line_ratios_maximum': max(r['processed_max_line_ratio'] for r in trial_rows),
        'source_quality_passed':not flagged_rows, 'execution_status': execution['status'],
        'original_ratings_decoded': execution['original_ratings_decoded'], 'original_ridge_fits': execution['original_ridge_fits'],
        'validation_trials_opened': execution['validation_trials_prepared'], 'protected_outcomes_or_waveforms_opened': 0,
        'raw_arrays_saved': 0, 'research_question_changed': False,
        'limits': ['Engineering thresholds are not validated artifact labels; exceeding them is not proof of bad participants or absent EEG information.',
                   'Source-only amplitude/rail/stationarity checks do not establish artifact freedom, physical timing or neural origin.',
                   'One separately declared source-informed interpolation control; no threshold, trial eligibility or target changes. The original held experiment remains preserved.',
                   'No incremental-prediction or conference-contribution evidence without the gated controls and independent confirmation.']}
    if execution['status'] == 'held_by_source_quality' and (summary['source_quality_passed'] or execution['original_ratings_decoded'] or execution['original_ridge_fits'] or execution['validation_trials_prepared']):
        raise ValueError('Independent source gate and execution access disagree')
    return summary, trial_rows, flags, caches


def main(mode,pilot=False):
    global PUBLIC,PRIVATE,OUTPUT
    if pilot:PUBLIC=BASE/'pilot';PRIVATE=PRIVATE/'pilot';OUTPUT=PUBLIC/'source_analysis'
    summary, trials, flags, caches = analyze(pilot)
    if mode == 'analyze':
        save(OUTPUT/'summary.json', summary)
        save(OUTPUT/'private_cache_manifest.json', {'sha256': caches, 'raw_or_target_files_read': 0})
        for name, rows in [('trial_quality.csv', trials), ('flagged_channels.csv', flags)]:
            if not rows: continue
            path = OUTPUT/name
            with path.open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        files = [Path(__file__), REPO/'tests/test_faced_classical_source_analysis.py', BASE/'activation_plan.json', PUBLIC/'source_records.json', PUBLIC/'source_gate.json',
                 PUBLIC/'execution.json', OUTPUT/'summary.json', OUTPUT/'private_cache_manifest.json', OUTPUT/'trial_quality.csv']
        if flags: files.append(OUTPUT/'flagged_channels.csv')
        save(OUTPUT/'verification.json', {'bindings': {path.relative_to(REPO).as_posix(): digest(path) for path in files},
             'quality_thresholds_recomputed':len(trials)*30*8, 'selective_range_entries_reconstructed': summary['selective_range_entries_independently_reconstructed'],
             'raw_recordings_or_ratings_reread': False, 'private_cache_objects_verified': len(caches)})
    else:
        if load(OUTPUT/'summary.json') != summary or load(OUTPUT/'private_cache_manifest.json')['sha256'] != caches:
            raise ValueError('Independent source analysis differs')
        for relative, expected in load(OUTPUT/'verification.json')['bindings'].items():
            if digest(REPO/relative) != expected: raise ValueError('Source analysis binding changed')
    print({k: summary[k] for k in ('source_trials','flagged_trials','flagged_participants','rule_counts',
          'source_quality_passed','original_ratings_decoded','original_ridge_fits')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('analyze','verify'));parser.add_argument('--pilot',action='store_true')
    args = parser.parse_args(); main(args.mode,args.pilot)
