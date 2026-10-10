"""Recover pinned, deleted timing metadata without opening ratings or EEG samples.

The schema must contain ONLY onset, duration, stim_type and trial_type. Original
tables stay private. File identities are checked against the frozen Git tree.
Historical records are evidence to reconcile, not an override of current data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path

from qualify_emo_mc import REPO, WORK, git_blob, sha, small_get, write_json

PIN = 'aea206971c265988c8ca16d14a69031367434d85'
ROOT = WORK / 'publication_runs/emo_mc_join_resolution_2026-10-10'
PUBLIC = REPO / 'results/development/emo_mc_join_resolution_2026-10-10'
FIELDS = ['onset', 'duration', 'stim_type', 'trial_type']


def parse_events(body):
    # Validate bytes BEFORE decoding any rows; fail closed on extra columns.
    if body.splitlines()[0].decode('ascii').split('\t') != FIELDS:
        raise ValueError('Unexpected event schema; no rows decoded')
    rows = []
    for row in csv.DictReader(io.StringIO(body.decode('ascii')), delimiter='\t'):
        if set(row) != set(FIELDS) or any(v is None for v in row.values()):
            raise ValueError('Malformed event row')
        code = int(row['stim_type'])
        onset, duration = float(row['onset']), float(row['duration'])
        if row['trial_type'] not in ('ima', 'vid', 'fade', 'rating', 'n/a'):
            raise ValueError('Unknown event role')
        if onset < 0 or duration != 0 or not 0 <= code <= 100:
            raise ValueError('Unexpected timing or trigger code')
        rows.append(dict(onset_s=onset, duration_s=duration, trigger_code=code,
                         documented_role=row['trial_type']))
    return rows


def declare():
    historical = json.loads((ROOT / 'source_metadata/historical_tree_older.json').read_text())
    current = json.loads((WORK / 'publication_runs/data_target_review_2026-10-10/source_metadata/emo_mc_tree.response').read_text())
    older = {x['path']: x for x in historical['tree']}
    latest = {x['path']: x for x in current['tree']}
    assert historical['sha'] == PIN and not historical['truncated'] and not current['truncated']
    sources = [{k: x[k] for k in ('path', 'mode', 'sha', 'size')}
               for x in historical['tree'] if x['path'].endswith('_events.tsv')]
    assert len(sources) == 60 and all(x['mode'] == '100644' for x in sources)
    identities = {}
    for suffix in ('.edf', '_beh.tsv'):
        paths = [p for p in older if p.endswith(suffix)]
        assert all(p in latest and older[p]['sha'] == latest[p]['sha'] for p in paths)
        identities[suffix] = dict(compared=len(paths), unchanged_git_blobs=len(paths))
    plan = dict(historical_git_pin=PIN, current_git_pin='83b5b7f28dea9964a5389509548219a27b755cc4',
                collector_sha256=sha(Path(__file__).read_bytes()),
                historical_tree_sha256=sha((ROOT / 'source_metadata/historical_tree_older.json').read_bytes()),
                reservation_sha256=sha((REPO / 'results/development/emo_mc_qualification_2026-10-10/reservation.json').read_bytes()),
                historical_sources=sources, unchanged_file_pointers=identities,
                rating_values_decoded=0, waveform_samples_decoded=0, models_fitted=0,
                allowed_scope='Timing/trigger metadata for all participants, including reserved identities. No target values or waveform samples.',
                no_reallocation=True, research_question_changed=False)
    if (PUBLIC / 'plan.json').exists():
        assert json.loads((PUBLIC / 'plan.json').read_text()) == plan
    else:
        write_json(PUBLIC / 'plan.json', plan)
    print(json.dumps({'declared_event_tables': len(sources), 'unchanged_file_pointers': identities}))


def collect():
    plan = json.loads((PUBLIC / 'plan.json').read_text())
    assert plan['collector_sha256'] == sha(Path(__file__).read_bytes())
    assert plan['historical_git_pin'] == PIN

    def one(source):
        path = ROOT / 'historical_events' / source['path']
        url = 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds005540/' + PIN + '/' + source['path']
        if path.exists():
            body = path.read_bytes()
            if git_blob(body) != source['sha']:
                raise ValueError('Existing historical source differs')
        else:
            body = small_get(url, limit=20000, expected_blob=source['sha'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        assert len(body) == source['size']
        events = parse_events(body)
        return dict(path=source['path'], url=url, git_blob=source['sha'],
                    bytes=len(body), sha256=sha(body), event_rows=len(events), events=events)

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(one, plan['historical_sources']))
    write_json(ROOT / 'historical_collection.json', records)
    public_records = [{k: v for k, v in record.items() if k != 'events'} for record in records]
    write_json(PUBLIC / 'historical_authentication.json',
               dict(plan_sha256=sha((PUBLIC / 'plan.json').read_bytes()),
                    collection_sha256=sha((ROOT / 'historical_collection.json').read_bytes()),
                    sources=public_records, source_tables=60,
                    rating_values_decoded=0, waveform_samples_decoded=0, models_fitted=0,
                    scope='Authenticated historical zero-duration markers. No actual stimulus-end codes or trial-material/rating-row IDs are supplied.'))
    print(json.dumps({'authenticated_event_tables': len(records),
                      'total_event_rows': sum(r['event_rows'] for r in records),
                      'ratings_or_samples_decoded': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('declare', 'collect'))
    args = parser.parse_args()
    (declare if args.command == 'declare' else collect)()
