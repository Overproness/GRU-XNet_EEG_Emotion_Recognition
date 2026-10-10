"""Explicit byte-budget addendum for four larger historical event tables.

The original collector and declaration remain unchanged. No schema or protected
outcome gate changes: only the exact byte sizes already in the pinned Git tree
replace the original 20,000-byte retrieval ceiling.
"""
import argparse
import json
from pathlib import Path

import collect_emo_mc_historical_events as original
from qualify_emo_mc import sha, small_get, write_json


def declaration():
    plan_path = original.PUBLIC / 'plan.json'
    plan = json.loads(plan_path.read_text())
    result = dict(original_plan_sha256=sha(plan_path.read_bytes()),
                  original_plan_commit='db1a6b93e',
                  adapter_sha256=sha(Path(__file__).read_bytes()),
                  reason='Four table sizes in the pinned Git tree exceed the initial 20,000-byte ceiling; the initial collector stopped before decoding these tables.',
                  exact_byte_limits={s['path']: s['size'] for s in plan['historical_sources']},
                  unchanged_schema=original.FIELDS,
                  rating_values_decoded=0, waveform_samples_decoded=0,
                  models_fitted=0, research_question_changed=False)
    path = original.PUBLIC / 'retrieval_addendum.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        write_json(path, result)
    return result


def collect():
    addendum = declaration()
    plan = json.loads((original.PUBLIC / 'plan.json').read_text())
    limits = {s['sha']: s['size'] for s in plan['historical_sources']}

    def exact_get(url, limit, expected_blob):
        assert expected_blob in limits and limit == 20000
        return small_get(url, limit=limits[expected_blob], expected_blob=expected_blob)

    assert addendum['adapter_sha256'] == sha(Path(__file__).read_bytes())
    original.small_get = exact_get
    original.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('declare', 'collect'))
    args = parser.parse_args()
    if args.command == 'declare':
        print(json.dumps({'larger_tables': sum(v > 20000 for v in declaration()['exact_byte_limits'].values()),
                          'maximum_bytes': max(declaration()['exact_byte_limits'].values())}))
    else:
        collect()
