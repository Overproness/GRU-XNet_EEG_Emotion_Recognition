"""Inspect author-hosted MAT identity maps, never numerical emotion ratings.

whosmat validates the complete variable schema before loadmat is allowed. Only
integer 'trial' and 'vid' vectors are accepted; any score/signal variable closes
the gate. This does not assert that an intended trial plan was actually recorded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import re

from scipy.io import loadmat, whosmat

from collect_emo_mc_historical_events import ROOT, PUBLIC
from qualify_emo_mc import REPO, sha, small_get, write_json


def identity_vectors(body, context):
    schema = whosmat(io.BytesIO(body))
    if len(schema) != 2 or {name for name, _, _ in schema} != {'trial', 'vid'}:
        raise ValueError('Not an identity-only MAT schema; numerical payload unopened')
    sizes = {shape for _, shape, _ in schema}
    if len(sizes) != 1 or any(len(shape) != 2 or shape[0] != 1 or not 1 <= shape[1] <= 21 for shape in sizes):
        raise ValueError('Unexpected identity-vector shape')
    if any(dtype not in ('int32', 'int64') for _, _, dtype in schema):
        raise ValueError('Unexpected identity dtype')
    data = loadmat(io.BytesIO(body), variable_names=['trial', 'vid'])
    trial, code = (data[name].ravel().tolist() for name in ('trial', 'vid'))
    lo, hi = (1, 21) if context == 'ima' else (22, 42)
    if not all(isinstance(v, int) and 1 <= v <= 21 for v in trial):
        raise ValueError('Unexpected context trial ordinals')
    if not all(isinstance(v, int) and lo <= v <= hi for v in code):
        raise ValueError('Unexpected material identity codes')
    if len(set(trial)) != len(trial) or len(set(code)) != len(code):
        raise ValueError('Duplicate identity entries')
    return dict(trial=trial, vid=code)


def declare():
    entries = []
    listing_hashes = {}
    for context in ('ima', 'vid'):
        listing = ROOT / f'source_metadata/sciencedb_remarks_{context}.json'
        listing_hashes[context] = sha(listing.read_bytes())
        rows = json.loads(listing.read_text(encoding='utf-8'))['data']
        assert len(rows) == 60
        for row in rows:
            match = re.fullmatch(r'/V6/Remarks_new/' + context + r'/sub_(\d+)_output_remark.mat', row['path'])
            assert match and row['dir'] is False and 0 < row['size'] <= 10000
            entries.append(dict(participant=f'sub-{int(match[1]):02d}', context=context,
                                archive_path=row['path'], file_id=row['id'], bytes=row['size'],
                                url='https://china.scidb.cn/download?fileId=' + row['id']))
    result = dict(archive_doi='10.57760/sciencedb.14025', archive_version='V6',
                  collector_sha256=sha(Path(__file__).read_bytes()), listing_sha256=listing_hashes,
                  reservation_sha256=sha((REPO / 'results/development/emo_mc_qualification_2026-10-10/reservation.json').read_bytes()),
                  schema_allowlist=['trial', 'vid'], sources=entries,
                  allowed_scope='Integer trial ordinals and stimulus codes only, including reserved identities. No EEG samples, physiological features, numerical rating variables or embeddings.',
                  prior_metadata_pilots=['ima:sub-37','vid:sub-37','ima:sub-60','vid:sub-60'],
                  prior_pilot_values='Only identity arrays were decoded after schema validation; all rating values remain unopened.',
                  research_question_changed=False, no_reallocation=True)
    path = PUBLIC / 'identity_plan.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        write_json(path, result)
    print(json.dumps({'declared_identity_only_files': len(entries), 'reserved_outcomes_remain_closed': True}))


def collect():
    plan_path = PUBLIC / 'identity_plan.json'
    plan = json.loads(plan_path.read_text())
    assert plan['collector_sha256'] == sha(Path(__file__).read_bytes())

    def one(source):
        path = ROOT / 'identity_maps' / source['context'] / (source['participant'] + '.mat')
        if path.exists():
            body = path.read_bytes()
        else:
            body = small_get(source['url'], limit=source['bytes'])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        if len(body) != source['bytes']:
            raise ValueError('Incorrect published file size')
        try:
            identities = identity_vectors(body, source['context'])
        except ValueError as exc:
            return dict(**source, sha256=sha(body), status='quarantined', error=str(exc))
        return dict(**source, sha256=sha(body), status='identity_schema_verified', identities=identities)

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(one, plan['sources']))
    write_json(ROOT / 'identity_collection.json', records)
    public = [{k: v for k, v in r.items() if k != 'identities'} for r in records]
    write_json(PUBLIC / 'identity_authentication.json',
               dict(plan_sha256=sha(plan_path.read_bytes()),
                    collection_sha256=sha((ROOT / 'identity_collection.json').read_bytes()), sources=public,
                    schema_verified=sum(r['status'] == 'identity_schema_verified' for r in records),
                    rating_values_decoded=0, waveform_samples_decoded=0, models_fitted=0,
                    source_authentication='First-party DOI/version listing and HTTPS file-ID retrieval, exact listed byte sizes and recorded SHA-256. The listing supplies no independent file checksum.',
                    scope='Identity-vector schema verification; not execution-log or rating-value certification'))
    print(json.dumps({'identity_files': len(records), 'schema_verified': sum(r['status'] == 'identity_schema_verified' for r in records),
                      'ratings_or_samples_decoded': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('declare', 'collect'))
    args = parser.parse_args()
    (declare if args.command == 'declare' else collect)()
