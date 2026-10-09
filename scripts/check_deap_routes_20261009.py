"""Bounded public first-party DEAP route recheck; no credentials or form submission."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

URLS = (
    'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/',
    'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html',
    'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html',
    'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/metadata_xls.zip',
    'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/data/data_preprocessed_python.zip',
    'https://anaxagoras.eecs.qmul.ac.uk/request.php?dataset=DEAP',
    'https://eecs.qmul.ac.uk/mmv/datasets/deap/download.html',
    'https://www.epfl.ch/labs/mmspg/research/page-58317-en-html/bci-2/bci_datasets/emotion_dataset/',
)


def check(output, ca_bundle=None):
    if output.exists():
        raise FileExistsError('Preserve earlier route checks')
    output.mkdir(parents=True)

    def probe(item):
        index, url = item
        record = {'url': url, 'checked_utc': datetime.now(timezone.utc).isoformat()}
        verify = str(ca_bundle) if ca_bundle and '.eecs.qmul.ac.uk/' in url else True
        record['tls_verification'] = 'existing audited CA bundle' if isinstance(verify, str) else 'system trust'
        try:
            with requests.get(url, timeout=(5, 12), stream=True, verify=verify) as response:
                record.update(status=response.status_code, final_url=response.url,
                              content_type=response.headers.get('Content-Type'),
                              www_authenticate=response.headers.get('WWW-Authenticate'),
                              redirects=[{'url': r.url, 'status': r.status_code,
                                          'location': r.headers.get('Location')} for r in response.history])
                chunks, remaining = [], 131072
                for chunk in response.iter_content(chunk_size=16384):
                    chunks.append(chunk[:remaining])
                    remaining -= min(len(chunk), remaining)
                    if not remaining:
                        break
                body = b''.join(chunks)
                filename = f'route_{index:02d}.body'
                (output/filename).write_bytes(body)
                record.update(body_file=filename, bounded_bytes=len(body),
                              body_sha256=hashlib.sha256(body).hexdigest(),
                              contains_deap=b'DEAP' in body or b'deap' in body)
        except requests.RequestException as error:
            record.update(error_type=type(error).__name__, error=str(error)[:600])
        return record

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(probe, enumerate(URLS, 1)))
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'routes': records,
              'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'first_party_recording_authentication_completed': False,
              'scope': 'Public service/access-route availability only. Bounded responses are not EEG archives, and no recording-content comparison follows from an HTTP status or third-party mirror agreement.',
              'ca_bundle_sha256': hashlib.sha256(ca_bundle.read_bytes()).hexdigest() if ca_bundle else None}
    (output/'route_checks.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ca-bundle', type=Path)
    args = parser.parse_args()
    check(args.output.resolve(), args.ca_bundle)
