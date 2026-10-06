"""Fetch the immutable, hash-pinned author audit sources; never overwrite changes."""
from argparse import ArgumentParser
import json
from pathlib import Path
import hashlib
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def download(root):
    manifest = json.loads((REPO/'results/development/eegnet_author_audit_2026-10-06/download_manifest.json').read_text())
    if manifest['commit'] != '4a512e503198db2010848813ead9afbf8cd54c97':
        raise ValueError('Unexpected pinned author commit')
    if set(manifest['files']) != {'EEGModels.py', 'LICENSE.txt', 'README.md'}:
        raise ValueError('Unexpected author-source filenames')
    output = root/'eegnet_author_audit_2026-10-06'; output.mkdir(parents=True, exist_ok=True)
    for name, record in manifest['files'].items():
        expected_url = f'https://raw.githubusercontent.com/vlawhern/arl-eegmodels/{manifest["commit"]}/{name}'
        if record['url'] != expected_url: raise ValueError('Unexpected download location')
        path = output/name
        if path.exists():
            data = path.read_bytes()
        else:
            data = urllib.request.urlopen(expected_url, timeout=30).read()
        if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['bytes']:
            raise ValueError(f'Changed or incorrect author source:{name}')
        if not path.exists(): path.write_bytes(data)
    metadata = output/'download_manifest.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != manifest: raise ValueError('Changed existing manifest')
    else: metadata.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'author_commit': manifest['commit'], 'files_checked': 3}))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    args = parser.parse_args(); download(args.root.resolve())
