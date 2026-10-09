"""Download pinned author CBraMod sources/weights without importing author code."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import urllib.request

GITHUB_REVISION = 'b9e961003214326972c567eff390e75b0287e32a'
HF_REVISION = '500543c7e30bda1b22bfd51a49301b238dee21fd'
FILES = ('LICENSE', 'README.md', 'requirements.txt', 'models/__init__.py',
         'models/cbramod.py', 'models/criss_cross_transformer.py',
         'models/model_for_faced.py', 'models/model_for_seedv.py',
         'preprocessing/README.md', 'preprocessing/preprocessing_faced.py',
         'preprocessing/preprocessing_SEEDV.py',
         'preprocessing/preprocessing_tueg_for_pretraining.py',
         'datasets/pretraining_dataset.py', 'datasets/faced_dataset.py',
         'datasets/seedv_dataset.py', 'pretrain_main.py', 'finetune_main.py',
         'pretrained_weights/README.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Preserve downloaded audit assets')
    out.mkdir(parents=True)
    records = []
    for name in FILES:
        url = f'https://raw.githubusercontent.com/wjq-learning/CBraMod/{GITHUB_REVISION}/{name}'
        path = out/'author'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'GRUXNet-publication-audit'})
        with urllib.request.urlopen(request, timeout=30) as stream:
            data = stream.read(2000000)
            if stream.read(1):
                raise ValueError('Unexpectedly large author source')
        path.write_bytes(data)
        records.append({'file': path.relative_to(out).as_posix(), 'sha256': sha(path), 'url': url})
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(token=False).model_info('weighting666/CBraMod', revision=HF_REVISION, files_metadata=True)
    if info.sha != HF_REVISION:
        raise ValueError('Unexpected checkpoint revision')
    for name in ('README.md', 'pretrained_weights.pth'):
        metadata = next(item for item in info.siblings if item.rfilename == name)
        path = Path(hf_hub_download('weighting666/CBraMod', name, revision=HF_REVISION,
                                  token=False, local_dir=out/'checkpoint'))
        actual = sha(path)
        if path.stat().st_size != metadata.size or (metadata.lfs and actual != metadata.lfs.sha256):
            raise ValueError('Checkpoint integrity mismatch')
        records.append({'file': path.relative_to(out).as_posix(), 'sha256': actual,
                        'size_bytes': path.stat().st_size,
                        'url': f'https://huggingface.co/weighting666/CBraMod/blob/{HF_REVISION}/{name}'})
    manifest = {'github_revision': GITHUB_REVISION, 'checkpoint_revision': HF_REVISION,
                'files': records, 'execution': 'None: author source review precedes imports',
                'corpus': 'Authors document TUEG; full training membership not independently certified'}
    (out/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'files': len(records), 'manifest': str(out/'download_manifest.json')}))


if __name__ == '__main__':
    main()
