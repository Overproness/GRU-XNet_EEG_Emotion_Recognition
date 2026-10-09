"""Export compact declared evidence only; EEG and candidate weights remain local."""
import json
from pathlib import Path
import shutil
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.preprocessing_diagnostic import STUDY, sha, atomic


def export(output):
    destination = REPO/'results/development'/STUDY
    destination.mkdir(parents=True, exist_ok=True)
    records = []

    def copy(source):
        relative = source.relative_to(output)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(source) != sha(target):
            raise ValueError('Export byte check failed')
        records.append({'file': relative.as_posix(), 'sha256': sha(target)})

    for name in ('plan.json', 'config.json', 'progress.json', 'verification.json', 'summary.json', 'FINDINGS.md'):
        if (output/name).is_file():
            copy(output/name)
    for folder in sorted((output/'panels').glob('*')):
        for name in ('panel.json', 'trials.csv'):
            copy(folder/name)
    for category in ('fits', 'classical'):
        for folder in sorted((output/category).glob('*')):
            if not (folder/'verification.json').is_file():
                continue
            record = json.loads((folder/'record.json').read_text())
            for name in ('record.json', 'verification.json', 'history.json', *record['artifact_sha256']):
                source = folder/name
                if source.is_file() and source.suffix in ('.json', '.csv'):
                    copy(source)
    atomic(destination/'export_manifest.json', {'files': records,
           'scope': 'Compact source-only development evidence; exact byte hashes. Native EEG, baseline waveforms, normalizer arrays and candidate model weights excluded.'})
    return destination
