"""Publish only verified derived probabilities/declarations; retain EEG/coefficients locally."""
import json
from pathlib import Path
import shutil
import sys
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.deap_baseline_folds_v2 import STUDY, MODELS, cell_id
from gruxnet.preprocessing_diagnostic_v2 import atomic, sha


def export(output):
    destination = REPO/'results/development'/STUDY
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    def copy(source):
        relative = source.relative_to(output); target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != sha(source): shutil.copyfile(source, target)
        if sha(target) != sha(source): raise ValueError('Export changed bytes')
        records[relative.as_posix()] = sha(target)
    for name in ('plan.json', 'config.json', 'feasibility.json', 'progress.json', 'verification.json',
                 'comparison.json', 'alignment.json', 'analysis_verification.json', 'FINDINGS.md'):
        if (output/name).is_file(): copy(output/name)
    if (output/'inputs/prepared.json').exists(): copy(output/'inputs/prepared.json')
    if (output/'plan.json').exists():
        plan = json.loads((output/'plan.json').read_text())
        for job in plan['jobs']:
            for model in MODELS:
                folder = output/'cells'/cell_id(job)/model
                if not (folder/'verification.json').exists(): continue
                for name in ('selection.json', 'record.json', 'verification.json',
                             'candidates_source.csv', 'selected_predictions.csv'):
                    if (folder/name).exists(): copy(folder/name)
    for folder in ('aggregate', 'plots'):
        if (output/folder).exists():
            for source in sorted((output/folder).iterdir()):
                if source.is_file() and source.suffix in ('.csv', '.png', '.svg'): copy(source)
    atomic(destination/'export_manifest.json', {'files': [{'file': name, 'sha256': checksum} for name, checksum in sorted(records.items())],
        'scope': 'Verified compact derived probabilities and source selection, complete fixed-grid analysis when finished. Raw EEG, per-trial feature arrays and scaler/model coefficient arrays remain local.'})
    return destination
