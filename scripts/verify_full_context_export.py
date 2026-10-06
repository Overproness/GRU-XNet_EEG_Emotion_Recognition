"""Recompute full-control numbers from the public probability export, without EEG."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.data import digest, sha256, write_json
from gruxnet.full_context_controls_v2 import ALL, ARMS, analyze
from gruxnet.grouped_material_controls import probabilities
from scripts import within_video_alignment as alignment


def verify(root):
    root = root.resolve()
    alignment.ROOT = root
    alignment.PLAN = root / 'within_video_alignment_plan_2026-10-06.json'
    primary_count = supplement_count = rows_count = points_count = 0
    input_hashes = {}
    for dataset in ('SEEDIV', 'DEAP'):
        folder = root / f'full_context_v2_{dataset.lower()}'
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        for relative, expected in config['source_sha256'].items():
            if sha256(REPO / relative) != expected:
                raise ValueError(f'Changed fitting source: {relative}')
        saved = json.loads((folder / 'comparison.json').read_text(encoding='utf-8'))
        if analyze(dataset, folder, write=False) != saved:
            raise ValueError(f'{dataset} primary contrasts failed public recomputation')
        primary_count += len(saved['contrasts'])
        reference_metadata = None
        for model in ALL:
            for arm in ARMS:
                path = folder / f'predictions_{model}_{arm}.csv'
                input_hashes[str(path.relative_to(root))] = sha256(path)
                frame = pd.read_csv(path)
                if len(frame) != (1080 if dataset == 'SEEDIV' else 1264):
                    raise ValueError('Incomplete predeclared trial population')
                metadata = frame[['trial_id', 'subject_id', 'source_session',
                                  'material_rotation', 'fold', 'material_key',
                                  'label', 'original_label']]
                if reference_metadata is None:
                    reference_metadata = metadata
                else:
                    pd.testing.assert_frame_equal(reference_metadata, metadata, check_exact=True)
                if frame.trial_id.duplicated().any():
                    raise ValueError('Duplicate out-of-fold trial')
                p = probabilities(frame)
                if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError('Invalid probability')
                np.testing.assert_allclose(p.sum(1), 1., atol=1e-6, rtol=0)
                rows_count += len(frame)
                tasks = ('coarse3', 'binary') if dataset == 'SEEDIV' else ('binary',)
                for task in tasks:
                    target = frame.label.to_numpy(dtype=int)
                    selected_probability = p
                    if dataset == 'SEEDIV' and task == 'binary':
                        keep = frame.original_label.ne(0).to_numpy()
                        target = frame.original_label.to_numpy()[keep] == 3
                        positive = p[keep, 2] / np.maximum(p[keep, 1:].sum(1), 1e-12)
                        selected_probability = np.stack([1-positive, positive], axis=1)
                    target = target.astype(int)
                    ba = balanced_accuracy_score(target, selected_probability.argmax(1))
                    losses = -np.log(np.clip(selected_probability[np.arange(len(target)), target], 1e-12, 1))
                    balanced_loss = np.mean([losses[target == c].mean()
                                             for c in range(selected_probability.shape[1])])
                    expected = saved['models'][model][arm][task]
                    np.testing.assert_allclose([ba, balanced_loss],
                                               [expected['balanced_accuracy'], expected['balanced_log_loss']],
                                               atol=1e-10, rtol=0)
                    points_count += 2
        supplemental = alignment.analyze(dataset, write=False)
        expected_supplement = json.loads((root / 'within_video_alignment' /
                                          f'comparison_{dataset.lower()}.json').read_text(encoding='utf-8'))
        if supplemental != expected_supplement:
            raise ValueError(f'{dataset} alignment contrasts failed public recomputation')
        supplement_count += len(supplemental['contrasts'])
    return {'passed': True, 'probability_rows_checked': rows_count,
            'independently_calculated_point_metrics': points_count,
            'primary_contrasts_recomputed': primary_count,
            'supplemental_contrasts_recomputed': supplement_count,
            'source_sha256': sha256(Path(__file__)),
            'input_hashes': input_hashes,
            'input_digest': digest(input_hashes),
            'scope': 'Public-export fitting-source bindings, complete matched OOF metadata, independent sklearn accuracy and class-mean log-loss calculations, and exact recomputation of the declared primary and EEG-exchange analyses. No EEG, checkpoint replay, optimization-trajectory reproduction or first-party signal authentication.'}


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'results' / 'development')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'input_hashes'}, indent=2))
