"""Synthetic-only complete public-grid audit; no model training or EEG."""
import json
from pathlib import Path
import shutil
from uuid import uuid4
import numpy as np
import pandas as pd
from scripts.analyze_cbramod_readout_boundary_adapter import api, PUBLIC, REPO
from scripts.analyze_cbramod_learning import sha, write, criterion
from gruxnet.cbramod_adaptation import sampling
from gruxnet.cbramod_learning import exposure_records


def test_complete_public_grid_with_shared_validation_people_and_exact_anchors():
    namespace = api()
    fixture = REPO.parent/'publication_runs'/f'cbramod_readout_public_test_{uuid4().hex}'
    repository = fixture/'synthetic_repo'; public = repository/'results/development/synthetic_readout'
    public.mkdir(parents=True)
    write(fixture/'SYNTHETIC_ONLY.json', {'synthetic_only': True, 'task_fits': 0, 'EEG_accessed': False})
    plan = json.loads((PUBLIC/'plan.json').read_text())
    for name in plan['source_sha256']:
        target = repository/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, target)
    write(public/'plan.json', plan)
    selections = []
    for job in plan['jobs']:
        dataset, group, pretrained, head = [job[k] for k in ('dataset', 'group', 'pretrained', 'head')]
        classes = 2 if dataset == 'DEAP' else 3
        name = f'{dataset.lower()}_g{group}_{"pretrained" if pretrained else "random42"}_{head}'
        folder = public/'runs'/name; folder.mkdir(parents=True)
        tables = {}
        for role, subject in (('train', 'train_person'), ('validation_unseen', 'validation_person'), ('validation_familiar', 'validation_person')):
            tables[role] = pd.DataFrame({'trial_id': [f'{role}_{c}' for c in range(classes)],
                'subject_id': subject, 'material_key': [f'{"unseen" if role == "validation_unseen" else "training"}_{c}' for c in range(classes)],
                'original_label': np.arange(classes), 'label': np.arange(classes), 'role': role})
        candidates = []
        for step in (0, 200, 600, 1200):
            metrics = {}
            for role, table in tables.items():
                frame = table.copy(); p = np.full((classes, classes), 1/classes)
                for c in range(classes):
                    frame[f'p{c}'] = p[:, c]
                frame.to_csv(folder/f'step{step}_{role}.csv', index=False)
                metrics[role] = namespace['measures'](table.label.to_numpy(), p)
            candidates.append({'step': step, 'metrics': metrics})
        draws, windows, signature = sampling(tables['train'], np.arange(classes), 1200)
        reused = head == 'pooled_linear'
        history = [{'step': s, 'mean_last100_minibatch_loss': .7,
            **({} if reused else {'clipped_updates_last100': 0})} for s in range(100, 1201, 100)]
        write(folder/'history.json', history)
        record = {'job': job, 'plan_sha256': sha(public/'plan.json'), 'reused': reused,
            'candidates': candidates, 'initial_head_digest': f'{dataset}_{head}',
            'initial_encoder_digest': f'{dataset}_{group}_{pretrained}', 'sampling_digest': signature,
            'exposure': exposure_records(np.arange(classes), draws, windows, (0, 200, 600, 1200), 'long'),
            'artifact_sha256': {p.name: sha(p) for p in folder.iterdir()},
            'dropout_schema': [{'p': .1}]*61,
            'head_dropout_rng': {'p': .1, 'seed': 424243, 'index': 'seed + update; encoder global RNG restored'}}
        if reused:
            source = repository/'results/development/synthetic_prior/runs'/f'{dataset.lower()}_g{group}_{"pretrained" if pretrained else "random42"}_finetune_raw_dropout_on'
            source.mkdir(parents=True)
            for p in folder.iterdir():
                shutil.copyfile(p, source/p.name)
            write(source/'record.json', {'candidates': candidates})
            write(source/'verification.json', {'synthetic_only': True})
            record['source_record_sha256'] = sha(source/'record.json')
            record['source_verification_sha256'] = sha(source/'verification.json')
        write(folder/'record.json', record)
        write(folder/'verification.json', {'complete': True, 'record_sha256': sha(folder/'record.json'),
            'states_checked': 4, 'metric_sets': 12, 'synthetic_only': True})
        choices = candidates[1:]
        selections.append({'job': job, 'selected': choices[criterion(choices)], 'candidates': choices})
    write(public/'summary.json', {'development_only': True, 'outer_test_inferences': 0,
        'research_question_change_approved': False, 'selected': selections})
    write(public/'verification.json', {'complete': True, 'plan_sha256': sha(public/'plan.json'),
        'summary_sha256': sha(public/'summary.json'), 'synthetic_only': True})
    namespace['REPO'] = repository
    namespace['PRIOR'] = 'synthetic_prior'
    frames, points, discrepancy = namespace['collect'](public)
    assert len(frames['all_metrics']) == 288 and len(frames['selected_metrics']) == 72
    assert len(points) == 1080 and discrepancy == 0
    assert all(p['delta'] == 0 for p in points)
