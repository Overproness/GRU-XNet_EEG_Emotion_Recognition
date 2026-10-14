"""Bounded source-only EEGNet learning remedy; no original-file I/O.

One metadata-fixed complete crossed cell, one schedule, two initializations.
The existing author-checked engine is imported without changing old experiments.
"""
from __future__ import annotations

import math
import time
import hashlib
import json
import re
from pathlib import Path
import numpy as np

from .faced_classical_contract import fit_source_ridge, predict_ridge

MAIN_EPOCHS = 360
MAIN_CHECKPOINTS = (120, 240, 360)
CAPACITY_EPOCHS = 200
CAPACITY_CHECKPOINTS = (100, 200)
RATE = .001
SEEDS = (20261101, 20261102)
PERMUTATION_SEED = 20261014
CONTEXT_ALPHA = 100.
CHECKPOINT_ATOL = 1e-4
CHECKPOINT_RTOL = 1e-5


def cache_metadata_projection(rows):
    fields = ('subject', 'vid', 'family', 'acquisition_group', 'rotation', 'model', 'arm', 'alpha', 'oof_fold')
    values = [[row.get(field, '') for field in fields] for row in rows]
    return hashlib.sha256(json.dumps(values, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def guard_source_original_path(workspace, raw_root, person, name, allowed_files):
    """Source technical-file allowlist and ancestor guard before content access."""
    if not re.fullmatch(r'sub-\d{3}', person) or name not in ('data.bdf', 'evt.bdf') or (person, name) not in set(allowed_files):
        raise PermissionError('Original path outside exact source file allowlist')
    workspace, raw_root = Path(workspace).absolute(), Path(raw_root).absolute()
    target = raw_root / person.replace('sub-', 'sub') / name
    if not raw_root.is_relative_to(workspace) or not target.is_relative_to(raw_root):
        raise PermissionError('Original root/path outside declared workspace')
    # lstat inspects metadata only. Cloud hydration tags are not path redirects;
    # Windows junction and symbolic-link tags are rejected explicitly.
    for current in (target, *target.parents):
        info = current.lstat()
        if current.is_symlink() or getattr(info, 'st_reparse_tag', 0) in (0xA0000003, 0xA000000C):
            raise PermissionError('Symbolic link/junction ancestor in source original path')
    resolved_workspace = workspace.resolve(strict=True)
    resolved_root, resolved_target = raw_root.resolve(strict=True), target.resolve(strict=True)
    if resolved_root != raw_root or resolved_target != target or not resolved_root.is_relative_to(resolved_workspace) or not resolved_target.is_relative_to(resolved_root):
        raise PermissionError('Resolved original path escaped or aliased the allowlist')
    return resolved_target


class PreparationBudget:
    """Cooperative clocks checked during hashing, selected reads and preparation."""
    def __init__(self, wall_cap=7200., cpu_cap=7200., wall_clock=time.monotonic, cpu_clock=time.process_time):
        self.wall_cap, self.cpu_cap = wall_cap, cpu_cap
        self.wall_clock, self.cpu_clock = wall_clock, cpu_clock
        self.wall_started, self.cpu_started = wall_clock(), cpu_clock()

    def check(self):
        if self.wall_clock() - self.wall_started >= self.wall_cap:
            raise TimeoutError('Fixed source preparation wall budget exhausted')
        if self.cpu_clock() - self.cpu_started >= self.cpu_cap:
            raise TimeoutError('Fixed source preparation CPU budget exhausted')

    def receipt(self):
        self.check()
        return {'wall_seconds': self.wall_clock() - self.wall_started,
                'cpu_seconds': self.cpu_clock() - self.cpu_started}


def complete_cell(cells, missing_pairs):
    """Choose using identities/previous quality only, never ratings or scores."""
    missing = {tuple(p) for p in missing_pairs}
    if [c['cell'] for c in cells] != list(range(10)):
        raise PermissionError('Changed metadata order of crossed source cells')
    for cell in cells:
        train = [tuple(p) for p in cell['training']]
        held = [tuple(p) for p in cell['held']]
        if not ((set(train) | set(held)) & missing):
            return cell
    raise PermissionError('No complete source cell; no implicit availability remedy')


def guard_cell(cell, metadata, source_people, missing_pairs):
    train = tuple(tuple(p) for p in cell['training'])
    held = tuple(tuple(p) for p in cell['held'])
    if len(train) != 448 or len(held) != 84 or len(set(train)) != 448 or len(set(held)) != 84:
        raise PermissionError('Changed bounded source cell denominator')
    tp, tv = {p[0] for p in train}, {p[1] for p in train}
    hp, hv = {p[0] for p in held}, {p[1] for p in held}
    if (len(tp), len(tv), len(hp), len(hv)) != (56, 8, 14, 6):
        raise PermissionError('Changed source person/title geometry')
    if tp & hp or tv & hv or not (tp | hp) <= set(source_people):
        raise PermissionError('Source person/title overlap or protected person')
    if set(train) != {(p, v) for p in tp for v in tv} or set(held) != {(p, v) for p in hp for v in hv}:
        raise PermissionError('Nonrectangular bounded cell')
    chosen = set(train) | set(held)
    if chosen & {tuple(p) for p in missing_pairs}:
        raise PermissionError('Incomplete cell requires a separately qualified policy')
    for pair in chosen:
        if pair not in metadata or metadata[pair].get('role') != 'source':
            raise PermissionError('Non-source metadata before data callback')
    if {metadata[p]['family'] for p in train} & {metadata[p]['family'] for p in held}:
        raise PermissionError('Source/held catalogue family overlap')
    tiny = tuple(p for p in train if p[0] in sorted(tp)[:4])
    if len(tiny) != 32:
        raise PermissionError('Changed metadata-first tiny capacity subset')
    return train, held, tiny


def guarded_source_targets(rows, metadata, selected, *, return_receipt=False):
    """Global metadata barrier before any selected target numeric conversion.

    The canonical cache legitimately contains old development-validation rows;
    their target/prediction fields are never converted. Unknown/protected rows
    fail in the first pass before even an earlier eligible row is converted.
    """
    selected = set(selected)
    if not selected or not selected <= set(metadata):
        raise PermissionError('Unknown selected source identity')
    if any(metadata[p].get('role') != 'source' for p in selected):
        raise PermissionError('Selected target scope contains non-source roles')
    eligible = []
    seen = set()
    models = {'global_median', 'coarse_valence_median', 'category_median', 'context_only',
              'spectral_eeg_only', 'context_plus_spectral_eeg', 'familiar_video_median'}
    arms = {'source_fit', 'source_oof', 'validation_unseen', 'validation_familiar'}
    row_count = 0
    for row in rows:
        row_count += 1
        pair = row['subject'], int(row['vid'])
        if str(row['vid']) != str(pair[1]) or str(row['rotation']) not in ('0', '1', '2'):
            raise PermissionError('Noncanonical cache video/rotation token')
        if pair not in metadata or row.get('family') != metadata[pair]['family']:
            raise PermissionError('Unknown/protected cache identity before numeric access')
        if row.get('model') not in models or row.get('arm') not in arms or int(row['rotation']) not in (0, 1, 2):
            raise PermissionError('Unknown canonical cache model/arm/rotation')
        linear = row['model'] in ('context_only', 'spectral_eeg_only', 'context_plus_spectral_eeg')
        allowed_alpha = {'0.1', '1.0', '10.0', '100.0', '1000.0'} if linear else {''}
        if row.get('alpha', '') not in allowed_alpha:
            raise PermissionError('Changed canonical model/alpha token')
        if row['arm'] == 'source_oof' and row.get('oof_fold') not in ('0', '1', '2', '3', '4'):
            raise PermissionError('Invalid source OOF fold token')
        if row['arm'] != 'source_oof' and row.get('oof_fold', '') != '':
            raise PermissionError('Unexpected OOF fold outside source OOF arm')
        role = 'source' if row['arm'].startswith('source_') else 'validation'
        if metadata[pair].get('role') != role or row.get('acquisition_group') != metadata[pair]['acquisition_group']:
            raise PermissionError('Canonical cache role/acquisition mismatch')
        identity = (int(row['rotation']), row['model'], row['arm'], row.get('alpha', ''), row.get('oof_fold', ''), pair)
        if identity in seen:
            raise PermissionError('Duplicate canonical cache row')
        seen.add(identity)
        if pair in selected:
            eligible.append((pair, row))
    if {p for p, _ in eligible} != selected:
        raise PermissionError('Incomplete selected source target coverage')
    targets = {}
    for pair, row in eligible:
        value = float(row['target'])
        if not math.isfinite(value) or not 0 <= value <= 7:
            raise ValueError('Invalid source original-scale target')
        if pair in targets and targets[pair] != value:
            raise ValueError('Repeated canonical source target differs')
        targets[pair] = value
    receipt = {'metadata_rows_checked_before_numeric_conversion': row_count,
               'selected_target_conversion_calls_including_duplicates': len(eligible),
               'selected_unique_target_pairs': len(targets),
               'duplicate_selected_target_conversion_calls': len(eligible) - len(targets),
               'unselected_target_conversion_calls': 0, 'prediction_conversion_calls': 0}
    return (targets, receipt) if return_receipt else targets


def source_offsets(training, folds, features, targets):
    """Four participant-excluded folds, all constructed inside outer source."""
    training = tuple(training)
    people = {p[0] for p in training}
    groups = [tuple(p for p in fold if p in people) for fold in folds]
    groups = [g for g in groups if g]
    flattened = [p for g in groups for p in g]
    if len(groups) != 4 or len(flattened) != 56 or len(set(flattened)) != 56 or set(flattened) != people:
        raise PermissionError('Require four disjoint inner source-person groups')
    full = fit_source_ridge(training, features, targets, CONTEXT_ALPHA)
    oof, ledger = {}, []
    for fold, group in enumerate(groups):
        source = [p for p in training if p[0] not in group]
        held = [p for p in training if p[0] in group]
        if {p[0] for p in source} & {p[0] for p in held} or len(source) != 336 or len(held) != 112:
            raise PermissionError('Inner person separation/denominator changed')
        model = fit_source_ridge(source, features, targets, CONTEXT_ALPHA)
        values = predict_ridge(model, held, features)
        oof.update(zip(held, values))
        ledger.append({'fold': fold, 'training': source, 'held': held, 'source_identity_sha256': model['source_identity_sha256'],
                       'mean': model['scaler'].mean_.tolist(), 'scale': model['scaler'].scale_.tolist(),
                       'coefficient': model['estimator'].coef_.tolist(), 'intercept': float(model['estimator'].intercept_)})
    if set(oof) != set(training):
        raise PermissionError('Incomplete source-only OOF offsets')
    return full, oof, ledger


def capacity_targets(tiny, targets, *, permuted):
    values = np.asarray([targets[p] for p in tiny], dtype=float)
    if values.shape != (32,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 7)):
        raise ValueError('Invalid tiny source targets')
    order = np.random.default_rng(PERMUTATION_SEED).permutation(32) if permuted else np.arange(32)
    return dict(zip(tiny, values[order])), order.tolist(), float(np.median(values))


def learning_flags(baseline, mse240, mse360, seed_maes):
    if len(mse240) != 2 or len(mse360) != 2 or len(seed_maes) != 2:
        raise ValueError('Complete two-seed learning panel required')
    a = np.asarray([baseline, *mse240, *mse360, *seed_maes], dtype=float)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Invalid learning metrics')
    flags = []
    if any(v > .8 * baseline for v in mse360):
        flags.append('insufficient_source_learning')
    if any(last < .9 * middle for middle, last in zip(mse240, mse360)):
        flags.append('budget_limited_convergence')
    if abs(seed_maes[0] - seed_maes[1]) > .15:
        flags.append('held_seed_divergence')
    if max(mse360) > 2 * max(min(mse360), 1e-8):
        flags.append('source_seed_divergence')
    return flags


def verify_reloaded_state(model):
    """Geometry is enforced by strict loading; finite state and norm checks here."""
    import torch
    values = model.state_dict()
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite reloaded checkpoint tensor')
    if torch.sqrt(torch.sum(model.spatial.weight.square(), dim=2)).max() > 1.000001:
        raise ValueError('Reloaded spatial max-norm violation')
    if torch.sqrt(torch.sum(model.head.weight.square(), dim=1)).max() > .250001:
        raise ValueError('Reloaded head max-norm violation')


def verify_replayed_predictions(expected, replayed):
    expected, replayed = np.asarray(expected, dtype=float), np.asarray(replayed, dtype=float)
    if expected.shape != replayed.shape or expected.ndim != 1 or len(expected) < 2:
        raise ValueError('Incomplete independently reloaded forward panel')
    if not np.isfinite(expected).all() or not np.isfinite(replayed).all():
        raise ValueError('Nonfinite independently reloaded forward value')
    if not np.allclose(expected, replayed, atol=CHECKPOINT_ATOL, rtol=CHECKPOINT_RTOL):
        raise ValueError('Independently reloaded checkpoint predictions differ')
    return float(np.max(np.abs(expected - replayed)))
