"""Legible linear-axis renderings of immutable, audited conditioning results."""
from pathlib import Path
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from analyze_cbramod_conditioning import REPO, STUDY, sha, read, write, verify as verify_analysis

PUBLIC = REPO/'results/development'/STUDY
INPUTS = ('postfit_analysis/all_metrics.csv', 'postfit_analysis/contrasts.json', 'postfit_analysis/verification.json')


def generate():
    verify_analysis(PUBLIC)
    output = REPO.parent/'publication_runs'/STUDY/'linear_figures'
    destination = PUBLIC/'linear_figures'
    if output.exists() or destination.exists():
        raise FileExistsError('Preserve figure revision')
    output.mkdir()
    write(output/'declaration.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'input_sha256': {n: sha(PUBLIC/n) for n in INPUTS},
        'scope': 'Presentation-only revision with linear axes and visible numeric ticks. Same768loss points and320final validation effects; original five figure pairs/analysis preserved, no new outcome or fit.'})
    os.environ.setdefault('MPLCONFIGDIR', str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    rows = pd.read_csv(PUBLIC/INPUTS[0]); points = pd.DataFrame(read(PUBLIC/INPUTS[1])['contrasts'])
    panels = (('DEAP', 1), ('DEAP', 2), ('SEEDIV', 1), ('SEEDIV', 2))
    roles = ('train', 'validation_unseen', 'validation_familiar')
    colors = {(False, True): '#596776', (False, False): '#2463aa', (True, True): '#bd4c25', (True, False): '#348344'}
    curve_count = effect_count = 0
    def ticks(axis):
        axis.set_major_locator(MaxNLocator(nbins=5))
        formatter = ScalarFormatter(useOffset=False); formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    def save(fig, name):
        for extension in ('png', 'svg'):
            fig.savefig(output/f'{name}.{extension}', dpi=160)
        plt.close(fig)
    for pretrained in (True, False):
        for trainable in (False, True):
            fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
            for i, (dataset, group) in enumerate(panels):
                part = rows[(rows.dataset == dataset)&(rows.group == group)&(rows.pretrained == pretrained)&(rows.trainable == trainable)]
                for j, role in enumerate(roles):
                    ax = axes[i, j]
                    for (scaled, dropout), frame in part[part.role == role].groupby(['scaled', 'dropout']):
                        frame = frame.sort_values('step')
                        line, = ax.plot(frame.step, frame.balanced_log_loss, color=colors[(scaled, dropout)],
                            linestyle='-' if scaled else '--', marker='o', markersize=3,
                            label=('Standardized' if scaled else 'Raw')+(', dropout on' if dropout else ', dropout off'))
                        np.testing.assert_array_equal(line.get_xdata(), frame.step.to_numpy())
                        np.testing.assert_array_equal(line.get_ydata(), frame.balanced_log_loss.to_numpy())
                        curve_count += len(frame)
                    ax.axhline(np.log(2 if dataset == 'DEAP' else 3), color='#999999', linestyle=':', linewidth=.8)
                    ticks(ax.yaxis)
                    low, high = ax.get_ylim(); ax.set_ylim(max(0, low), high)
                    task = 'DEAP' if dataset == 'DEAP' else 'SEED-IV'
                    ax.set_title(f'{task}, group {group}: '+{'train': 'training', 'validation_unseen': 'unseen material', 'validation_familiar': 'familiar material'}[role])
                    ax.set_xticks((0, 200, 600, 1200)); ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Balanced log loss'); ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=9)
            name = ('pretrained' if pretrained else 'random')+'_'+('finetune' if trainable else 'frozen')
            fig.suptitle(('Pretrained' if pretrained else 'Random')+(' fine-tuned' if trainable else ' frozen')+': all matched source trajectories', fontsize=13)
            save(fig, name+'_learning')
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    names = ('normalization_dropout_on', 'normalization_dropout_off', 'dropout_off_raw', 'dropout_off_standardized', 'interaction')
    labels = ('Normalization: dropout on', 'Normalization: dropout off', 'Dropout off: raw', 'Dropout off: standardized', 'Interaction')
    for i, (dataset, group) in enumerate(panels):
        part = points[(points.scope == 'fixed_step')&(points.step == 1200)&(points.dataset == dataset)&(points.group == group)&(points.role != 'train')]
        for j, metric in enumerate(('balanced_log_loss', 'balanced_accuracy')):
            ax = axes[i, j]
            for pretrained in (True, False):
                for trainable in (False, True):
                    for role in roles[1:]:
                        block = part[(part.pretrained == pretrained)&(part.trainable == trainable)&(part.role == role)&(part.metric == metric)].set_index('contrast')
                        offset = (.10 if pretrained else -.10)+(.035 if trainable else -.035)+(.013 if role == roles[1] else -.013)
                        values = np.array([block.loc[n, 'delta'] for n in names])*(100 if j else 1)
                        line, = ax.plot(values, np.arange(5)+offset, linestyle='none', color='#2463aa' if pretrained else '#bd4c25',
                            marker='o' if trainable else 's', markersize=5, markerfacecolor='auto' if role == roles[1] else 'none',
                            label=('Pretrained' if pretrained else 'Random')+(' fine-tuned' if trainable else ' frozen')+(' unseen' if role == roles[1] else ' familiar'))
                        np.testing.assert_array_equal(line.get_xdata(), values); effect_count += len(values)
            ax.axvline(0, color='#777777', linestyle='--'); ticks(ax.xaxis); ax.grid(axis='x', alpha=.2); ax.invert_yaxis()
            ax.set_yticks(range(5), labels if j == 0 else ['']*5)
            ax.set_title(f'{"DEAP" if dataset == "DEAP" else "SEED-IV"}, group {group}: fixed update 1,200')
            ax.set_xlabel('Loss difference (lower better)' if j == 0 else 'Balanced accuracy difference (pp; higher better)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=8)
    fig.suptitle('Matched final-step effects: both validation roles; descriptive development points', fontsize=13)
    save(fig, 'matched_final_step_effects')
    if curve_count != 768 or effect_count != 320:
        raise ValueError('Figure coordinate coverage changed')
    write(output/'verification.json', {'complete': True, 'declaration_sha256': sha(output/'declaration.json'),
        'curve_points_checked': curve_count, 'effect_points_checked': effect_count,
        'artifact_sha256': {p.name: sha(p) for p in output.iterdir() if p.suffix in ('.png', '.svg')}})
    destination.mkdir()
    for path in output.iterdir():
        shutil.copyfile(path, destination/path.name)
    manifest = read(PUBLIC/'export_manifest.json')
    for item in manifest['files']:
        if sha(PUBLIC/item['file']) != item['sha256']:
            raise ValueError('Original export changed')
    manifest['files'] += [{'file': p.relative_to(PUBLIC).as_posix(), 'sha256': sha(p)} for p in sorted(destination.iterdir())]
    manifest['linear_figures_source_sha256'] = sha(Path(__file__))
    write(PUBLIC/'export_manifest.json', manifest)
    verify()


def verify():
    folder = PUBLIC/'linear_figures'; declaration = read(folder/'declaration.json'); proof = read(folder/'verification.json')
    if not proof['complete'] or sha(folder/'declaration.json') != proof['declaration_sha256'] or sha(Path(__file__)) != declaration['source_sha256']:
        raise ValueError('Changed figure declaration/source')
    for name, checksum in declaration['input_sha256'].items():
        if sha(PUBLIC/name) != checksum:
            raise ValueError('Changed figure inputs')
    for name, checksum in proof['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed figure bytes')
    print(json.dumps({'passed': True, 'linear_figure_pairs': 5, 'curve_points_checked': 768, 'effect_points_checked': 320,
                      'scope': 'Presentation-only revision; all original numerical analysis and figure bytes preserved.'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('generate', 'verify'))
    args = parser.parse_args(); (generate if args.action == 'generate' else verify)()
