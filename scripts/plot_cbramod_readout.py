"""Complete, linear-axis figures from independently audited readout results."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha, read, write
from scripts.analyze_cbramod_readout_boundary_adapter import api, PUBLIC

INPUTS = ('postfit_analysis/all_metrics.csv', 'postfit_analysis/contrasts.json', 'postfit_analysis/verification.json')


def generate():
    api()['verify'](PUBLIC)
    output = REPO.parent/'publication_runs'/PUBLIC.name/'figures'
    destination = PUBLIC/'figures'
    if output.exists() or destination.exists():
        raise FileExistsError('Preserve readout figure declaration')
    output.mkdir()
    write(output/'declaration.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': sha(Path(__file__)), 'input_sha256': {n: sha(PUBLIC/n) for n in INPUTS},
        'scope': 'All288loss coordinates and144final validation head/pretraining contrasts. Linear axes, no selected winning condition.'})
    os.environ.setdefault('MPLCONFIGDIR', str(REPO.parent/'publication_runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    rows = pd.read_csv(PUBLIC/INPUTS[0]); points = pd.DataFrame(read(PUBLIC/INPUTS[1])['contrasts'])
    panels = (('DEAP', 1), ('DEAP', 2), ('SEEDIV', 1), ('SEEDIV', 2))
    roles = ('train', 'validation_unseen', 'validation_familiar')
    names = ('pooled_linear', 'pooled_mlp', 'flattened_mlp')
    labels = {'pooled_linear': 'Pooled linear (exact control)', 'pooled_mlp': 'Pooled two-layer MLP', 'flattened_mlp': 'Flattened two-layer MLP'}
    colors = {'pooled_linear': '#596776', 'pooled_mlp': '#2463aa', 'flattened_mlp': '#bd4c25'}
    curves = effects = 0
    def ticks(axis):
        axis.set_major_locator(MaxNLocator(nbins=5))
        formatter = ScalarFormatter(useOffset=False); formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    def save(fig, name):
        for ext in ('png', 'svg'):
            fig.savefig(output/f'{name}.{ext}', dpi=160)
        plt.close(fig)
    for pretrained in (True, False):
        fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
        for i, (dataset, group) in enumerate(panels):
            for j, role in enumerate(roles):
                ax = axes[i, j]
                block = rows[(rows.dataset == dataset)&(rows.group == group)&(rows.pretrained == pretrained)&(rows.role == role)]
                for head in names:
                    part = block[block['head'] == head].sort_values('step')
                    line, = ax.plot(part.step, part.balanced_log_loss, marker='o', markersize=3,
                        color=colors[head], linestyle='--' if head == 'pooled_linear' else '-', label=labels[head])
                    np.testing.assert_array_equal(line.get_xdata(), part.step.to_numpy())
                    np.testing.assert_array_equal(line.get_ydata(), part.balanced_log_loss.to_numpy())
                    curves += len(part)
                ax.axhline(np.log(2 if dataset == 'DEAP' else 3), color='#999999', linestyle=':', linewidth=.8)
                ticks(ax.yaxis); low, high = ax.get_ylim(); ax.set_ylim(max(0, low), high)
                ax.set_xticks((0, 200, 600, 1200)); ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Balanced log loss')
                ax.set_title(f'{"DEAP" if dataset == "DEAP" else "SEED-IV"}, group {group}: '+{'train': 'training', 'validation_unseen': 'unseen material', 'validation_familiar': 'familiar material'}[role])
                ax.grid(alpha=.2)
        handles, legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend, loc='outside lower center', ncol=3, fontsize=9)
        fig.suptitle(('Pretrained' if pretrained else 'Random42')+': all matched fine-tuning trajectories', fontsize=13)
        save(fig, ('pretrained' if pretrained else 'random')+'_readout_learning')
    for kind in ('head', 'pretraining'):
        fig, axes = plt.subplots(4, 2, figsize=(14, 11), constrained_layout=True)
        descriptions = ('Pooled MLP − linear', 'Flattened MLP − linear', 'Flattened − pooled MLP') if kind == 'head' else tuple(labels[n] for n in names)
        comparisons = (('pooled_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_linear'), ('flattened_mlp', 'pooled_mlp'))
        for i, (dataset, group) in enumerate(panels):
            block = points[(points.scope == 'fixed_step')&(points.step == 1200)&(points.kind == kind)&(points.dataset == dataset)&(points.group == group)&(points.role != 'train')]
            for j, metric in enumerate(('balanced_log_loss', 'balanced_accuracy')):
                ax = axes[i, j]
                for role in roles[1:]:
                    for pretrained in ((True, False) if kind == 'head' else (None,)):
                        part = block[(block.role == role)&(block.metric == metric)]
                        if kind == 'head':
                            part = part[part.pretrained == pretrained]
                            values = [float(part[(part.changed == a)&(part.reference == b)].delta.iloc[0]) for a, b in comparisons]
                            color = '#2463aa' if pretrained else '#bd4c25'
                        else:
                            values = [float(part[part['head'] == head].delta.iloc[0]) for head in names]
                            color = '#2463aa'
                        values = np.array(values)*(100 if j else 1)
                        offset = (.12 if pretrained else -.12) if kind == 'head' else 0
                        offset += .04 if role == roles[1] else -.04
                        line, = ax.plot(values, np.arange(3)+offset, linestyle='none', marker='o', markersize=5,
                            color=color, markerfacecolor='auto' if role == roles[1] else 'none',
                            label=(('Pretrained' if pretrained else 'Random42')+' ' if kind == 'head' else '')+('Unseen' if role == roles[1] else 'Familiar'))
                        np.testing.assert_array_equal(line.get_xdata(), values); effects += len(values)
                ax.axvline(0, color='#777777', linestyle='--'); ticks(ax.xaxis); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
                ax.set_yticks(range(3), descriptions if j == 0 else ['']*3)
                ax.set_xlabel('Loss difference (lower better)' if j == 0 else 'Balanced accuracy difference (pp; higher better)')
                ax.set_title(f'{"DEAP" if dataset == "DEAP" else "SEED-IV"}, group {group}: fixed update 1,200')
        handles, legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend, loc='outside lower center', ncol=4, fontsize=9)
        fig.suptitle(('Matched readout effects' if kind == 'head' else 'Pretrained minus random, within each head')+'; descriptive development points', fontsize=13)
        save(fig, kind+'_final_step_effects')
    if curves != 288 or effects != 144:
        raise ValueError('Missing readout plot coordinates')
    write(output/'verification.json', {'complete': True, 'declaration_sha256': sha(output/'declaration.json'),
        'curve_points_checked': curves, 'effect_points_checked': effects,
        'artifact_sha256': {p.name: sha(p) for p in output.iterdir() if p.suffix in ('.png', '.svg')}})
    destination.mkdir()
    for path in output.iterdir():
        shutil.copyfile(path, destination/path.name)
    manifest = read(PUBLIC/'export_manifest.json')
    for item in manifest['files']:
        if sha(PUBLIC/item['file']) != item['sha256']:
            raise ValueError('Changed numerical readout export')
    manifest['files'] += [{'file': p.relative_to(PUBLIC).as_posix(), 'sha256': sha(p)} for p in sorted(destination.iterdir())]
    write(PUBLIC/'export_manifest.json', manifest)
    verify()


def verify():
    folder = PUBLIC/'figures'; proof = read(folder/'verification.json'); declaration = read(folder/'declaration.json')
    if not proof['complete'] or sha(folder/'declaration.json') != proof['declaration_sha256'] or sha(Path(__file__)) != declaration['source_sha256']:
        raise ValueError('Changed readout figure source/declaration')
    for name, checksum in declaration['input_sha256'].items():
        if sha(PUBLIC/name) != checksum:
            raise ValueError('Changed figure input')
    for name, checksum in proof['artifact_sha256'].items():
        if sha(folder/name) != checksum:
            raise ValueError('Changed figure output')
    print(json.dumps({'passed': True, 'figure_pairs': 4, 'loss_coordinates': 288, 'effect_coordinates': 144}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('generate', 'verify'))
    args = parser.parse_args(); (generate if args.action == 'generate' else verify)()
