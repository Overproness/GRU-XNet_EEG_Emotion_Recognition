"""Compare the port with executed pinned author outputs; benchmark synthetic inputs."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.eegnet_control import EEGNetControl
from gruxnet.full_context_models_v2 import FullControl
from gruxnet.data import sha256, write_json
from gruxnet.train import seed_everything

ROOT = REPO.parent/'publication_runs'
AUTHOR = ROOT/'eegnet_author_audit_2026-10-06'


def read_weights(data, stage):
    weights = {}
    for name in ('conv2d', 'depthwise_conv2d', 'separable_conv2d',
                 'batch_normalization', 'batch_normalization_1', 'batch_normalization_2', 'dense'):
        weights[name] = [data[f'{stage}:{name}:{j}'] for j in range(4)
                         if f'{stage}:{name}:{j}' in data]
    return weights


def run():
    torch.set_num_threads(4)
    manifest = json.loads((AUTHOR/'download_manifest.json').read_text())
    for name, item in manifest['files'].items():
        assert sha256(AUTHOR/name) == item['sha256']
    details = []
    for classes in (2, 3):
        data = np.load(AUTHOR/f'author_arrays_classes{classes}.npz', allow_pickle=False)
        model = EEGNetControl(classes, dropout=0.)
        model.copy_author_weights(read_weights(data, 'before'))
        inputs = torch.from_numpy(data['input'][..., 0])
        errors = {}
        for training, name in ((False, 'evaluation'), (True, 'training')):
            model.train(training)
            actual = torch.softmax(model(inputs), 1).detach().numpy()
            errors[name] = float(np.max(np.abs(actual-data[name])))
            np.testing.assert_allclose(actual, data[name], atol=1e-5, rtol=0)
        author_after = read_weights(data, 'after')
        for layer, name in ((model.bn1, 'batch_normalization'),
                            (model.bn2, 'batch_normalization_1'),
                            (model.bn3, 'batch_normalization_2')):
            for j, field in ((2, 'running_mean'), (3, 'running_var')):
                actual = getattr(layer, field).detach().numpy()
                expected = author_after[name][j]
                errors[name+':'+field] = float(np.max(np.abs(actual-expected)))
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        model.constrain()
        spatial = model.spatial.weight.detach().numpy().reshape(8, 2, 14, 1).transpose(2, 3, 0, 1)
        head = model.head.weight.detach().numpy().T
        for name, actual in (('depthwise_conv2d', spatial), ('dense', head)):
            expected = data[f'constrained:{name}']
            errors[name+':constraint'] = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        count = sum(p.numel() for p in model.parameters())
        author_count = next(r['trainable_parameters'] for r in
                            json.loads((AUTHOR/'tensorflow_execution.json').read_text())['models']
                            if r['classes'] == classes)
        assert count == author_count
        details.append({'classes': classes, 'parameters': count, 'maximum_errors': errors,
                        'array_sha256': sha256(AUTHOR/f'author_arrays_classes{classes}.npz')})
    resources = []
    seed_everything(19)
    for name in ('gru', 'lstm', 'cbsatt_local', 'eegnet'):
        model = (EEGNetControl(3) if name == 'eegnet' else FullControl(name, 3)).cuda()
        x = torch.randn((12, 14, 5120) if name == 'eegnet' else (12, 14, 37, 79), device='cuda')
        y = torch.arange(12, device='cuda') % 3
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        for step in range(8):
            if step == 3:
                torch.cuda.synchronize(); started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if name == 'eegnet': model.constrain()
        torch.cuda.synchronize()
        resources.append({'model': name, 'seconds_per_synthetic_step': (time.perf_counter()-started)/5,
                          'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated()})
        del x, y, model, optimizer
    result = {'passed': True, 'author_commit': manifest['commit'],
              'author_source_sha256': manifest['files']['EEGModels.py']['sha256'],
              'source_sha256': {f: sha256(REPO/f) for f in ('gruxnet/eegnet_control.py',
                               'scripts/audit_eegnet_author_tf.py', 'scripts/audit_eegnet_control.py')},
              'forward_absolute_tolerance': 1e-5, 'state_constraint_absolute_tolerance': 1e-6,
              'models': details, 'synthetic_resource_pilot': resources,
              'scope': 'Executed pinned author function versus weight-mapped port: eval and dropout-zero training outputs, BatchNorm moving-state updates, max-norm constraints and parameter counts. Synthetic-step resource pilot only. Dropout RNG, initial random draws, optimizer paths and published scores are not reproduced.'}
    write_json(AUTHOR/'port_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
