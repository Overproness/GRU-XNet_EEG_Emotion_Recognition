"""Outcome-free CPU shape audit of pinned author classifier wrappers."""
from pathlib import Path
import importlib.util
import json
import sys
import types
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_cbramod_learning import sha


class SyntheticBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj_out = nn.Identity()

    def forward(self, x):
        # Both patch and token dimensions are200; no real encoder is executed.
        return x


def main():
    torch.set_num_threads(2)
    source = REPO.parent/'publication_runs/cbramod_audit_2026-10-09/author'
    output = REPO.parent/'publication_runs/cbramod_readout_audit_2026-10-09'
    if output.exists():
        raise FileExistsError('Preserve shape audit')
    names = ('models', 'models.cbramod', 'models.model_for_faced', 'models.model_for_seedv')
    saved = {name: sys.modules.get(name) for name in names}
    package = types.ModuleType('models'); package.__path__ = [str(source/'models')]
    stub = types.ModuleType('models.cbramod'); stub.CBraMod = SyntheticBackbone
    sys.modules['models'] = package; sys.modules['models.cbramod'] = stub
    records = []
    try:
        for dataset, channels, patches, classes in (('faced', 32, 10, 9), ('seedv', 62, 1, 5)):
            name = 'models.model_for_'+dataset
            spec = importlib.util.spec_from_file_location(name, source/'models'/('model_for_'+dataset+'.py'))
            module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
            tokens = torch.randn(2, channels, patches, 200)
            for readout in ('avgpooling_patch_reps', 'all_patch_reps_onelayer', 'all_patch_reps_twolayer'):
                param = types.SimpleNamespace(use_pretrained_weights=False, classifier=readout, num_of_classes=classes, dropout=.1)
                torch.manual_seed(42)
                model = module.Model(param).eval()
                with torch.inference_mode():
                    direct = model.classifier(tokens)
                    if direct.shape != (2, classes) or not torch.isfinite(direct).all():
                        raise ValueError('Unexpected standalone author head shape')
                    try:
                        wrapped = model(tokens)
                        error = None
                        torch.testing.assert_close(wrapped, direct, rtol=0, atol=0)
                    except Exception as exception:
                        error = repr(exception)
                records.append({'dataset_adapter': dataset, 'token_shape': list(tokens.shape), 'readout': readout,
                    'standalone_head_ok': True, 'wrapper_matches_standalone_head': error is None,
                    'wrapper_error': error, 'classifier_parameters': sum(p.numel() for p in model.classifier.parameters())})
                del model
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    if len(records) != 6 or any(not r['wrapper_matches_standalone_head'] for r in records[:3]) or any(r['wrapper_matches_standalone_head'] for r in records[3:]):
        raise ValueError('Shape audit differs from inspected source paths')
    result = {'complete': True, 'synthetic_only': True, 'task_fits': 0, 'device': 'CPU',
        'author_commit': 'b9e961003214326972c567eff390e75b0287e32a', 'source_sha256': sha(Path(__file__)),
        'author_sha256': {name: sha(source/name) for name in ('models/model_for_faced.py', 'models/model_for_seedv.py', 'finetune_main.py')},
        'scope': 'Exact author head/wrapper source, synthetic identity backbone and unmodified documented token shapes. No pretrained weights, real encoder forward, EEG, probabilities or score reproduction. SEED-V wrapper flattens tokens before heads expecting4D; standalone heads are well-defined. Does not assert the paper used this faulty wrapper/version.',
        'records': records}
    output.mkdir()
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
