"""Run untouched pinned author code under TensorFlow, for cross-framework QA."""
from pathlib import Path
import sys
import os
import json
import hashlib

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent/'publication_runs'
AUTHOR = ROOT/'eegnet_author_audit_2026-10-06'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.insert(0, str(ROOT/'tf_numpy_compat'))
sys.path.insert(0, str(AUTHOR))
import numpy as np
import tensorflow as tf
from EEGModels import EEGNet


def run():
    manifest = json.loads((AUTHOR/'download_manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((AUTHOR/name).read_bytes()).hexdigest() == record['sha256']
    details = []
    for classes in (2, 3):
        tf.keras.backend.clear_session()
        model = EEGNet(classes, Chans=14, Samples=5120, dropoutRate=0.)
        rng = np.random.default_rng(600+classes)
        x = rng.normal(size=(3, 14, 5120, 1)).astype('float32')
        arrays = {'input': x}
        for layer in model.layers:
            if not layer.get_weights():
                continue
            values = []
            for j, weight in enumerate(layer.get_weights()):
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    value = (rng.uniform(.8, 1.2, weight.shape) if j == 0 else
                             rng.uniform(.3, 1.3, weight.shape) if j == 3 else
                             rng.normal(0, .03, weight.shape))
                else:
                    value = rng.normal(0, .08, weight.shape)
                values.append(value.astype('float32'))
                arrays[f'before:{layer.name}:{j}'] = values[-1]
            layer.set_weights(values)
        arrays['evaluation'] = model(x, training=False).numpy()
        # Disable stochastic dropout only for deterministic operator/BN QA.
        arrays['training'] = model(x, training=True).numpy()
        for layer in model.layers:
            for j, value in enumerate(layer.get_weights()):
                arrays[f'after:{layer.name}:{j}'] = value
            for variable in layer.trainable_weights:
                if variable.constraint is not None:
                    constrained = variable.constraint(variable).numpy()
                    arrays[f'constrained:{layer.name}'] = constrained
        np.savez(AUTHOR/f'author_arrays_classes{classes}.npz', **arrays)
        details.append({'classes': classes, 'parameters_including_BN_buffers': model.count_params(),
                        'trainable_parameters': int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                        'layers': [layer.name for layer in model.layers]})
    result = {'completed': True, 'tensorflow': tf.__version__, 'numpy_overlay': np.__version__,
              'cpu_only': True, 'author_commit': manifest['commit'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'models': details,
              'scope': 'Untouched authors EEGNet function; declared common14/5120 input and deterministic dropout-zero variant solely for numerical operator, BatchNorm update and constraint QA. No accuracy experiment or optimizer reproduction.'}
    (AUTHOR/'tensorflow_execution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
