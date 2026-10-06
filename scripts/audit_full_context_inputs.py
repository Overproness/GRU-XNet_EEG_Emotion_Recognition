"""Independent SciPy STFT and exact historical-local forward mapping audit."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.signal import stft
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256, write_json
from gruxnet.full_context_controls import load, GROUP, contexts, batches
from gruxnet.full_context_models import FullControl, copy_legacy
from gruxnet.grouped_material_controls import annotate, cells, indices, ARMS
from gruxnet.reve_probe import load_inputs
from gruxnet.train import seed_everything

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent/'publication_runs'

def audit():
    seed_everything(42); mapping = []
    for name, relative, kind in (('gru', 'OurApproach/CBSAtt/model.py', 'gru_xnetDynamic'),
                                ('cbsatt_local', 'CBSAtt_og/model.py', 'CBSAtt')):
        path = REPO.parent/relative; spec = importlib.util.spec_from_file_location('audited_local', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for classes in (2,3):
            legacy = getattr(module, kind)(n_channels=14, n_freq_bins=37, n_time_bins=79, n_classes=classes).eval()
            model = FullControl(name, classes).eval(); copy_legacy(model, legacy)
            x = torch.randn(2,14,37,79)
            with torch.no_grad():
                a = legacy(x); a = a[0] if isinstance(a,tuple) else a; b = model(x)
            error = float((a-b).abs().max())
            if error > 1e-5: raise ValueError('Historical weight-mapped forward mismatch')
            mapping.append({'model': name, 'classes': classes, 'local_source': relative,
                            'local_source_sha256': sha256(path), 'maximum_logit_error': error})
            del legacy, model, x
    seed_raw, _, _ = load_inputs(ROOT/'cache_reve_input_seediv')
    lineage = {r['trial_id']: r for r in json.loads((ROOT/'cache_common14/lineage.json').read_text()) if r['dataset']=='DEAP'}
    for dataset in ('SEEDIV','DEAP'):
        cache = ROOT/f'cache_full_context_{dataset.lower()}'; x, original, info = load(cache); maximum = 0.
        for i, row in original.iterrows():
            if dataset == 'SEEDIV': raw = seed_raw[i]
            else:
                record = lineage[row.trial_id]; path = ROOT/'cache_common14'/record['cache_file']
                if sha256(path) != record['cache_sha256']: raise ValueError('Changed DEAP waveform')
                raw = np.load(path, allow_pickle=False)[:10].transpose(1,0,2).reshape(14,5120)
            f, _, z = stft(raw.astype(np.float64), fs=128, window='hann', nperseg=128, noverlap=64, nfft=128,
                           detrend=False, boundary=None, padded=False, scaling='spectrum', axis=-1)
            expected = np.log1p(np.abs(z[:,(f>=4)&(f<=40)])).astype(np.float32)
            error = float(np.max(np.abs(expected-x[i]))); maximum = max(maximum,error)
            if error > 3e-6: raise ValueError('Independent STFT input replay failed')
        table = annotate(original,dataset,GROUP); records=[]
        for session,rotation,fold in cells(table,dataset,GROUP)[1]:
            paired=[]
            for arm in ARMS:
                idx=indices(table,fold,session,rotation,arm,dataset,GROUP)
                _,signature=batches(table,idx['train'],42+1000*fold['fold']+10000*rotation,3 if dataset=='SEEDIV' else 2)
                q=contexts(table,idx,3 if dataset=='SEEDIV' else 2)
                if not np.isfinite(q).all() or not np.allclose(q[np.concatenate(list(idx.values()))].sum(1),1): raise ValueError('Invalid source-only priors')
                paired.append((idx,signature))
                records.append({'session':int(session),'rotation':rotation,'fold':fold['fold'],'arm':arm,
                                'trials':{part:table.iloc[v].trial_id.tolist() for part,v in idx.items()},'canonical_draw_signature':signature})
            for part in ('validation','test'): np.testing.assert_array_equal(paired[0][0][part],paired[1][0][part])
            if paired[0][1]!=paired[1][1]: raise ValueError('Unpaired original trial sampling')
        write_json(cache/'input_audit.json',{'passed':True,'dataset':dataset,'cache_fingerprint':info['fingerprint'],
                   'source_sha256':sha256(Path(__file__)),'trials_independently_replayed':len(table),
                   'maximum_scipy_stft_error':maximum,'stft_tolerance':3e-6,'legacy_weight_mapping':mapping,
                   'all_predeclared_splits':records,'scope':'All fixed input STFTs independently replayed via SciPy, historical-local weights mapped, every original trial partition and context prior feasibility before fitting; no outcome-based grouping selection.'})
        print(dataset,len(table),'independent STFT error',maximum,flush=True)

if __name__=='__main__': audit()
