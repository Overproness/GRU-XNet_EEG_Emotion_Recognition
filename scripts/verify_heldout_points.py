"""Independently recompute all aggregate points/contrasts from public test CSVs."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]


def checksum(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(65536),b''): value.update(chunk)
    return value.hexdigest()


def verify(output):
    checked=0; maximum=0.; prediction_rows=0; contrasts=0; bindings={}
    for dataset,original_count in (('seediv',1080),('deap',1264)):
        report_path=output/f'comparison_{dataset}.json'
        report=json.loads(report_path.read_text()); bindings[report_path.name]=checksum(report_path)
        reference=None
        for model in ('gru','eegnet','eegnet_context','prior','context_logistic'):
            for arm in ('exposed','unexposed'):
                path=output/f'predictions_{dataset}_{model}_{arm}.csv'
                rows=pd.read_csv(path); bindings[path.name]=checksum(path); prediction_rows+=len(rows)
                if len(rows)!=4*original_count: raise ValueError('Incomplete repeated OOF cohort')
                for _,part in rows.groupby(['group','seed']):
                    if len(part)!=original_count or part.trial_id.nunique()!=original_count: raise ValueError('OOF repetition duplicates/missing trials')
                metadata=rows[['group','seed','trial_id','subject_id','material_key','label','original_label']].sort_values(['group','seed','trial_id']).reset_index(drop=True)
                if reference is None: reference=metadata
                else: pd.testing.assert_frame_equal(reference,metadata,check_exact=True)
                scopes=[('combined',None,None)]+[(f'group{g}_init{i}',g,i) for g in (1,2) for i in (42,91)]
                for scope,g,i in scopes:
                    part=rows if g is None else rows[rows.group.eq(g)&rows.seed.eq(i)]
                    p=part.filter(regex='^p_').to_numpy(dtype=float)
                    if not np.isfinite(p).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid probability')
                    np.testing.assert_allclose(p.sum(1),1.,atol=1e-6,rtol=0)
                    tasks={'coarse3' if dataset=='seediv' else 'binary':(part.label.to_numpy(dtype=int),p)}
                    if dataset=='seediv':
                        keep=part.original_label.ne(0).to_numpy(); positive=p[keep,2]/np.maximum(p[keep,1:].sum(1),1e-12)
                        tasks['binary']=((part.original_label.to_numpy()[keep]==3).astype(int),np.stack([1-positive,positive],axis=1))
                    for task,(y,z) in tasks.items():
                        correct=(z.argmax(1)==y).astype(float)
                        losses=-np.log(np.clip(z[np.arange(len(y)),y],1e-12,1))
                        expected=report['models'][scope][model][arm][task]
                        for metric,values in (('BA',correct),('logloss',losses)):
                            actual=float(np.mean([values[y==c].mean() for c in range(z.shape[1])]))
                            error=abs(actual-expected[metric]['point']); maximum=max(maximum,error)
                            if error>=2e-12: raise ValueError('Aggregate point mismatch')
                            checked+=1
        for contrast in report['contrasts']:
            a=report['models'][contrast['scope']][contrast['model_a']][contrast['arm_a']][contrast['task']][contrast['metric']]['point']
            b=report['models'][contrast['scope']][contrast['model_b']][contrast['arm_b']][contrast['task']][contrast['metric']]['point']
            if abs(a-b-contrast['difference'])>=2e-12: raise ValueError('Paired contrast point mismatch')
            contrasts+=1
    if checked!=300 or prediction_rows!=93760 or contrasts!=390: raise ValueError('Unexpected analysis size')
    result={'passed':True,'aggregate_metric_points':checked,'paired_contrast_points':contrasts,
            'outer_prediction_rows':prediction_rows,'maximum_absolute_error':maximum,
            'source_sha256':checksum(Path(__file__)),'input_sha256':bindings,
            'scope':'Independent direct class-mean correctness/logloss for every model/arm/task and combined/all four repetitions, complete matched OOF coverage and all contrast point subtraction. Existing bootstrap percentile endpoints and within-video dyadic intervals are not re-executed by this check.'}
    (output/'aggregate_point_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'results/development/heldout_tuning_2026-10-06')
    args=parser.parse_args(); print(json.dumps(verify(args.output.resolve())))
