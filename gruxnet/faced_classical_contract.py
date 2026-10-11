"""Synthetic-testable bounded FACED classical controls; no original-data I/O."""
from __future__ import annotations
import hashlib
import json
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CATEGORIES=['Amusement','Anger','Disgust','Fear','Inspiration','Joy','Sadness','Tenderness']
ACQUISITIONS=['legacy_1000','legacy_250','modern_1000']
ALPHAS=[.1,1.,10.,100.,1000.]
BANDS=[('delta',1,4),('theta',4,8),('alpha',8,13),('beta',13,30),('gamma',30,45)]


def role_pairs(reservation,rotation):
    source=reservation['development_source'];validation=reservation['development_validation']
    protected=set(reservation['confirmation']);c=rotation['clips']
    train=set(c['development_source']);val=set(c['development_validation']);confirm=set(c['confirmation'])
    if set(source)&set(validation) or (set(source)|set(validation))&protected:
        raise ValueError('Participant roles overlap')
    if train&val or (train|val)&confirm:
        raise ValueError('Film roles overlap')
    product=lambda people,clips:sorted((p,v) for p in people for v in clips)
    return {'source_fit':product(source,train),'validation_unseen':product(validation,val),
            'validation_familiar':product(validation,train)}


def spectral_features(windows):
    values=np.asarray(windows,dtype=np.float64)
    if values.shape!=(7,30,512) or not np.isfinite(values).all():
        raise ValueError('Require seven finite common-scalp windows')
    f,p=welch(values,fs=128,window='hann',nperseg=512,noverlap=0,nfft=512,
              detrend='constant',scaling='density',axis=-1)
    powers=np.stack([p[...,(f>=lo)&(f<hi)].sum(axis=-1)*.25 for _,lo,hi in BANDS],axis=-1)
    # Equal contribution from all seven windows; one vector and target per trial.
    result=np.log10(np.maximum(powers,1e-8)).mean(axis=0).reshape(-1)
    if result.shape!=(150,) or not np.isfinite(result).all():raise ValueError('Spectral geometry changed')
    return result


def context_features(metadata):
    category=metadata['assigned_category'];acquisition=metadata['acquisition_group']
    duration=float(metadata['catalogue_duration_seconds']);trial=metadata['trial']
    if category not in CATEGORIES or acquisition not in ACQUISITIONS or not np.isfinite(duration) or duration<=0 or not 1<=trial<=28:
        raise ValueError('Unknown or invalid declared metadata')
    return np.asarray([float(category==c) for c in CATEGORIES]+
                      [float(acquisition==a) for a in ACQUISITIONS]+[duration,float(trial)])


def fit_source_ridge(source_pairs,features,targets,alpha):
    """Only fetch explicitly declared source identities from mappings."""
    pairs=list(source_pairs)
    if not pairs or len(pairs)!=len(set(pairs)) or alpha not in ALPHAS:
        raise ValueError('Invalid declared source pairs or alpha')
    x=np.stack([features[p] for p in pairs]).astype(np.float64)
    y=np.asarray([targets[p] for p in pairs],dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y<0)|(y>7)):
        raise ValueError('Invalid source feature/target; stop rather than discard rows')
    scaler=StandardScaler().fit(x)
    estimator=Ridge(alpha=alpha,solver='svd',fit_intercept=True).fit(scaler.transform(x),y)
    source_digest=hashlib.sha256(json.dumps(pairs,separators=(',',':')).encode()).hexdigest()
    return {'scaler':scaler,'estimator':estimator,'source_pairs':pairs,'source_identity_sha256':source_digest}


def predict_ridge(fitted,pairs,features):
    x=np.stack([features[p] for p in pairs]).astype(np.float64)
    if not np.isfinite(x).all():raise ValueError('Invalid evaluated feature needs declared fallback')
    return np.clip(fitted['estimator'].predict(fitted['scaler'].transform(x)),0,7)


def participant_film_mae(rows):
    """Average clips within participant/film, films within person, then people."""
    grouped={}
    identities=set()
    for row in rows:
        identity=(row['subject'],row['vid'])
        if identity in identities:raise ValueError('Duplicate trial outcome')
        identities.add(identity)
        target,prediction=float(row['target']),float(row['prediction'])
        if not np.isfinite([target,prediction]).all():raise ValueError('Nonfinite evaluation')
        key=(row['subject'],row['family'])
        grouped.setdefault(key,[]).append(abs(target-prediction))
    if not grouped:raise ValueError('Empty evaluation')
    people={}
    for (person,family),errors in grouped.items():people.setdefault(person,[]).append(float(np.mean(errors)))
    return float(np.mean([np.mean(errors) for errors in people.values()]))


def select_alpha(candidates):
    if len(candidates)!=len(ALPHAS) or [r['alpha'] for r in candidates]!=ALPHAS:
        raise ValueError('Candidate grid differs from declaration')
    if any(not np.isfinite([r['unseen_mae'],r['unseen_mse']]).all() for r in candidates):
        raise ValueError('Invalid selection outcome')
    return min(range(len(candidates)),key=lambda i:(candidates[i]['unseen_mae'],candidates[i]['unseen_mse'],i))
