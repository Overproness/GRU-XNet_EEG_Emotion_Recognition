"""Exploratory fixed-duration temporal and label-granularity controls.

Independent module: preceding experiment source hashes are intentionally preserved.
This is an authored bandpower transformer control, not an EEG-Conformer reproduction.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .data import COMMON_CHANNELS, SEED_LABELS, digest, roots, seed_channels, sha256, write_json
from .prepare import load_prepared, preprocess_signal
from .train import seed_everything
from scripts.native_seediv_diagnostic import BANDS, C_VALUES, load as load_native, make_folds, view

FRAMES = 10
COARSE = np.array([0, 1, 1, 2], dtype=np.int64)
CONDITIONS = [(a, r, o) for a in ("mean_mlp", "transformer")
              for r in ("absolute", "relative") for o in ("coarse3", "native4")]


def sequence_features(windows):
    if windows.ndim != 3 or windows.shape[1:] != (14, 512) or not np.isfinite(windows).all():
        raise ValueError("Expected finite [windows, 14 electrodes, 512 samples]")
    frequency, density = welch(windows, fs=128, window="hann", nperseg=256,
                               noverlap=128, nfft=256, detrend="constant", scaling="density", axis=-1)
    power = np.stack([density[..., (frequency >= low) & (frequency < high)].sum(-1)*.5
                      for low, high in BANDS], axis=-1)
    return np.log(np.maximum(power, 1e-12)).reshape(len(windows), 56).astype(np.float32)


def represent(x, name):
    if name == "absolute":
        return x.copy()
    if name != "relative":
        raise ValueError(name)
    shaped = x.reshape(*x.shape[:-1], 14, 4)
    return (shaped-logsumexp(shaped, axis=-1, keepdims=True)).reshape(x.shape).astype(np.float32)


def prepare(data_root, native_cache, common_cache, cache):
    if cache.exists():
        raise FileExistsError("Use a fresh temporal feature cache")
    old, table, native = load_native(native_cache)
    expected, _, _ = view(old, table, native["channels"], "common14", "native4")
    manifest, common = load_prepared(common_cache, verify_files=True)
    old_windows = manifest[manifest.dataset == "SEEDIV"].groupby("trial_id").first()
    root = roots(data_root)["SEEDIV"]
    channels = seed_channels(root)
    if channels != native["channels"]:
        raise ValueError("Changed SEED channel ordering")
    for name, recorded in native["metadata_sources"].items():
        if sha256(root/name) != recorded:
            raise ValueError("Changed original metadata")
    sequences = np.empty((len(table), FRAMES, 56), dtype=np.float32)
    paired = 0
    start = time.perf_counter()
    for source in native["source_files"]:
        path = root/source["file"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Changed raw SEED source")
        values = loadmat(path)
        for i, row in table[table.source_file == source["file"]].iterrows():
            trial_number = int(row.trial_id.rsplit("T", 1)[1])
            if row.original_label != SEED_LABELS[int(row.session)][trial_number-1]:
                raise ValueError("Changed native label")
            filtered, mask = preprocess_signal(values[row.source_key], 200, channels, COMMON_CHANNELS)
            n = filtered.shape[1]//512
            if n != row.windows or n < FRAMES or not mask.all():
                raise ValueError("Incorrect duration or electrodes")
            windows = filtered[:, :n*512].reshape(14, n, 512).transpose(1, 0, 2)
            features = sequence_features(windows)
            np.testing.assert_array_equal(features.mean(0), expected[i])
            if row.trial_id in old_windows.index:
                retained = np.load(common_cache/old_windows.loc[row.trial_id, "cache_file"], allow_pickle=False)
                np.testing.assert_array_equal(windows, retained)
                paired += 1
            sequences[i] = features[:FRAMES]
        print(f"Sequence features: {source['file']}", flush=True)
    if paired != 810 or sequences.shape != (1080, 10, 56):
        raise ValueError("Incomplete native/common overlap")
    cache.mkdir(parents=True)
    np.save(cache/"sequences.npy", sequences, allow_pickle=False)
    table.to_csv(cache/"trials.csv", index=False)
    info = {"development_only": True, "research_question_change_approved": False,
            "native_cache_fingerprint": native["fingerprint"], "common_cache_fingerprint": common["cache_fingerprint"],
            "shape": list(sequences.shape), "channels": COMMON_CHANNELS, "window_seconds": 4,
            "frames": FRAMES, "duration_seconds": 40, "observation": "First 10 non-overlapping windows of each trial",
            "filter_context": "Zero-phase filtering and resampling use the full source trial before taking its prefix; offline protocol, not causal 40-second acquisition",
            "features": "Welch Hann 256, overlap 128, FFT 256 at 128 Hz, density; half-open bands 4..8/8..14/14..31/31..40; natural log of bin sum * .5 Hz, floor 1e-12",
            "full_trial_mean_exact_matches": 1080, "full_trial_waveforms_exact_matches": paired,
            "source_files": native["source_files"], "metadata_sources": native["metadata_sources"],
            "sequences_sha256": sha256(cache/"sequences.npy"), "trials_sha256": sha256(cache/"trials.csv"),
            "elapsed_seconds": time.perf_counter()-start}
    info["fingerprint"] = digest(info)
    write_json(cache/"prepared.json", info)
    return info


def load(cache):
    info = json.loads((cache/"prepared.json").read_text())
    check = dict(info)
    fingerprint = check.pop("fingerprint")
    if digest(check) != fingerprint:
        raise ValueError("Changed temporal cache metadata")
    for filename, key in [("sequences.npy", "sequences_sha256"), ("trials.csv", "trials_sha256")]:
        if sha256(cache/filename) != info[key]:
            raise ValueError("Changed temporal features or labels")
    x = np.load(cache/"sequences.npy", allow_pickle=False)
    table = pd.read_csv(cache/"trials.csv")
    if x.shape != (1080, 10, 56) or table.trial_id.duplicated().any():
        raise ValueError("Changed cohort")
    return x, table, info


class TemporalControl(nn.Module):
    def __init__(self, architecture, objective):
        super().__init__()
        self.architecture, self.objective = architecture, objective
        if architecture == "mean_mlp":
            self.encoder = nn.Sequential(nn.Linear(56,128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.2),
                                         nn.Linear(128,86), nn.LayerNorm(86), nn.GELU(), nn.Dropout(.2))
            width = 86
        elif architecture == "transformer":
            self.projection = nn.Linear(56,32)
            # Separate constructors give independent layer initialization (no cloned initial weights).
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(32,4,64,.2,"gelu",
                                        batch_first=True,norm_first=True) for _ in range(2)])
            self.encoder = nn.LayerNorm(32)
            position = torch.arange(FRAMES, dtype=torch.float32)[:,None]*4
            scale = torch.exp(torch.arange(0,32,2, dtype=torch.float32)*(-math.log(10000)/32))
            encoding = torch.empty(FRAMES,32)
            encoding[:,0::2], encoding[:,1::2] = torch.sin(position*scale), torch.cos(position*scale)
            self.register_buffer("positions", encoding)
            width = 32
        else:
            raise ValueError(architecture)
        base = nn.Linear(width,3)
        if objective == "coarse3":
            self.head = base
        elif objective == "native4":
            self.head = nn.Linear(width,4)
            with torch.no_grad():
                self.head.weight.copy_(base.weight[[0,1,1,2]])
                self.head.bias.copy_(base.bias[[0,1,1,2]])
                self.head.bias[1:3] -= math.log(2)
        else:
            raise ValueError(objective)

    def forward(self, x):
        if self.architecture == "mean_mlp":
            latent = self.encoder(x.mean(1))
        else:
            latent = self.projection(x)+self.positions
            for layer in self.layers:
                latent = layer(latent)
            latent = self.encoder(latent.mean(1))
        return self.head(latent)


def coarse_probabilities(probabilities, objective):
    if objective == "coarse3":
        return probabilities
    if objective != "native4":
        raise ValueError(objective)
    return np.stack([probabilities[:,0], probabilities[:,1]+probabilities[:,2], probabilities[:,3]], axis=1)


def metric(y, p):
    prediction = p.argmax(1)
    return {"n": len(y), "accuracy": float(accuracy_score(y,prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y,prediction)),
            "macro_f1": float(f1_score(y,prediction,labels=list(range(p.shape[1])),average="macro",zero_division=0)),
            "balanced_log_loss": float(np.mean([-np.log(np.clip(p[y==c,c],1e-12,1)).mean()
                                                 for c in range(p.shape[1])])),
            "confusion_matrix": confusion_matrix(y,prediction,labels=list(range(p.shape[1]))).tolist()}


def common_metrics(native_labels, p, objective):
    coarse = coarse_probabilities(p,objective)
    selected = native_labels != 0
    positive = coarse[selected,2]/np.maximum(coarse[selected,1:].sum(1),1e-12)
    result = {"coarse3": metric(COARSE[native_labels],coarse),
              "binary": metric((native_labels[selected]==3).astype(int),np.stack([1-positive,positive],axis=1))}
    if objective == "native4":
        result["native4"] = metric(native_labels,p)
    return result


def draw_batches(labels, train_indices, seed, updates=600):
    rng = np.random.default_rng(seed)
    pools = [train_indices[COARSE[labels[train_indices]]==c] for c in range(3)]
    if any(len(pool)==0 for pool in pools):
        raise ValueError("Missing training class")
    return np.stack([np.concatenate([rng.choice(pool,20,replace=True) for pool in pools]) for _ in range(updates)])


def plan():
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True,
            "research_question_change_approved": False, "target": "SEED-IV, all 1080 original trials including neutral",
            "conditions": [list(c) for c in CONDITIONS], "initialization_seeds": [42,43,44], "fold_seed": 42,
            "folds": "Same five 9 train / 3 validation / 3 test participant rotations as earlier native controls",
            "observation": "Fixed first 40 seconds (10 four-second bandpower tokens) per trial; full-trial offline filter context retained",
            "features": "14 common physical electrodes, four half-open bands per channel; absolute ln power or per-window ln fraction of power in the four bands",
            "normalization": "Per-feature StandardScaler fitted on training trial/windows only, shared across label/architecture arms",
            "models": "Mean MLP 56->128->86->3/4 vs two-layer 32-wide transformer, four heads, feedforward 64; fixed sinusoidal positions in seconds; mean token pooling; LayerNorm/GELU/dropout .2. Coarse heads: 19079 vs 19075 parameters",
            "labels": "Coarse3: neutral / negative (sad+fear) / positive (happy); native4: neutral / sad / fear / happy. Identical training trials, 20 draws of each coarse class per batch; native classes intentionally not equally sampled",
            "initialization": "Same backbone and identical initial grouped probabilities within architecture; native negative rows duplicate coarse row with -ln2 bias. Reset stochastic training RNG after model construction",
            "optimization": "600 AdamW updates, batch 60, lr .001, weight decay .01, gradient clip 1; deterministic CUDA math SDPA; no AMP/augmentation/scheduler/early stopping",
            "selection": "Every 25 updates: maximum validation common THREE-class trial BA, then minimum common three-class balanced log loss; first checkpoint for exact ties. Test only after selection",
            "evaluation": "Primary common THREE-class valence all 1080; secondary binary valence 810 nonneutral via conditional positive/(negative+positive); native four-class is secondary descriptive within native objective",
            "comparisons": "Paired transformer-minus-MLP, relative-minus-absolute, native-minus-coarse for all factorial strata; mean across three initializations. Participant-block percentile bootstrap 10000 draws seeded 20261006, fixed seeds. No correction: exploratory, multiple comparisons",
            "linear_controls": "Coarse3 multinomial logistic: absolute/relative prefix mean features and scalar full-trial window-count diagnostic, identical folds; C .01/.1/1/10; class_weight balanced; training-only scaler; validation common three-class BA then balanced log loss; 60 candidate fits. Length-only model demonstrates label-duration association, not proof previous neural models used it",
            "limitations": "Previously inspected development cohort, one fold grouping, only one corpus, no pooled/unseen-dataset experiment; bandpower transformer is not a foundation model or raw-waveform Conformer reproduction; sampling/loss changed vs earlier binary experiments",
            "decision_gate": "Show findings before proposing a changed question; archive then-current paper and obtain explicit approval before adopting any change"}


@torch.no_grad()
def infer(model, x):
    model.eval()
    return torch.softmax(model(x),dim=1).cpu().numpy()


def run(cache, output, plan_path, device="cuda"):
    if (output/"comparison.json").exists():
        raise FileExistsError("Completed temporal run exists")
    x, table, info = load(cache)
    declaration = json.loads(plan_path.read_text())
    current = plan()
    for key in current:
        if key != "created_utc" and declaration[key] != current[key]:
            raise ValueError(f"Plan differs: {key}")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"plan.json",declaration)
    folds = make_folds(table.subject_id.tolist())
    write_json(output/"folds.json",folds)
    config = {"cache_fingerprint": info["fingerprint"], "plan_sha256": sha256(plan_path),
              "source_hashes": {p: sha256(Path(__file__).resolve().parents[1]/p) for p in
                 ["gruxnet/temporal_controls.py","gruxnet/data.py","gruxnet/prepare.py","gruxnet/train.py","scripts/native_seediv_diagnostic.py"]},
              "torch_version": torch.__version__, "device": device,
              "device_name": torch.cuda.get_device_name() if device=="cuda" else "CPU"}
    write_json(output/"config.json",config)
    labels = table.original_label.to_numpy(dtype=np.int64)
    summaries = []
    seed_everything(42)
    for architecture, representation, objective in CONDITIONS:
        name = f"{architecture}_{representation}_{objective}"
        rows = []
        for seed in [42,43,44]:
            for fold in folds:
                identifier = f"{name}_seed{seed}_fold{fold['fold']}"
                folder = output/"models"/identifier
                folder.mkdir(parents=True,exist_ok=True)
                indices = {part: np.flatnonzero(table.subject_id.isin(fold[part])) for part in ("train","validation","test")}
                transformed = represent(x,representation)
                scaler = StandardScaler().fit(transformed[indices["train"]].reshape(-1,56))
                standardized = scaler.transform(transformed.reshape(-1,56)).reshape(transformed.shape).astype(np.float32)
                data = torch.from_numpy(standardized).to(device)
                y = labels if objective=="native4" else COARSE[labels]
                target = torch.from_numpy(y).to(device)
                initialization = seed+fold["fold"]*1000
                batches = draw_batches(labels,indices["train"],initialization)
                if (folder/"metrics.json").exists():
                    record = json.loads((folder/"metrics.json").read_text())
                    if record["config_sha256"] != sha256(output/"config.json"):
                        raise ValueError("Cannot resume changed experiment")
                    predictions = pd.read_csv(folder/"predictions.csv")
                else:
                    seed_everything(initialization)
                    model = TemporalControl(architecture,objective).to(device)
                    seed_everything(initialization+1000000)
                    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                    best_key, best_state, history, selected = None,None,[],None
                    if device=="cuda":
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with sdpa_kernel(SDPBackend.MATH):
                        for step, drawn in enumerate(batches,1):
                            model.train()
                            optimizer.zero_grad(set_to_none=True)
                            loss = nn.functional.cross_entropy(model(data[drawn]),target[drawn])
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(),1.)
                            optimizer.step()
                            if step%25==0:
                                metrics = common_metrics(labels[indices["validation"]],infer(model,data[indices["validation"]]),objective)
                                entry = {"step": step,"training_loss":float(loss.item()),"validation":metrics}
                                history.append(entry)
                                key = (metrics["coarse3"]["balanced_accuracy"],-metrics["coarse3"]["balanced_log_loss"])
                                if best_key is None or key>best_key:
                                    best_key,best_state,selected = key,copy.deepcopy(model.state_dict()),step
                        model.load_state_dict(best_state)
                        probability = infer(model,data[indices["test"]])
                        training = common_metrics(labels[indices["train"]],infer(model,data[indices["train"]]),objective)
                    elapsed = time.perf_counter()-start
                    checkpoint = {"state": {k:v.cpu() for k,v in best_state.items()},"mean":scaler.mean_,"scale":scaler.scale_,
                                  "architecture":architecture,"objective":objective,"representation":representation,
                                  "fold":fold,"seed":seed,"selected_step":selected,"batch_digest":digest(batches.tolist())}
                    torch.save(checkpoint,folder/"selected.pt")
                    write_json(folder/"history.json",history)
                    predictions = table.iloc[indices["test"]][["trial_id","subject_id","original_label"]].copy()
                    predictions["condition"],predictions["seed"],predictions["fold"] = name,seed,fold["fold"]
                    for c in range(probability.shape[1]):
                        predictions[f"p_{c}"] = probability[:,c]
                    predictions.to_csv(folder/"predictions.csv",index=False)
                    record = {"condition":name,"seed":seed,"fold":fold["fold"],"selected_step":selected,
                              "training":training,"test":common_metrics(labels[indices["test"]],probability,objective),
                              "elapsed_seconds":elapsed,"parameters":sum(p.numel() for p in model.parameters()),
                              "peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                              "config_sha256":sha256(output/"config.json"),"checkpoint_sha256":sha256(folder/"selected.pt"),
                              "history_sha256":sha256(folder/"history.json"),"predictions_sha256":sha256(folder/"predictions.csv"),
                              "batch_digest":checkpoint["batch_digest"],"selected_batch_prefix_digest":digest(batches[:selected].tolist())}
                    write_json(folder/"metrics.json",record)
                    print(f"{identifier}: step={selected}, val={best_key[0]:.4f}, test3={record['test']['coarse3']['balanced_accuracy']:.4f}, test2={record['test']['binary']['balanced_accuracy']:.4f}, {elapsed:.1f}s",flush=True)
                    del model,optimizer,best_state
                rows.append(predictions)
                summaries.append(record)
                del data,target
        pd.concat(rows,ignore_index=True).to_csv(output/f"predictions_{name}.csv",index=False)
    write_json(output/"model_metrics.json",summaries)
    linear_controls(x,table,folds,output)
    return analyze(output)


def linear_controls(x,table,folds,output):
    labels = table.original_label.to_numpy(dtype=int)
    y = COARSE[labels]
    rows, reports = [],[]
    for representation in ("absolute","relative","duration"):
        features = table.windows.to_numpy()[:,None].astype(float) if representation=="duration" else represent(x,representation).mean(1)
        for fold in folds:
            indices = {part:np.flatnonzero(table.subject_id.isin(fold[part])) for part in ("train","validation","test")}
            scaler = StandardScaler().fit(features[indices["train"]])
            transformed = scaler.transform(features)
            candidates, best_key, best, selected = [],None,None,None
            for c in C_VALUES:
                model = LogisticRegression(C=c,class_weight="balanced",solver="lbfgs",max_iter=2000,tol=1e-6,random_state=42)
                model.fit(transformed[indices["train"]],y[indices["train"]])
                result = common_metrics(labels[indices["validation"]],model.predict_proba(transformed[indices["validation"]]),"coarse3")
                candidates.append({"C":c,"validation":result})
                key = (result["coarse3"]["balanced_accuracy"],-result["coarse3"]["balanced_log_loss"])
                if best_key is None or key>best_key:
                    best_key,best,selected = key,model,c
            probability = best.predict_proba(transformed[indices["test"]])
            frame = table.iloc[indices["test"]][["trial_id","subject_id","original_label"]].copy()
            frame["condition"],frame["fold"],frame["seed"] = f"linear_{representation}",fold["fold"],42
            for c in range(3):
                frame[f"p_{c}"] = probability[:,c]
            rows.append(frame)
            reports.append({"condition":f"linear_{representation}","fold":fold["fold"],"selected_C":selected,
                            "candidates":candidates,"test":common_metrics(labels[indices["test"]],probability,"coarse3"),
                            "coefficients":best.coef_.tolist(),"intercept":best.intercept_.tolist(),
                            "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()})
    pd.concat(rows,ignore_index=True).to_csv(output/"linear_predictions.csv",index=False)
    write_json(output/"linear_models.json",reports)


def probabilities(frame,objective):
    return frame[[f"p_{i}" for i in range(4 if objective=="native4" else 3)]].to_numpy(dtype=float)


def analyze(output):
    reports, tables = {},{}
    for architecture,representation,objective in CONDITIONS:
        name = f"{architecture}_{representation}_{objective}"
        frame = pd.read_csv(output/f"predictions_{name}.csv")
        tables[name] = frame
        by_seed = {str(seed):common_metrics(part.original_label.to_numpy(),probabilities(part,objective),objective)
                   for seed,part in frame.groupby("seed")}
        reports[name] = {"seeds":by_seed,"mean_coarse3_BA":float(np.mean([m["coarse3"]["balanced_accuracy"] for m in by_seed.values()])),
                         "mean_binary_BA":float(np.mean([m["binary"]["balanced_accuracy"] for m in by_seed.values()]))}
    linear = pd.read_csv(output/"linear_predictions.csv")
    for name,part in linear.groupby("condition"):
        reports[name] = common_metrics(part.original_label.to_numpy(),probabilities(part,"coarse3"),"coarse3")
    subjects = sorted(next(iter(tables.values())).subject_id.unique())
    draws = np.random.default_rng(20261006).integers(0,len(subjects),size=(10000,len(subjects)))
    def counts(name,task):
        frame = tables[name]
        objective = name.rsplit("_",1)[1]
        result = []
        for seed in [42,43,44]:
            blocks = []
            for subject in subjects:
                part = frame[(frame.seed==seed)&(frame.subject_id==subject)]
                y = part.original_label.to_numpy()
                p = coarse_probabilities(probabilities(part,objective),objective)
                if task=="binary":
                    eligible = y!=0
                    y = (y[eligible]==3).astype(int)
                    # Conditional normalization does not change binary argmax.
                    p = p[eligible,1:]
                else:
                    y = COARSE[y]
                blocks.append(np.array([[(p.argmax(1)[y==c]==c).sum(),(y==c).sum()] for c in range(p.shape[1])]))
            result.append(blocks)
        return np.asarray(result)
    def paired(a,b,task):
        ca,cb = counts(a,task),counts(b,task)
        def boot(c):
            pooled = c[:,draws].sum(2)
            return (pooled[...,0]/pooled[...,1]).mean(-1).mean(0)
        differences = boot(ca)-boot(cb)
        key = "mean_binary_BA" if task=="binary" else "mean_coarse3_BA"
        return {"a":a,"b":b,"task":task,"mean_BA_difference":reports[a][key]-reports[b][key],
                "paired_subject_percentile_95":np.quantile(differences,[.025,.975]).tolist(),
                "bootstrap_draws":len(draws),"participants":len(subjects),"multiplicity_corrected":False}
    pairs = []
    for task in ["coarse3","binary"]:
        for representation in ["absolute","relative"]:
            for objective in ["coarse3","native4"]:
                pairs.append(paired(f"transformer_{representation}_{objective}",f"mean_mlp_{representation}_{objective}",task))
        for architecture in ["mean_mlp","transformer"]:
            for objective in ["coarse3","native4"]:
                pairs.append(paired(f"{architecture}_relative_{objective}",f"{architecture}_absolute_{objective}",task))
            for representation in ["absolute","relative"]:
                pairs.append(paired(f"{architecture}_{representation}_native4",f"{architecture}_{representation}_coarse3",task))
    result = {"development_only":True,"research_question_change_approved":False,"conditions":reports,"paired_comparisons":pairs,
              "uncertainty_scope":"Participant blocks, fixed folds and three initialization seeds; 24 exploratory unadjusted comparisons, no independent confirmatory cohort"}
    write_json(output/"comparison.json",result)
    return result


def verify(cache,output,device="cuda"):
    x,table,info = load(cache)
    config = json.loads((output/"config.json").read_text())
    if info["fingerprint"] != config["cache_fingerprint"] or sha256(output/"plan.json") != config["plan_sha256"]:
        raise ValueError("Changed cache or plan")
    for path,recorded in config["source_hashes"].items():
        if sha256(Path(__file__).resolve().parents[1]/path) != recorded:
            raise ValueError("Changed bound source")
    folds = json.loads((output/"folds.json").read_text())
    if folds != make_folds(table.subject_id.tolist()):
        raise ValueError("Changed participant folds")
    records = json.loads((output/"model_metrics.json").read_text())
    if len(records)!=120:
        raise ValueError("Incomplete factorial experiment")
    labels = table.original_label.to_numpy(dtype=int)
    errors,paired_batches = [],{}
    seed_everything(42)
    for record in records:
        name,seed,fold_number = record["condition"],record["seed"],record["fold"]
        folder = output/"models"/f"{name}_seed{seed}_fold{fold_number}"
        for filename,key in [("selected.pt","checkpoint_sha256"),("history.json","history_sha256"),("predictions.csv","predictions_sha256")]:
            if sha256(folder/filename)!=record[key]:
                raise ValueError("Changed run bundle")
        checkpoint = torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
        fold = folds[fold_number]
        indices = {part:np.flatnonzero(table.subject_id.isin(fold[part])) for part in ("train","validation","test")}
        batches = draw_batches(labels,indices["train"],seed+fold_number*1000)
        if digest(batches.tolist())!=record["batch_digest"] or digest(batches[:record["selected_step"]].tolist())!=record["selected_batch_prefix_digest"]:
            raise ValueError("Changed or unpaired training draw stream")
        paired_batches.setdefault((seed,fold_number),set()).add(record["batch_digest"])
        transformed = represent(x,checkpoint["representation"])
        scaler = StandardScaler().fit(transformed[indices["train"]].reshape(-1,56))
        np.testing.assert_array_equal(scaler.mean_,checkpoint["mean"])
        np.testing.assert_array_equal(scaler.scale_,checkpoint["scale"])
        data = torch.from_numpy(scaler.transform(transformed.reshape(-1,56)).reshape(x.shape).astype(np.float32)).to(device)
        model = TemporalControl(checkpoint["architecture"],checkpoint["objective"]).to(device)
        model.load_state_dict(checkpoint["state"])
        with sdpa_kernel(SDPBackend.MATH):
            p = infer(model,data[indices["test"]])
            v = common_metrics(labels[indices["validation"]],infer(model,data[indices["validation"]]),checkpoint["objective"])
            training = common_metrics(labels[indices["train"]],infer(model,data[indices["train"]]),checkpoint["objective"])
        saved = pd.read_csv(folder/"predictions.csv")
        expected_trials = table.iloc[indices["test"]]
        if saved.trial_id.tolist()!=expected_trials.trial_id.tolist() or saved.subject_id.tolist()!=expected_trials.subject_id.tolist() or not np.array_equal(saved.original_label,labels[indices["test"]]):
            raise ValueError("Incorrect test coverage or labels")
        history = json.loads((folder/"history.json").read_text())
        best = max(history,key=lambda h:(h["validation"]["coarse3"]["balanced_accuracy"],-h["validation"]["coarse3"]["balanced_log_loss"]))
        if best["step"]!=record["selected_step"] or best["validation"]!=v or training!=record["training"]:
            raise ValueError("Incorrect validation selection or training metrics")
        maximum = float(np.max(np.abs(p-probabilities(saved,checkpoint["objective"]))))
        if maximum>1e-6 or common_metrics(labels[indices["test"]],p,checkpoint["objective"])!=record["test"]:
            raise ValueError("Checkpoint prediction/metric replay failed")
        errors.append(maximum)
        del model,data
    if any(len(values)!=1 for values in paired_batches.values()):
        raise ValueError("Training batches differ among factorial arms")
    for architecture,representation,objective in CONDITIONS:
        name = f"{architecture}_{representation}_{objective}"
        merged = pd.read_csv(output/f"predictions_{name}.csv")
        for seed,part in merged.groupby("seed"):
            if len(part)!=1080 or part.trial_id.duplicated().any() or set(part.trial_id)!=set(table.trial_id):
                raise ValueError("Incomplete OOF coverage")
        pieces = [pd.read_csv(output/"models"/f"{name}_seed{seed}_fold{i}"/"predictions.csv") for seed in [42,43,44] for i in range(5)]
        np.testing.assert_allclose(probabilities(merged,objective),probabilities(pd.concat(pieces,ignore_index=True),objective),atol=1e-15,rtol=0)
    linear_models = json.loads((output/"linear_models.json").read_text())
    predictions = pd.read_csv(output/"linear_predictions.csv")
    for record in linear_models:
        representation = record["condition"].removeprefix("linear_")
        features = table.windows.to_numpy()[:,None].astype(float) if representation=="duration" else represent(x,representation).mean(1)
        fold = folds[record["fold"]]
        indices = {part:np.flatnonzero(table.subject_id.isin(fold[part])) for part in ("train","validation","test")}
        scaler = StandardScaler().fit(features[indices["train"]])
        np.testing.assert_array_equal(scaler.mean_,record["scaler_mean"])
        np.testing.assert_array_equal(scaler.scale_,record["scaler_scale"])
        best = max(record["candidates"],key=lambda c:(c["validation"]["coarse3"]["balanced_accuracy"],-c["validation"]["coarse3"]["balanced_log_loss"]))
        if best["C"]!=record["selected_C"]:
            raise ValueError("Incorrect linear C selection")
        for part in ("validation","test"):
            logits = scaler.transform(features[indices[part]])@np.array(record["coefficients"]).T+record["intercept"]
            p = np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
            expected_metric = best["validation"] if part=="validation" else record["test"]
            if common_metrics(labels[indices[part]],p,"coarse3")!=expected_metric:
                # Floating-point order can perturb log loss below 1e-12, but no classification metric.
                actual = common_metrics(labels[indices[part]],p,"coarse3")
                for task in ("coarse3","binary"):
                    for key in actual[task]:
                        if key=="balanced_log_loss":
                            if abs(actual[task][key]-expected_metric[task][key])>1e-10:
                                raise ValueError("Linear log-loss replay failed")
                        elif actual[task][key]!=expected_metric[task][key]:
                            raise ValueError("Linear metric replay failed")
            if part=="test":
                saved = predictions[(predictions.condition==record["condition"])&(predictions.fold==record["fold"])]
                if saved.trial_id.tolist()!=table.iloc[indices[part]].trial_id.tolist():
                    raise ValueError("Incorrect linear test coverage")
                np.testing.assert_allclose(p,probabilities(saved,"coarse3"),rtol=0,atol=1e-12)
    original = json.loads((output/"comparison.json").read_text())
    if analyze(output)!=original:
        raise ValueError("Aggregate/bootstrap analysis changed")
    result = {"passed":True,"checkpoint_replays":120,"neural_probability_rows":25920,"linear_models_replayed":15,
              "linear_probability_rows":3240,"max_neural_probability_error":max(errors),
              "paired_sampling_streams":len(paired_batches),"scope":"Source/cache hashes, complete participant folds, target-training-only scalers, paired identical 600-batch streams, validation selection and checkpoint test/validation/training replay, linear coefficients, aggregate/bootstrap analysis",
              "limitation":"Linear coefficient replay does not independently refit all candidates; historical evaluation cohorts are development data"}
    write_json(output/"verification.json",result)
    return result
