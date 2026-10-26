"""Matched feature-level neural controls, separate from the historical GRU-XNet."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from .data import digest, sha256, write_json
from .deap_control import candidate_is_better
from .prepare import load_prepared
from .splits import make_split
from .train import scores, seed_everything

CONDITIONS = {"single": ["SEEDIV"], "plus_deap": ["SEEDIV", "DEAP"],
              "plus_game": ["SEEDIV", "GAMEEMO"], "joint": ["SEEDIV", "DEAP", "GAMEEMO"],
              "joint_heads": ["SEEDIV", "DEAP", "GAMEEMO"]}
DATASET_INDEX = {"SEEDIV": 0, "DEAP": 1, "GAMEEMO": 2}


class FeatureMLP(nn.Module):
    def __init__(self, separate_heads=False):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(56,64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(.2),
                                      nn.Linear(64,32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(.2))
        head = nn.Linear(32,2)
        self.heads = nn.ModuleList([head] + [deepcopy(head) for _ in range(2)] if separate_heads else [head])
        self.separate_heads = separate_heads

    def forward(self, features, datasets):
        if features.ndim != 2 or features.shape[1] != 56 or len(features) != len(datasets):
            raise ValueError("Expected trial features [batch,56] and dataset IDs")
        encoded = self.backbone(features)
        if not self.separate_heads:
            return self.heads[0](encoded)
        logits = torch.stack([head(encoded) for head in self.heads], dim=1)
        return logits[torch.arange(len(features),device=features.device),datasets]


def load_inputs(pack: Path, common_cache: Path):
    previous = json.loads((pack / "verification.json").read_text())
    if sha256(pack / "features.npz") != previous["features_sha256"]:
        raise ValueError("Previously verified feature pack changed")
    with np.load(pack / "features.npz", allow_pickle=False) as values:
        target_x, source_x = values["target"].copy(), values["others"].copy()
    target, sources = [pd.read_csv(pack / name) for name in ["target_trials.csv","source_trials.csv"]]
    manifest, prepared = load_prepared(common_cache,verify_files=True)
    original_split, _ = make_split(manifest,"subject",42)
    for table, features in [(target,target_x),(sources,source_x)]:
        if features.shape != (len(table),56) or not np.isfinite(features).all() or table.trial_id.duplicated().any():
            raise ValueError("Invalid trial feature/identifier alignment")
        original = manifest.groupby("trial_id",sort=False).first().loc[table.trial_id]
        for field in ["dataset","subject_id","label"]:
            np.testing.assert_array_equal(table[field],original[field])
    if set(target.dataset) != {"SEEDIV"} or len(target) != 810 or set(sources.dataset) != {"DEAP","GAMEEMO"}:
        raise ValueError("Unexpected target/source task")
    expected = original_split[(original_split.dataset != "SEEDIV") & (original_split.split == "train")]
    if set(sources.trial_id) != set(expected.trial_id):
        raise ValueError("Sources include missing trials or validation/test participants")
    if set(target.subject_id) & set(sources.subject_id):
        raise ValueError("Dataset-qualified participant collision")
    binding = {name:sha256(pack/name) for name in ["features.npz","target_trials.csv","source_trials.csv","config.json","verification.json"]}
    return target_x, target, source_x, sources, {"pack_sha256":binding,"common_cache_fingerprint":prepared["cache_fingerprint"]}


def balanced_batches(table, datasets, steps, seed, fold, batch_size=60):
    """Exact quotas; independent draw streams keep target exposure comparable."""
    if steps < 1 or batch_size % (2*len(datasets)):
        raise ValueError("Batch size must divide equally across datasets and binary classes")
    quota = batch_size // (2*len(datasets))
    blocks = []
    for dataset in datasets:
        for label in [0,1]:
            eligible = np.flatnonzero((table.dataset == dataset) & (table.label == label))
            if not len(eligible):
                raise ValueError("Missing original training class")
            rng = np.random.default_rng(np.random.SeedSequence([seed,fold,DATASET_INDEX[dataset],label]))
            draws = eligible[rng.integers(0,len(eligible),size=steps*quota)].reshape(steps,quota)
            blocks.append(draws)
    return np.concatenate(blocks,axis=1)


def frame_for(target_x,target,source_x,sources,fold,condition):
    masks = {part:target.subject_id.isin(fold[part]).to_numpy() for part in ["train","validation","test"]}
    selected = sources.dataset.isin(CONDITIONS[condition][1:]).to_numpy()
    table = pd.concat([target[masks["train"]],sources[selected]],ignore_index=True)
    features = np.concatenate([target_x[masks["train"]],source_x[selected]])
    scaler = StandardScaler().fit(target_x[masks["train"]])
    return masks, table, features, scaler


def balanced_log_loss(labels,probabilities):
    y, p = np.asarray(labels), np.clip(np.asarray(probabilities),1e-7,1-1e-7)
    losses = -(y*np.log(p)+(1-y)*np.log(1-p))
    return float(np.mean([losses[y==c].mean() for c in [0,1]]))


@torch.no_grad()
def predict(model,features,datasets):
    model.eval()
    return model(features,datasets).softmax(-1)[:,1].cpu().numpy()


def linear_controls(target_x,target,source_x,sources,folds,output):
    reports, predictions = {}, []
    for condition in ["single","plus_deap","plus_game","joint"]:
        results = []
        for fold in folds:
            masks,table,features,scaler = frame_for(target_x,target,source_x,sources,fold,condition)
            weights = np.zeros(len(table))
            for dataset in CONDITIONS[condition]:
                for label in [0,1]:
                    rows = (table.dataset==dataset)&(table.label==label)
                    weights[rows] = 1/rows.sum()
            # Fixed effective training weight avoids increasing loss relative to L2 just by adding rows.
            weights *= int(masks["train"].sum())/weights.sum()
            x = scaler.transform(features)
            validation_x,test_x = [scaler.transform(target_x[masks[p]]) for p in ["validation","test"]]
            candidates, best, best_score, best_c = [],None,-1.,None
            for c in [.01,.1,1.,10.]:
                model = LogisticRegression(C=c,solver="lbfgs",tol=1e-6,max_iter=2000,random_state=42)
                model.fit(x,table.label,sample_weight=weights)
                probabilities = model.predict_proba(validation_x)[:,1]
                result = scores(target.label[masks["validation"]],probabilities)
                candidates.append({"C":c,**result})
                if result["balanced_accuracy"]>best_score:
                    best,best_score,best_c = model,result["balanced_accuracy"],c
            probabilities = best.predict_proba(test_x)[:,1]
            test = target[masks["test"]].reset_index(drop=True)
            for index,row in test.iterrows():
                predictions.append({"condition":condition,"fold":fold["fold"],"subject_id":row.subject_id,
                                    "trial_id":row.trial_id,"label":int(row.label),"positive_probability":float(probabilities[index])})
            results.append({"fold":fold["fold"],"selected_C":best_c,"validation_candidates":candidates,
                            "test":scores(test.label,probabilities),"effective_training_weight":float(weights.sum())})
        rows = pd.DataFrame([r for r in predictions if r["condition"]==condition])
        reports[condition] = {"folds":results,"test_out_of_fold":scores(rows.label,rows.positive_probability)}
        print(f"Linear anchored {condition}: BA={reports[condition]['test_out_of_fold']['balanced_accuracy']:.4f}",flush=True)
    pd.DataFrame(predictions).to_csv(output/"linear_trial_predictions.csv",index=False)
    write_json(output/"linear_comparison.json",reports)
    return reports


def train_one(target_x,target,source_x,sources,fold,condition,seed,output,device,primary_steps=600):
    if output.exists():
        raise FileExistsError("Model run already exists")
    masks,table,features,scaler = frame_for(target_x,target,source_x,sources,fold,condition)
    n_datasets = len(CONDITIONS[condition])
    max_steps = primary_steps*n_datasets
    seed_everything(seed+1000*fold["fold"])
    torch.set_num_threads(2)
    model = FeatureMLP(condition=="joint_heads").to(device)
    backbone_hash = digest({k:v.detach().cpu().tolist() for k,v in model.backbone.state_dict().items()})
    head_hash = digest({k:v.detach().cpu().tolist() for k,v in model.heads[0].state_dict().items()})
    batches = balanced_batches(table,CONDITIONS[condition],max_steps,seed,fold["fold"])
    dataset_ids = table.dataset.map(DATASET_INDEX).to_numpy()
    output.mkdir(parents=True)
    np.savez(output/"normalizer.npz",mean=scaler.mean_,scale=scaler.scale_)
    table.to_csv(output/"training_trials.csv",index=False)
    config = {"condition":condition,"seed":seed,"fold":fold,"device":str(device),
              "datasets":CONDITIONS[condition],"primary_steps":primary_steps,"max_steps":max_steps,
              "batch_size":60,"learning_rate":.001,"weight_decay":.01,"validation_every":25,
              "separate_heads":condition=="joint_heads","parameters":sum(p.numel() for p in model.parameters()),
              "initial_backbone_sha256":backbone_hash,"initial_target_head_sha256":head_hash,
              "sampling_sha256":hashlib.sha256(batches.tobytes()).hexdigest(),
              "normalizer_sha256":sha256(output/"normalizer.npz"),"training_trials_sha256":sha256(output/"training_trials.csv"),
              "training_subjects":sorted(table.subject_id.unique()),"scaler_subjects":sorted(target[masks["train"]].subject_id.unique()),
              "target_presentations_primary":primary_steps*60//n_datasets,"target_presentations_max":max_steps*60//n_datasets}
    write_json(output/"config.json",config)
    training_x = torch.from_numpy(scaler.transform(features)).to(device)
    training_y = torch.tensor(table.label.to_numpy(),dtype=torch.long,device=device)
    training_d = torch.tensor(dataset_ids,dtype=torch.long,device=device)
    batch_tensor = torch.from_numpy(batches).to(device)
    target_training_x = torch.from_numpy(scaler.transform(target_x[masks["train"]])).to(device)
    validation_x = torch.from_numpy(scaler.transform(target_x[masks["validation"]])).to(device)
    target_training_d = torch.zeros(len(target_training_x),dtype=torch.long,device=device)
    validation_d = torch.zeros(len(validation_x),dtype=torch.long,device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best = {name:[-1.,float("inf"),None] for name in ["primary","exposure"]}
    history, rolling_loss = [],0.
    started = time.perf_counter()
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step,indices in enumerate(batch_tensor,start=1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(training_x[indices],training_d[indices])
        loss = nn.functional.cross_entropy(logits,training_y[indices])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        rolling_loss += float(loss.detach())
        if step%25:
            continue
        probabilities = predict(model,validation_x,validation_d)
        value = scores(target.label[masks["validation"]],probabilities)
        tie_loss = balanced_log_loss(target.label[masks["validation"]],probabilities)
        train_probabilities = predict(model,target_training_x,target_training_d)
        train_value = scores(target.label[masks["train"]],train_probabilities)
        record = {"step":step,"training_balanced_batch_loss":rolling_loss/25,
                  "target_training_BA":train_value["balanced_accuracy"],"validation":value,"validation_balanced_log_loss":tie_loss}
        rolling_loss=0.
        for name in ["primary","exposure"]:
            if name=="primary" and step>primary_steps:
                continue
            previous_score,previous_loss,_ = best[name]
            if candidate_is_better(value["balanced_accuracy"],tie_loss,previous_score,previous_loss):
                best[name]=[value["balanced_accuracy"],tie_loss,step]
                torch.save({"state_dict":model.state_dict(),"step":step,"validation":value,
                            "validation_balanced_log_loss":tie_loss},output/f"{name}.pt")
            record[f"best_{name}_step"]=best[name][2]
        history.append(record)
    write_json(output/"history.json",history)
    # Neither test nor test features are inferred until both validation selections are final.
    test_x = torch.from_numpy(scaler.transform(target_x[masks["test"]])).to(device)
    test_d = torch.zeros(len(test_x),dtype=torch.long,device=device)
    test = target[masks["test"]].reset_index(drop=True)
    records,results = [],{}
    for budget in ["primary","exposure"] if n_datasets>1 else ["primary"]:
        checkpoint = torch.load(output/f"{budget}.pt",map_location=device,weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities = predict(model,test_x,test_d)
        target_training = predict(model,target_training_x,target_training_d)
        results[budget] = {"selected_step":checkpoint["step"],"validation":checkpoint["validation"],
                           "train_target":scores(target.label[masks["train"]],target_training),
                           "test":scores(test.label,probabilities),"checkpoint_sha256":sha256(output/f"{budget}.pt")}
        for index,row in test.iterrows():
            records.append({"condition":condition,"seed":seed,"fold":fold["fold"],"budget":budget,
                            "subject_id":row.subject_id,"trial_id":row.trial_id,"label":int(row.label),
                            "positive_probability":float(probabilities[index])})
    predictions = pd.DataFrame(records)
    predictions.to_csv(output/"test_trial_predictions.csv",index=False)
    result = {"condition":condition,"seed":seed,"fold":fold["fold"],"budgets":results,
              "elapsed_seconds":time.perf_counter()-started,"predictions_sha256":sha256(output/"test_trial_predictions.csv"),
              "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated(device)/2**20 if device.type=="cuda" else 0}
    write_json(output/"metrics.json",result)
    print(f"{condition} seed={seed} fold={fold['fold']} primary_BA={results['primary']['test']['balanced_accuracy']:.4f} "
          f"exposure_BA={results.get('exposure',results['primary'])['test']['balanced_accuracy']:.4f} seconds={result['elapsed_seconds']:.1f}",flush=True)
    return predictions,result


def investigate(pack:Path,common_cache:Path,plan:Path,output:Path):
    if output.exists():
        raise FileExistsError("Use a fresh investigation directory")
    declaration = json.loads(plan.read_text())
    if declaration["seeds"]!=[42,43,44] or declaration["conditions"]!=list(CONDITIONS):
        raise ValueError("Different predeclared conditions/seeds")
    target_x,target,source_x,sources,binding = load_inputs(pack,common_cache)
    from scripts.native_seediv_diagnostic import make_folds
    folds = make_folds(target.subject_id.tolist())
    output.mkdir(parents=True)
    (output/"plan.json").write_bytes(plan.read_bytes())
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    source_hashes={}
    for path in [Path(__file__),Path(__file__).with_name("train.py"),Path(__file__).with_name("deap_control.py")]:
        (snapshot/path.name).write_bytes(path.read_bytes())
        source_hashes[path.name]=sha256(path)
    config = {"created_utc":datetime.now(timezone.utc).isoformat(),"development_only":True,
              "research_question_change_approved":False,"plan_sha256":sha256(plan),"input_binding":binding,
              "pack":str(pack.resolve()),"common_cache":str(common_cache.resolve()),"source_sha256":source_hashes,
              "hardware":{"device":"cuda","gpu":torch.cuda.get_device_name(0)},
              "versions":{"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__},
              "scope":"Feature-level MLP controls, not full GRU-XNet/recurrent/STFT ablations or unseen-dataset transfer"}
    write_json(output/"config.json",config)
    write_json(output/"folds.json",folds)
    linear_controls(target_x,target,source_x,sources,folds,output)
    frames,models=[],[]
    for seed in declaration["seeds"]:
        for fold in folds:
            for condition in CONDITIONS:
                name=f"{condition}_seed{seed}_fold{fold['fold']}"
                predictions,result=train_one(target_x,target,source_x,sources,fold,condition,seed,output/"models"/name,torch.device("cuda"))
                frames.append(predictions)
                models.append(result)
                pd.concat(frames,ignore_index=True).to_csv(output/"trial_predictions.csv",index=False)
                write_json(output/"model_metrics.json",models)
    all_predictions=pd.concat(frames,ignore_index=True)
    report={}
    for (condition,budget),rows in all_predictions.groupby(["condition","budget"]):
        seeds={str(seed):scores(group.label,group.positive_probability) for seed,group in rows.groupby("seed")}
        report[f"{condition}:{budget}"]={"per_seed":seeds,
                  "mean_seed_BA":float(np.mean([v["balanced_accuracy"] for v in seeds.values()])),
                  "std_seed_BA":float(np.std([v["balanced_accuracy"] for v in seeds.values()],ddof=1))}
    write_json(output/"neural_comparison.json",report)
    print(json.dumps({key:value['mean_seed_BA'] for key,value in report.items()}),flush=True)
    return report
