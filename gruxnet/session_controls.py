"""Source-session-only selection and participant-separated SEED-IV transfer.

Adds a new experiment without changing any source bound to preceding results.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn.attention import SDPBackend,sdpa_kernel
from .data import digest,sha256,write_json
from .train import seed_everything
from .temporal_controls import COARSE,TemporalControl,common_metrics,infer,load,make_folds,C_VALUES


def source_masks(table,fold,source):
    result={part:np.flatnonzero(table.subject_id.isin(fold[part]) & (True if source==0 else table.session.eq(source)))
            for part in ("train","validation")}
    result["test"]={session:np.flatnonzero(table.subject_id.isin(fold["test"]) & table.session.eq(session)) for session in (1,2,3)}
    if any(set(fold[a])&set(fold[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]):
        raise ValueError("Participant leakage")
    for target,indices in result["test"].items():
        if source not in (0,target):
            selected=pd.concat([table.iloc[result[p]] for p in ("train","validation")])
            source_keys=set(zip(selected.session,selected.trial_id.str.rsplit(':',n=1).str[-1]))
            tested=table.iloc[indices]
            if source_keys&set(zip(tested.session,tested.trial_id.str.rsplit(':',n=1).str[-1])):
                raise ValueError("Held-out stimulus keys in training/selection")
    return result


def paired_batches(table,indices,seed,updates=600):
    selected=table.iloc[indices].copy()
    selected["position"]=indices
    selected=selected.sort_values(["subject_id","original_label","trial_id"])
    selected["emotion_rank"]=selected.groupby(["subject_id","original_label"]).cumcount()
    selected["canonical_key"]=[f"{r.subject_id}:{r.original_label}:{r.emotion_rank}" for r in selected.itertuples()]
    labels=table.original_label.to_numpy(dtype=int)
    pools=[selected.loc[COARSE[selected.original_label.to_numpy(dtype=int)]==c,"position"].to_numpy() for c in range(3)]
    rng=np.random.default_rng(seed)
    batches=np.stack([np.concatenate([rng.choice(pool,20,replace=True) for pool in pools]) for _ in range(updates)])
    keys=selected.set_index("position").canonical_key.to_dict()
    signature=digest([[keys[int(i)] for i in row] for row in batches])
    if any(np.bincount(COARSE[labels[row]],minlength=3).tolist()!=[20,20,20] for row in batches):
        raise ValueError("Unbalanced source classes")
    return batches,signature


def plan():
    return {"created_utc":datetime.now(timezone.utc).isoformat(),"development_only":True,"research_question_change_approved":False,
            "cohort":"All 1080 SEED-IV original trials; same fixed 40-second common14 cache and coarse3 target",
            "models":["mean_mlp","transformer"],"representation":"absolute log-bandpower only",
            "seeds":[42,43,44],"folds":"Same five seed42 rotations of 9/3/3 participants; all sessions of test participants excluded from fitting/selection",
            "source_sessions":[1,2,3],"test_sessions":[1,2,3],
            "selection_access":"Train nine participants and validate three others using only the chosen source session. No other-session trial/covariate/label used for scaling, fitting or selection",
            "paired_design":"90 fitted neural models, each selected once and evaluated on all three sessions of the same held-out people. Same-session and different-session results use identical target test trials when comparing source models",
            "sampling":"Batch60, 20 per coarse class; identical participant/native-emotion/within-emotion-rank draw stream across source sessions and architectures for a fold/seed",
            "training":"Identical preceding TemporalControl architectures, AdamW lr .001 weight decay .01 clip1; 600 updates, validation every25, source-validation common three-class BA then balanced logloss; no early stop/scheduler/AMP/augmentation",
            "normalization":"Per-feature scaler fitted on chosen source session's training participants' windows only",
            "summary":"Same-session diagonal; two cyclic unseen-source directions: source=(test % 3)+1 and source=((test+1)%3)+1. Each direction covers all1080 test trials once/seed. Average direction BAs, not probability ensemble or independent duplicate trials",
            "linear":"Absolute prefix-mean bandpower and scalar duration logistic controls; same source/folds, C .01/.1/1/10, balanced class weights, source-only validation. 30 selected models,120 candidates,6480 test probability rows",
            "uncertainty":"10000 paired participant-block percentile draws seed20261006; average seeds and source directions within each draw; exploratory unadjusted intervals, fixed folds",
            "limitation":"Sessions change clips and recording conditions together; source keys (session,trial) follow published SEED-IV design, not an independent stimulus-file hash audit. Previously inspected cohort, no causal isolation or unseen-dataset claim"}


def save_config(output,cache_info,plan_path,device):
    sources=["gruxnet/session_controls.py","gruxnet/temporal_controls.py","gruxnet/train.py","scripts/native_seediv_diagnostic.py"]
    config={"cache_fingerprint":cache_info["fingerprint"],"plan_sha256":sha256(plan_path),
            "source_hashes":{p:sha256(Path(__file__).resolve().parents[1]/p) for p in sources},
            "torch_version":torch.__version__,"device":device,"device_name":torch.cuda.get_device_name() if device=="cuda" else "CPU"}
    write_json(output/"config.json",config)
    return config


def prediction_frame(table,indices,p,name,seed,fold,source,target):
    result=table.iloc[indices][["trial_id","subject_id","original_label","session"]].copy()
    result["model"],result["seed"],result["fold"],result["source_session"],result["test_session"]=name,seed,fold,source,target
    for c in range(3):
        result[f"p_{c}"]=p[:,c]
    return result


def run(cache,output,plan_path,device="cuda"):
    if (output/"verification.json").exists():
        raise FileExistsError("Completed session experiment exists")
    x,table,info=load(cache)
    declaration=json.loads(plan_path.read_text())
    for key,value in plan().items():
        if key!="created_utc" and declaration[key]!=value:
            raise ValueError(f"Changed plan: {key}")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"plan.json",declaration)
    save_config(output,info,plan_path,device)
    folds=make_folds(table.subject_id.tolist())
    write_json(output/"folds.json",folds)
    labels=table.original_label.to_numpy(dtype=int)
    target=torch.from_numpy(COARSE[labels]).to(device)
    records=[]
    for architecture in ("mean_mlp","transformer"):
        for source in (1,2,3):
            rows=[]
            for seed in (42,43,44):
                for fold in folds:
                    identifier=f"{architecture}_source{source}_seed{seed}_fold{fold['fold']}"
                    folder=output/"models"/identifier
                    folder.mkdir(parents=True,exist_ok=True)
                    masks=source_masks(table,fold,source)
                    scaler=StandardScaler().fit(x[masks["train"]].reshape(-1,56))
                    data=torch.from_numpy(scaler.transform(x.reshape(-1,56)).reshape(x.shape).astype(np.float32)).to(device)
                    initialization=seed+fold["fold"]*1000
                    batches,signature=paired_batches(table,masks["train"],initialization)
                    if (folder/"metrics.json").exists():
                        record=json.loads((folder/"metrics.json").read_text())
                        if record["config_sha256"]!=sha256(output/"config.json"):
                            raise ValueError("Changed resumed run")
                        predictions=pd.read_csv(folder/"predictions.csv")
                    else:
                        seed_everything(initialization)
                        model=TemporalControl(architecture,"coarse3").to(device)
                        seed_everything(initialization+1000000)
                        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                        best_key,state,selected=None,None,None
                        history=[]
                        torch.cuda.reset_peak_memory_stats() if device=="cuda" else None
                        start=time.perf_counter()
                        with sdpa_kernel(SDPBackend.MATH):
                            for step,indices in enumerate(batches,1):
                                model.train(); optimizer.zero_grad(set_to_none=True)
                                loss=nn.functional.cross_entropy(model(data[indices]),target[indices])
                                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
                                if step%25==0:
                                    metrics=common_metrics(labels[masks["validation"]],infer(model,data[masks["validation"]]),"coarse3")
                                    history.append({"step":step,"training_loss":float(loss.item()),"validation":metrics})
                                    key=(metrics["coarse3"]["balanced_accuracy"],-metrics["coarse3"]["balanced_log_loss"])
                                    if best_key is None or key>best_key:
                                        best_key,state,selected=key,copy.deepcopy(model.state_dict()),step
                            model.load_state_dict(state)
                            training=common_metrics(labels[masks["train"]],infer(model,data[masks["train"]]),"coarse3")
                            tested,parts={},[]
                            for session,indices in masks["test"].items():
                                p=infer(model,data[indices])
                                tested[str(session)]=common_metrics(labels[indices],p,"coarse3")
                                parts.append(prediction_frame(table,indices,p,architecture,seed,fold["fold"],source,session))
                        predictions=pd.concat(parts,ignore_index=True)
                        predictions.to_csv(folder/"predictions.csv",index=False)
                        write_json(folder/"history.json",history)
                        torch.save({"state":{k:v.cpu() for k,v in state.items()},"mean":scaler.mean_,"scale":scaler.scale_,
                                    "source_session":source,"fold":fold,"seed":seed,"architecture":architecture,"selected_step":selected},folder/"selected.pt")
                        record={"model":architecture,"source_session":source,"seed":seed,"fold":fold["fold"],"selected_step":selected,
                                "training":training,"test_sessions":tested,"canonical_draw_signature":signature,"batch_digest":digest(batches.tolist()),
                                "elapsed_seconds":time.perf_counter()-start,"peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
                                "config_sha256":sha256(output/"config.json"),"checkpoint_sha256":sha256(folder/"selected.pt"),
                                "history_sha256":sha256(folder/"history.json"),"predictions_sha256":sha256(folder/"predictions.csv")}
                        write_json(folder/"metrics.json",record)
                        print(f"{identifier}: step={selected}, source_val3={best_key[0]:.4f}, diagonal3={tested[str(source)]['coarse3']['balanced_accuracy']:.4f}",flush=True)
                        del model,optimizer,state
                    rows.append(predictions); records.append(record)
                    del data
            pd.concat(rows,ignore_index=True).to_csv(output/f"predictions_{architecture}_source{source}.csv",index=False)
    write_json(output/"model_metrics.json",records)
    linear(x.mean(1),table,folds,output,"bandpower")
    linear(table.windows.to_numpy()[:,None].astype(float),table,folds,output,"duration")
    return analyze(output)


def linear(features,table,folds,output,name,include_mixed=False):
    labels=table.original_label.to_numpy(dtype=int)
    y=COARSE[labels]
    rows,records=[],[]
    for source in ([0,1,2,3] if include_mixed else [1,2,3]):
        for fold in folds:
            masks=source_masks(table,fold,source)
            scaler=StandardScaler().fit(features[masks["train"]])
            inputs=scaler.transform(features)
            best,best_key,candidates,selected=None,None,[],None
            for c in C_VALUES:
                estimator=LogisticRegression(C=c,class_weight="balanced",max_iter=2000,tol=1e-6,random_state=42)
                estimator.fit(inputs[masks["train"]],y[masks["train"]])
                metrics=common_metrics(labels[masks["validation"]],estimator.predict_proba(inputs[masks["validation"]]),"coarse3")
                candidates.append({"C":c,"validation":metrics})
                key=(metrics["coarse3"]["balanced_accuracy"],-metrics["coarse3"]["balanced_log_loss"])
                if best_key is None or key>best_key:
                    best,best_key,selected=estimator,key,c
            tests={}
            for session,indices in masks["test"].items():
                p=best.predict_proba(inputs[indices])
                tests[str(session)]=common_metrics(labels[indices],p,"coarse3")
                rows.append(prediction_frame(table,indices,p,name,42,fold["fold"],source,session))
            records.append({"model":name,"source_session":source,"fold":fold["fold"],"selected_C":selected,
                            "candidates":candidates,"test_sessions":tests,"coefficients":best.coef_.tolist(),"intercept":best.intercept_.tolist(),
                            "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()})
    # Separate model files by source session keep even 512-dimensional probes bounded for export.
    for source in ([0,1,2,3] if include_mixed else [1,2,3]):
        write_json(output/f"linear_{name}_source{source}.json",[r for r in records if r["source_session"]==source])
    pd.concat(rows,ignore_index=True).to_csv(output/f"predictions_{name}.csv",index=False)
    return records


def protocol_views(frame):
    views={"same_session":frame[frame.source_session==frame.test_session]}
    views["unseen_cycle1"]=frame[frame.source_session==(frame.test_session%3)+1]
    views["unseen_cycle2"]=frame[frame.source_session==((frame.test_session+1)%3)+1]
    if (frame.source_session==0).any():
        views["mixed_sessions"]=frame[frame.source_session==0]
    for part in views.values():
        for seed,rows in part.groupby("seed"):
            if len(rows)!=1080 or rows.trial_id.duplicated().any():
                raise ValueError("Incomplete or duplicated OOF protocol coverage")
    return views


def probabilities(frame):
    return frame[[f"p_{i}" for i in range(3)]].to_numpy(dtype=float)


def analysis_table(frames,bootstrap_seed=20261006):
    views={name:protocol_views(frame) for name,frame in frames.items()}
    report={}
    for name,protocols in views.items():
        records={}
        for protocol,frame in protocols.items():
            per_seed={str(seed):common_metrics(rows.original_label.to_numpy(),probabilities(rows),"coarse3") for seed,rows in frame.groupby("seed")}
            records[protocol]={"seeds":per_seed,"mean_coarse3_BA":float(np.mean([r["coarse3"]["balanced_accuracy"] for r in per_seed.values()])),
                               "mean_binary_BA":float(np.mean([r["binary"]["balanced_accuracy"] for r in per_seed.values()]))}
        records["mean_unseen"]={key:float(np.mean([records[p][key] for p in ("unseen_cycle1","unseen_cycle2")])) for key in ("mean_coarse3_BA","mean_binary_BA")}
        cells={}
        for (source,target),cell in frames[name].groupby(["source_session","test_session"]):
            per_seed={str(seed):common_metrics(rows.original_label.to_numpy(),probabilities(rows),"coarse3") for seed,rows in cell.groupby("seed")}
            cells[f"{source}->{target}"]={"seeds":per_seed,
                "mean_coarse3_BA":float(np.mean([r["coarse3"]["balanced_accuracy"] for r in per_seed.values()])),
                "mean_binary_BA":float(np.mean([r["binary"]["balanced_accuracy"] for r in per_seed.values()]))}
        records["source_test_cells"]=cells
        report[name]=records
    subjects=sorted(next(iter(frames.values())).subject_id.unique())
    draws=np.random.default_rng(bootstrap_seed).integers(0,len(subjects),size=(10000,len(subjects)))
    def bootstrap(name,protocol,task):
        pieces=[views[name][p] for p in (["unseen_cycle1","unseen_cycle2"] if protocol=="mean_unseen" else [protocol])]
        matrices=[]
        for frame in pieces:
            for _,seedrows in frame.groupby("seed"):
                blocks=[]
                for subject in subjects:
                    rows=seedrows[seedrows.subject_id==subject]
                    y=rows.original_label.to_numpy(); p=probabilities(rows)
                    if task=="binary":
                        selected=y!=0; y=(y[selected]==3).astype(int); p=p[selected,1:]
                    else:
                        y=COARSE[y]
                    blocks.append([[int((p.argmax(1)[y==c]==c).sum()),int((y==c).sum())] for c in range(p.shape[1])])
                matrices.append(blocks)
        counts=np.asarray(matrices)
        total=counts[:,draws].sum(2)
        return (total[...,0]/total[...,1]).mean(-1).mean(0)
    pairs=[]
    for task in ("coarse3","binary"):
        key="mean_coarse3_BA" if task=="coarse3" else "mean_binary_BA"
        for name in frames:
            delta=bootstrap(name,"mean_unseen",task)-bootstrap(name,"same_session",task)
            pairs.append({"model":name,"comparison":"mean_unseen minus same_session","task":task,
                          "mean_BA_difference":report[name]["mean_unseen"][key]-report[name]["same_session"][key],
                          "paired_participant_percentile_95":np.quantile(delta,[.025,.975]).tolist()})
        if "transformer" in frames and "mean_mlp" in frames:
            for protocol in ("same_session","mean_unseen"):
                delta=bootstrap("transformer",protocol,task)-bootstrap("mean_mlp",protocol,task)
                pairs.append({"model":"transformer minus mean_mlp","comparison":protocol,"task":task,
                              "mean_BA_difference":report["transformer"][protocol][key]-report["mean_mlp"][protocol][key],
                              "paired_participant_percentile_95":np.quantile(delta,[.025,.975]).tolist()})
    return {"development_only":True,"research_question_change_approved":False,"models":report,"paired_comparisons":pairs,
            "bootstrap":"10000 paired participant draws; seeds/directions averaged, no independent duplicate clips; intervals unadjusted and conditional on fixed folds",
            "scope":"Changing sessions changes stimulus materials and recording conditions together; source-only selection, no causal isolation"}


def analyze(output):
    frames={name:pd.concat([pd.read_csv(output/f"predictions_{name}_source{s}.csv") for s in (1,2,3)],ignore_index=True) for name in ("mean_mlp","transformer")}
    frames.update({name:pd.read_csv(output/f"predictions_{name}.csv") for name in ("bandpower","duration")})
    report=analysis_table(frames)
    write_json(output/"comparison.json",report)
    return report


def verify_linear(features,table,folds,output,name,include_mixed=False):
    labels=table.original_label.to_numpy(dtype=int)
    predictions=pd.read_csv(output/f"predictions_{name}.csv")
    protocol_views(predictions)
    maximum=0.
    for source in ([0,1,2,3] if include_mixed else [1,2,3]):
        for record in json.loads((output/f"linear_{name}_source{source}.json").read_text()):
            masks=source_masks(table,folds[record["fold"]],source)
            scaler=StandardScaler().fit(features[masks["train"]])
            np.testing.assert_array_equal(scaler.mean_,record["scaler_mean"])
            np.testing.assert_array_equal(scaler.scale_,record["scaler_scale"])
            best=max(record["candidates"],key=lambda r:(r["validation"]["coarse3"]["balanced_accuracy"],-r["validation"]["coarse3"]["balanced_log_loss"]))
            if best["C"]!=record["selected_C"]:
                raise ValueError("Incorrect C selection")
            for session,indices in masks["test"].items():
                logits=scaler.transform(features[indices])@np.asarray(record["coefficients"]).T+record["intercept"]
                p=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
                saved=predictions[(predictions.source_session==source)&(predictions.test_session==session)&(predictions.fold==record["fold"])]
                if saved.trial_id.tolist()!=table.iloc[indices].trial_id.tolist() or not np.array_equal(saved.original_label,labels[indices]):
                    raise ValueError("Incorrect linear coverage or labels")
                error=float(np.max(np.abs(p-probabilities(saved)))); maximum=max(maximum,error)
                if error>1e-10:
                    raise ValueError("Linear probability replay failed")
                actual=common_metrics(labels[indices],p,"coarse3")
                expected=record["test_sessions"][str(session)]
                for task,metrics in expected.items():
                    for key,value in metrics.items():
                        if key=="balanced_log_loss":
                            if abs(actual[task][key]-value)>1e-10:
                                raise ValueError("Linear test logloss differs")
                        elif actual[task][key]!=value:
                            raise ValueError("Linear test metrics differ")
            for task,metrics in best["validation"].items():
                logits=scaler.transform(features[masks["validation"]])@np.asarray(record["coefficients"]).T+record["intercept"]
                p=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
                actual=common_metrics(labels[masks["validation"]],p,"coarse3")[task]
                for key in metrics:
                    if key=="balanced_log_loss":
                        if abs(actual[key]-metrics[key])>1e-10:
                            raise ValueError("Linear validation replay failed")
                    elif actual[key]!=metrics[key]:
                        raise ValueError("Linear validation metrics differ")
    return {"models_replayed":20 if include_mixed else 15,"probability_rows":len(predictions),"max_probability_error":maximum,
            "scope":"Replay selected coefficients and source-only scaler/selection; does not independently refit every candidate"}


def verify(cache,output,device="cuda"):
    x,table,info=load(cache)
    config=json.loads((output/"config.json").read_text())
    if config["cache_fingerprint"]!=info["fingerprint"] or config["plan_sha256"]!=sha256(output/"plan.json"):
        raise ValueError("Changed source data or plan")
    for p,expected in config["source_hashes"].items():
        if sha256(Path(__file__).resolve().parents[1]/p)!=expected:
            raise ValueError("Changed bound source")
    folds=json.loads((output/"folds.json").read_text())
    if folds!=make_folds(table.subject_id.tolist()):
        raise ValueError("Changed folds")
    labels=table.original_label.to_numpy(dtype=int)
    records=json.loads((output/"model_metrics.json").read_text())
    if len(records)!=90:
        raise ValueError("Incomplete neural session study")
    signatures={}; maximum=0.
    seed_everything(42)
    for record in records:
        name,source,seed,fold_number=[record[k] for k in ("model","source_session","seed","fold")]
        folder=output/"models"/f"{name}_source{source}_seed{seed}_fold{fold_number}"
        for filename,key in [("selected.pt","checkpoint_sha256"),("history.json","history_sha256"),("predictions.csv","predictions_sha256")]:
            if sha256(folder/filename)!=record[key]:
                raise ValueError("Changed fitted run")
        masks=source_masks(table,folds[fold_number],source)
        batches,signature=paired_batches(table,masks["train"],seed+fold_number*1000)
        if signature!=record["canonical_draw_signature"] or digest(batches.tolist())!=record["batch_digest"]:
            raise ValueError("Changed sampling")
        signatures.setdefault((seed,fold_number),set()).add(signature)
        checkpoint=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
        scaler=StandardScaler().fit(x[masks["train"]].reshape(-1,56))
        np.testing.assert_array_equal(scaler.mean_,checkpoint["mean"]); np.testing.assert_array_equal(scaler.scale_,checkpoint["scale"])
        data=torch.from_numpy(scaler.transform(x.reshape(-1,56)).reshape(x.shape).astype(np.float32)).to(device)
        model=TemporalControl(name,"coarse3").to(device); model.load_state_dict(checkpoint["state"])
        history=json.loads((folder/"history.json").read_text())
        best=max(history,key=lambda h:(h["validation"]["coarse3"]["balanced_accuracy"],-h["validation"]["coarse3"]["balanced_log_loss"]))
        if best["step"]!=record["selected_step"]:
            raise ValueError("Incorrect neural selection")
        saved=pd.read_csv(folder/"predictions.csv")
        with sdpa_kernel(SDPBackend.MATH):
            if common_metrics(labels[masks["validation"]],infer(model,data[masks["validation"]]),"coarse3")!=best["validation"]:
                raise ValueError("Validation checkpoint replay failed")
            if common_metrics(labels[masks["train"]],infer(model,data[masks["train"]]),"coarse3")!=record["training"]:
                raise ValueError("Training checkpoint replay failed")
            for session,indices in masks["test"].items():
                p=infer(model,data[indices]); rows=saved[saved.test_session==session]
                if rows.trial_id.tolist()!=table.iloc[indices].trial_id.tolist() or not np.array_equal(rows.original_label,labels[indices]):
                    raise ValueError("Incorrect neural test coverage")
                error=float(np.max(np.abs(p-probabilities(rows)))); maximum=max(maximum,error)
                if error>1e-6 or common_metrics(labels[indices],p,"coarse3")!=record["test_sessions"][str(session)]:
                    raise ValueError("Neural test replay failed")
        del model,data
    if any(len(s)!=1 for s in signatures.values()):
        raise ValueError("Exposure draws differ across source sessions or architectures")
    for name in ("mean_mlp","transformer"):
        for source in (1,2,3):
            merged=pd.read_csv(output/f"predictions_{name}_source{source}.csv")
            parts=[pd.read_csv(output/"models"/f"{name}_source{source}_seed{seed}_fold{i}"/"predictions.csv") for seed in (42,43,44) for i in range(5)]
            np.testing.assert_allclose(probabilities(merged),probabilities(pd.concat(parts,ignore_index=True)),atol=1e-15,rtol=0)
    linear_results={"bandpower":verify_linear(x.mean(1),table,folds,output,"bandpower"),
                    "duration":verify_linear(table.windows.to_numpy()[:,None].astype(float),table,folds,output,"duration")}
    original=json.loads((output/"comparison.json").read_text())
    if analyze(output)!=original:
        raise ValueError("Changed OOF/bootstrap analysis")
    result={"passed":True,"neural_models_replayed":90,"neural_probability_rows":19440,"max_neural_probability_error":maximum,
            "paired_sampling_groups":len(signatures),"linear":linear_results,
            "scope":"Bound sources/cache, participant and source-only session access, identical canonical exposure draws, selected checkpoint train/validation/all-test-session replay, linear coefficients, OOF coverage and bootstrap aggregates"}
    write_json(output/"verification.json",result)
    return result
