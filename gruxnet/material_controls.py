"""Matched material exposure within one session, with held-out participants.

Independent source module: existing experiment implementations remain unchanged.
"""
from __future__ import annotations
import copy
from datetime import datetime,timezone
from hashlib import sha256 as hasher
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn.attention import SDPBackend,sdpa_kernel
from .data import digest,sha256,write_json
from .temporal_controls import COARSE,C_VALUES,TemporalControl,common_metrics,infer,load,make_folds
from .session_controls import paired_batches,probabilities
from .reve_probe import load_features
from .train import seed_everything
from scripts.analyze_session_material_sensitivity import crossed_weights

REPO=Path(__file__).resolve().parents[1]
ARMS=("exposed","unexposed")
NEURAL=("mean_mlp","transformer")
LINEAR=("bandpower","duration","reve_pretrained","reve_random42")
SEEDS=(42,43,44)


def annotate(table):
    table=table.copy()
    table["material_key"]=table.session.astype(str)+":"+table.trial_id.str.rsplit(':',n=1).str[-1]
    if len(table)!=1080 or table.trial_id.duplicated().any() or table.subject_id.nunique()!=15:
        raise ValueError("Expected complete original cohort")
    if (table.groupby("material_key").original_label.nunique()!=1).any() or (table.groupby("material_key").subject_id.nunique()!=15).any():
        raise ValueError("Inconsistent material key/participant/label population")
    rng=np.random.default_rng(20261006); ranks={}
    materials=table[["material_key","session","original_label"]].drop_duplicates()
    for _,rows in materials.groupby(["session","original_label"],sort=True):
        keys=sorted(rows.material_key.tolist())
        if len(keys)!=6: raise ValueError("Expected six clips per session/native emotion")
        ranks.update({keys[int(i)]:rank for rank,i in enumerate(rng.permutation(6))})
    table["material_rank"]=table.material_key.map(ranks)
    return table


def assignment(rotation,arm):
    if rotation not in (0,1,2) or arm not in ARMS: raise ValueError("Unknown material condition")
    ordered=[(2*rotation+i)%6 for i in range(6)]
    return {"train":ordered[3:] if arm=="unexposed" else [ordered[0],ordered[1],ordered[3]],
            "validation":[ordered[2]],"test":ordered[:2]}


def masks(table,fold,session,rotation,arm):
    if any(set(fold[a])&set(fold[b]) for a,b in [("train","validation"),("train","test"),("validation","test")]):
        raise ValueError("Participant leakage")
    roles=assignment(rotation,arm)
    result={part:np.flatnonzero(table.subject_id.isin(fold[part]) & table.session.eq(session) & table.material_rank.isin(roles[part]))
            for part in ("train","validation","test")}
    if [len(result[p]) for p in ("train","validation","test")]!=[108,12,24]: raise ValueError("Incorrect matched trial counts")
    keys={p:set(table.iloc[idx].material_key) for p,idx in result.items()}
    if keys["validation"]&(keys["train"]|keys["test"]): raise ValueError("Validation material leakage")
    if arm=="unexposed" and keys["train"]&keys["test"]: raise ValueError("Held-out material in unexposed training")
    if arm=="exposed" and not keys["test"].issubset(keys["train"]): raise ValueError("Exposed arm does not include test materials")
    return result


def plan():
    return {"created_utc":datetime.now(timezone.utc).isoformat(),"development_only":True,"research_question_change_approved":False,
            "cohort":"Same1080 SEEDIV original trials,15people,3sessions,72published-design material keys; original video hashes unavailable",
            "input":"Existing common14 first40s absolute log-bandpower sequences; full-original-trial offline preprocessing. Reuse verified frozen REVE512 features and declared adapter (37s direct patches within40s normalized observation)",
            "material_order":"Within each session/native-emotion stratum sort six material keys, independently permute with one RNG seed20261006 in sorted stratum order; assignment fixed before fitting, not selected by results",
            "material_rotations":"r=0/1/2; cyclic rank order[(2r+i)%6 for i=0..5]. Test first2ranks/emotion; validation third; unexposed training final3; exposed training first2+fourth. One training clip/emotion shared,2replaced; all24session clips tested once over3rotations",
            "participants":"Same five seed42 9/3/3 rotations; test and validation participants excluded from training across all sessions",
            "matched_population":"Only one identical recording session in each pair;108training trials,12validation trials,24identical test trials. Training participants/native class counts and available budget identical. Validation clip unseen in both arms; no held-out population calibration",
            "neural_models":list(NEURAL),"initialization_seeds":list(SEEDS),"neural_fits":540,
            "neural_sampling":"600balanced60-trial batches,20percoarseclass. Pair participant/native-emotion/within-selected-emotion-trial-rank stream and initial model weights across arms/architectures/sessions for given fold,rotation,seed. RNG=seed+1000fold+10000rotation; reset stochastic model RNG+1000000",
            "neural_training":"Unchanged TemporalControl meanMLP19079/transformer19075parameters; AdamWlr.001wd.01clip1,600updates,validationevery25,source-validation3BA thenbalancedlogloss thenfirsttie; deterministicCUDAmathSDPA; noAMP/augmentation/scheduler/earlystop",
            "scaling":"Neural per-feature StandardScaler only chosen training windows; each linear feature scaler only chosen training trials. No statistic pooled over val/test people or clips",
            "linear_features":list(LINEAR),"linear_selected_heads":360,"linear_candidates":1440,
            "linear_training":"Coarse3 balanced multinomial logistic,C.01/.1/1/10,max_iter2000tol1e-6; validation3BAthenbalancedlogloss,firsttie. Source/participant/material access identical to neural; frozenREVE encoders unchanged, one random encoder seed42",
            "test_probability_rows":{"neural":12960,"linear":8640,"total":21600},
            "evaluation":"Primary3classvalence/all1080; secondaryconditionalbinary/810. Each model/arm/seed tests each original trial once across participant folds and material rotations. Average neural seeds, never select a winner by test results",
            "uncertainty":"10000pairedparticipant and crossedparticipant/material percentile draws seed20261006; six material draws per12session/nativeemotion strata. Same weights across arms/models, average seed correctness within each cell, conditionalfixedmodels/folds/sessions; unadjusted exploratory intervals",
            "contrasts":"Unexposed-minus-exposed foreach6models; transformer-minus-MLP foreacharm; pretrained-minus-randomforeacharm; bothtasks,20contrasts. Report all seeds,session/rotation cells and validation noise",
            "scope":"Within-session exposure protocol controls training size and session index, not pure causal videoidentity. Train clip difficulty/order/content/duration still change; cohort already inspected. No unseen-corpus/pooledtraining/new-method claim; independent full REVE checkpoint exclusion remains uncertified"}


def features(cache,reve_run):
    x,original,info=load(cache)
    frozen,other,record=load_features(reve_run)
    if not original.equals(other) or not json.loads((reve_run/"verification.json").read_text())["passed"]:
        raise ValueError("Unverified/mismatched frozen feature population")
    table=annotate(original)
    values={"bandpower":x.mean(1),"duration":table.windows.to_numpy()[:,None].astype(float),**frozen}
    if any(v.shape[0]!=1080 or not np.isfinite(v).all() for v in values.values()): raise ValueError("Invalid features")
    return x,table,info,values


def state_digest(state):
    h=hasher()
    for name,value in state.items(): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def frame(table,indices,p,model,arm,session,rotation,seed,fold):
    rows=table.iloc[indices][["trial_id","subject_id","session","original_label","material_key","material_rank"]].copy()
    rows["model"],rows["arm"],rows["source_session"],rows["material_rotation"],rows["seed"],rows["fold"]=model,arm,session,rotation,seed,fold
    for c in range(3): rows[f"p_{c}"]=p[:,c]
    return rows


def configuration(info,reve_run,plan_path,device):
    sources=["gruxnet/material_controls.py","gruxnet/temporal_controls.py","gruxnet/session_controls.py","gruxnet/train.py",
             "gruxnet/data.py","gruxnet/reve_probe.py","scripts/native_seediv_diagnostic.py","scripts/analyze_session_material_sensitivity.py"]
    return {"temporal_cache_fingerprint":info["fingerprint"],"reve_features_metadata_sha256":sha256(reve_run/"features.json"),
            "reve_verification_sha256":sha256(reve_run/"verification.json"),"plan_sha256":sha256(plan_path),
            "source_hashes":{name:sha256(REPO/name) for name in sources},"torch_version":torch.__version__,
            "device":device,"device_name":torch.cuda.get_device_name() if device=="cuda" else "CPU"}


def fit_neural(x,table,fold,session,rotation,arm,model_name,seed,output,device):
    identifier=f"{model_name}_{arm}_session{session}_rotation{rotation}_seed{seed}_fold{fold['fold']}"
    folder=output/"models"/identifier; folder.mkdir(parents=True,exist_ok=True)
    idx=masks(table,fold,session,rotation,arm); init=seed+fold["fold"]*1000+rotation*10000
    batches,signature=paired_batches(table,idx["train"],init)
    if (folder/"metrics.json").exists():
        record=json.loads((folder/"metrics.json").read_text())
        if record["config_sha256"]!=sha256(output/"config.json"): raise ValueError("Changed resumed fit")
        return record,pd.read_csv(folder/"predictions.csv")
    scaler=StandardScaler().fit(x[idx["train"]].reshape(-1,56))
    data=torch.from_numpy(scaler.transform(x.reshape(-1,56)).reshape(x.shape).astype(np.float32)).to(device)
    labels=table.original_label.to_numpy(dtype=int); target=torch.from_numpy(COARSE[labels]).to(device)
    seed_everything(init); model=TemporalControl(model_name,"coarse3").to(device); initial_hash=state_digest(model.state_dict())
    seed_everything(init+1000000); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    best_key,state,selected=None,None,None; history=[]
    if device=="cuda": torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        for step,indices in enumerate(batches,1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(data[indices]),target[indices]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            if step%25==0:
                val=common_metrics(labels[idx["validation"]],infer(model,data[idx["validation"]]),"coarse3")
                history.append({"step":step,"training_loss":float(loss.item()),"validation":val})
                key=(val["coarse3"]["balanced_accuracy"],-val["coarse3"]["balanced_log_loss"])
                if best_key is None or key>best_key: best_key,state,selected=key,copy.deepcopy(model.state_dict()),step
        model.load_state_dict(state)
        training=common_metrics(labels[idx["train"]],infer(model,data[idx["train"]]),"coarse3")
        p=infer(model,data[idx["test"]]); tested=common_metrics(labels[idx["test"]],p,"coarse3")
    rows=frame(table,idx["test"],p,model_name,arm,session,rotation,seed,fold["fold"])
    rows.to_csv(folder/"predictions.csv",index=False); write_json(folder/"history.json",history)
    torch.save({"state":{k:v.cpu() for k,v in state.items()},"mean":scaler.mean_,"scale":scaler.scale_,
                "selected_step":selected,"model":model_name,"arm":arm,"fold":fold,"session":session,"rotation":rotation,"seed":seed},folder/"selected.pt")
    record={"id":identifier,"model":model_name,"arm":arm,"session":session,"rotation":rotation,"seed":seed,"fold":fold["fold"],
            "training":training,"test":tested,"selected_step":selected,"initial_state_sha256":initial_hash,
            "canonical_draw_signature":signature,"batch_digest":digest(batches.tolist()),"elapsed_seconds":time.perf_counter()-start,
            "peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,"config_sha256":sha256(output/"config.json"),
            "checkpoint_sha256":sha256(folder/"selected.pt"),"history_sha256":sha256(folder/"history.json"),"predictions_sha256":sha256(folder/"predictions.csv")}
    write_json(folder/"metrics.json",record)
    del model,optimizer,state,data,target
    return record,rows


def fit_linear(values,table,folds,output,name):
    labels=table.original_label.to_numpy(dtype=int); y=COARSE[labels]; parts={arm:[] for arm in ARMS}
    for session in (1,2,3):
        for rotation in (0,1,2):
            records=[]
            for arm in ARMS:
                for fold in folds:
                    idx=masks(table,fold,session,rotation,arm); scaler=StandardScaler().fit(values[idx["train"]]); inputs=scaler.transform(values)
                    best_key,best,selected=None,None,None; candidates=[]
                    for c in C_VALUES:
                        estimator=LogisticRegression(C=c,class_weight="balanced",max_iter=2000,tol=1e-6,random_state=42)
                        estimator.fit(inputs[idx["train"]],y[idx["train"]])
                        val=common_metrics(labels[idx["validation"]],estimator.predict_proba(inputs[idx["validation"]]),"coarse3")
                        candidates.append({"C":c,"validation":val}); key=(val["coarse3"]["balanced_accuracy"],-val["coarse3"]["balanced_log_loss"])
                        if best_key is None or key>best_key: best_key,best,selected=key,estimator,c
                    p=best.predict_proba(inputs[idx["test"]]); tested=common_metrics(labels[idx["test"]],p,"coarse3")
                    parts[arm].append(frame(table,idx["test"],p,name,arm,session,rotation,42,fold["fold"]))
                    records.append({"model":name,"arm":arm,"session":session,"rotation":rotation,"fold":fold["fold"],"selected_C":selected,
                                    "candidates":candidates,"test":tested,"coefficients":best.coef_.tolist(),"intercept":best.intercept_.tolist(),
                                    "scaler_mean":scaler.mean_.tolist(),"scaler_scale":scaler.scale_.tolist()})
            write_json(output/f"linear_{name}_session{session}_rotation{rotation}.json",records)
    for arm,rows in parts.items(): pd.concat(rows,ignore_index=True).to_csv(output/f"predictions_{name}_{arm}.csv",index=False)
    print(f"Completed90selected/360candidate {name} heads",flush=True)


def run(cache,reve_run,output,plan_path,device="cuda"):
    if (output/"verification.json").exists(): raise FileExistsError("Completed experiment")
    x,table,info,values=features(cache,reve_run); declaration=json.loads(plan_path.read_text())
    if {k:v for k,v in declaration.items() if k!="created_utc"}!={k:v for k,v in plan().items() if k!="created_utc"}: raise ValueError("Changed declaration")
    config=configuration(info,reve_run,plan_path,device); output.mkdir(parents=True,exist_ok=True)
    if (output/"config.json").exists() and json.loads((output/"config.json").read_text())!=config: raise ValueError("Changed resumed config")
    write_json(output/"config.json",config); write_json(output/"plan.json",declaration)
    folds=make_folds(table.subject_id.tolist()); write_json(output/"folds.json",folds)
    table[["material_key","session","original_label","material_rank"]].drop_duplicates().sort_values("material_key").to_csv(output/"material_assignments.csv",index=False)
    index=[]; completed=0
    for name in NEURAL:
        for session in (1,2,3):
            records=[]; parts={arm:[] for arm in ARMS}
            for rotation in (0,1,2):
                for seed in SEEDS:
                    for fold in folds:
                        for arm in ARMS:
                            record,rows=fit_neural(x,table,fold,session,rotation,arm,name,seed,output,device)
                            records.append(record); parts[arm].append(rows); completed+=1
                            index.append({"id":record["id"],"metrics_sha256":sha256(output/"models"/record["id"]/"metrics.json")})
                            if completed%10==0: print(f"Neural fits{completed}/540: {name},session{session},rotation{rotation},seed{seed}",flush=True)
            write_json(output/f"model_metrics_{name}_session{session}.json",records)
            for arm,rows in parts.items(): pd.concat(rows,ignore_index=True).to_csv(output/f"predictions_{name}_{arm}_session{session}.csv",index=False)
    write_json(output/"model_index.json",index)
    for name in NEURAL:
        for arm in ARMS:
            pd.concat([pd.read_csv(output/f"predictions_{name}_{arm}_session{s}.csv") for s in (1,2,3)],ignore_index=True).to_csv(output/f"predictions_{name}_{arm}.csv",index=False)
    for name in LINEAR: fit_linear(values[name],table,folds,output,name)
    return analyze(output)


def coverage(rows):
    for _,part in rows.groupby(["model","arm","seed"]):
        if len(part)!=1080 or part.trial_id.duplicated().any() or part.subject_id.nunique()!=15: raise ValueError("Incorrect once-per-trial OOF coverage")


def correctness(rows,task):
    y=rows.original_label.to_numpy(dtype=int); p=probabilities(rows)
    if task=="coarse3": return p.argmax(1)==COARSE[y]
    positive=p[:,2]/np.maximum(p[:,1:].sum(1),1e-12)
    return np.stack([1-positive,positive],axis=1).argmax(1)==(y==3)


def bootstrap_scores(rows,subjects,materials,sw,mw,task):
    rows=rows.copy(); rows["correct"]=correctness(rows,task)
    matrix=rows.groupby(["subject_id","material_key"]).correct.mean().unstack().loc[subjects,materials.material].to_numpy(dtype=float)
    if not np.isfinite(matrix).all(): raise ValueError("Incomplete crossed observations")
    native=materials.original_label.to_numpy(dtype=int); target=COARSE[native] if task=="coarse3" else (native==3).astype(int)
    include=np.ones(72,dtype=bool) if task=="coarse3" else native!=0
    result=[]; point=[]
    for c in range(3 if task=="coarse3" else 2):
        selected=(target==c)&include
        values=sw@matrix[:,selected]; weights=mw[:,selected]
        result.append((values*weights).sum(1)/(sw.sum(1)*weights.sum(1))); point.append(matrix[:,selected].mean())
    return float(np.mean(point)),np.mean(result,axis=0)


def analyze(output):
    frames={name:{arm:pd.read_csv(output/f"predictions_{name}_{arm}.csv") for arm in ARMS} for name in (*NEURAL,*LINEAR)}
    table=next(iter(frames.values()))["exposed"][["trial_id","subject_id","session","original_label"]].drop_duplicates()
    subjects,materials,sw,mw=crossed_weights(table); constant_material=np.ones_like(mw)
    report={}; draws={}; participant={}
    for name,arms in frames.items():
        report[name]={}; draws[name]={}; participant[name]={}
        for arm,rows in arms.items():
            coverage(rows)
            per_seed={str(seed):common_metrics(part.original_label.to_numpy(),probabilities(part),"coarse3") for seed,part in rows.groupby("seed")}
            item={"seeds":per_seed}; draws[name][arm]={}; participant[name][arm]={}
            for task in ("coarse3","binary"):
                point,b=bootstrap_scores(rows,subjects,materials,sw,mw,task)
                _,bp=bootstrap_scores(rows,subjects,materials,sw,constant_material,task)
                if abs(point-np.mean([v[task]["balanced_accuracy"] for v in per_seed.values()]))>1e-12: raise ValueError("Bootstrap/metric point disagreement")
                item[task]={"mean_BA":point,"crossed_percentile_95":np.quantile(b,[.025,.975]).tolist(),"participant_percentile_95":np.quantile(bp,[.025,.975]).tolist()}
                draws[name][arm][task]=b; participant[name][arm][task]=bp
            item["cells"]={f"session{s}_rotation{r}":{str(seed):common_metrics(part.original_label.to_numpy(),probabilities(part),"coarse3")
                for seed,part in cell.groupby("seed")} for (s,r),cell in rows.groupby(["source_session","material_rotation"])}
            report[name][arm]=item
    pairs=[]
    def contrast(a,aa,b,ba,task):
        delta=draws[a][aa][task]-draws[b][ba][task]; part=participant[a][aa][task]-participant[b][ba][task]
        pairs.append({"model_a":a,"arm_a":aa,"model_b":b,"arm_b":ba,"task":task,
                      "BA_difference":report[a][aa][task]["mean_BA"]-report[b][ba][task]["mean_BA"],
                      "crossed_percentile_95":np.quantile(delta,[.025,.975]).tolist(),"participant_percentile_95":np.quantile(part,[.025,.975]).tolist()})
    for task in ("coarse3","binary"):
        for name in frames: contrast(name,"unexposed",name,"exposed",task)
        for arm in ARMS:
            contrast("transformer",arm,"mean_mlp",arm,task); contrast("reve_pretrained",arm,"reve_random42",arm,task)
    result={"development_only":True,"research_question_change_approved":False,"models":report,"contrasts":pairs,
            "bootstrap":{"draws":10000,"seed":20261006,"subject_weight_digest":digest(sw.tolist()),"material_weight_digest":digest(mw.tolist()),
                         "scope":"Unadjusted paired participant-only and crossed participant/material percentile intervals, averaged seeds; conditional fixed training folds/models,3sessions and observed material keys"},
            "scope":"Matched within-session training size,participants,native labels,val/test trials. Exposure changes train materials/content/order,not pure causal identity. Previously inspected cohort; no pooled or unseen-corpus claim"}
    write_json(output/"comparison.json",result); return result


def compare_metrics(actual,expected):
    for task,metrics in expected.items():
        for key,value in metrics.items():
            if key=="balanced_log_loss":
                if abs(actual[task][key]-value)>1e-10: raise ValueError("Logloss does not replay")
            elif actual[task][key]!=value: raise ValueError("Classification metrics do not replay")


def verify(cache,reve_run,output,device="cuda"):
    x,table,info,values=features(cache,reve_run); config=json.loads((output/"config.json").read_text())
    if configuration(info,reve_run,output/"plan.json",config["device"])!=config: raise ValueError("Changed input/config/source bindings")
    folds=make_folds(table.subject_id.tolist())
    if json.loads((output/"folds.json").read_text())!=folds: raise ValueError("Changed folds")
    expected_assignments=table[["material_key","session","original_label","material_rank"]].drop_duplicates().sort_values("material_key").reset_index(drop=True)
    if not pd.read_csv(output/"material_assignments.csv").equals(expected_assignments): raise ValueError("Changed material assignment")
    index=json.loads((output/"model_index.json").read_text())
    if len(index)!=540 or len({r["id"] for r in index})!=540: raise ValueError("Incomplete neural experiment")
    signatures={}; initializations={}; rows={name:{arm:[] for arm in ARMS} for name in NEURAL}; maximum=0.
    labels=table.original_label.to_numpy(dtype=int)
    for i,item in enumerate(index,1):
        folder=output/"models"/item["id"]
        if sha256(folder/"metrics.json")!=item["metrics_sha256"]: raise ValueError("Changed fit record")
        r=json.loads((folder/"metrics.json").read_text())
        for filename,key in [("selected.pt","checkpoint_sha256"),("history.json","history_sha256"),("predictions.csv","predictions_sha256")]:
            if sha256(folder/filename)!=r[key]: raise ValueError("Changed fitted artifact")
        idx=masks(table,folds[r["fold"]],r["session"],r["rotation"],r["arm"]); init=r["seed"]+r["fold"]*1000+r["rotation"]*10000
        batches,signature=paired_batches(table,idx["train"],init)
        if signature!=r["canonical_draw_signature"] or digest(batches.tolist())!=r["batch_digest"]: raise ValueError("Changed paired exposure")
        signatures.setdefault((r["seed"],r["fold"],r["rotation"]),set()).add(signature)
        seed_everything(init); model=TemporalControl(r["model"],"coarse3").to(device)
        if state_digest(model.state_dict())!=r["initial_state_sha256"]: raise ValueError("Initial state not reproduced")
        initializations.setdefault((r["model"],r["seed"],r["fold"],r["rotation"]),set()).add(r["initial_state_sha256"])
        checkpoint=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False); model.load_state_dict(checkpoint["state"])
        scaler=StandardScaler().fit(x[idx["train"]].reshape(-1,56)); np.testing.assert_array_equal(scaler.mean_,checkpoint["mean"]); np.testing.assert_array_equal(scaler.scale_,checkpoint["scale"])
        data=torch.from_numpy(scaler.transform(x.reshape(-1,56)).reshape(x.shape).astype(np.float32)).to(device)
        history=json.loads((folder/"history.json").read_text())
        if [h["step"] for h in history]!=list(range(25,601,25)): raise ValueError("Incomplete available training budget")
        best=max(history,key=lambda h:(h["validation"]["coarse3"]["balanced_accuracy"],-h["validation"]["coarse3"]["balanced_log_loss"]))
        if best["step"]!=r["selected_step"] or checkpoint["selected_step"]!=r["selected_step"]: raise ValueError("Incorrect selection")
        saved=pd.read_csv(folder/"predictions.csv")
        with sdpa_kernel(SDPBackend.MATH):
            compare_metrics(common_metrics(labels[idx["validation"]],infer(model,data[idx["validation"]]),"coarse3"),best["validation"])
            compare_metrics(common_metrics(labels[idx["train"]],infer(model,data[idx["train"]]),"coarse3"),r["training"])
            p=infer(model,data[idx["test"]]); compare_metrics(common_metrics(labels[idx["test"]],p,"coarse3"),r["test"])
        expected=frame(table,idx["test"],p,r["model"],r["arm"],r["session"],r["rotation"],r["seed"],r["fold"])
        if not saved.drop(columns=["p_0","p_1","p_2"]).equals(expected.drop(columns=["p_0","p_1","p_2"]).reset_index(drop=True)): raise ValueError("Changed trial metadata")
        error=float(np.max(np.abs(p-probabilities(saved)))); maximum=max(maximum,error)
        if error>1e-6: raise ValueError("Neural probability replay failed")
        rows[r["model"]][r["arm"]].append(saved); del model,data
        if i%90==0: print(f"Replayed neural{i}/540",flush=True)
    if any(len(v)!=1 for v in (*signatures.values(),*initializations.values())): raise ValueError("Unmatched paired samples or initial states")
    for name in NEURAL:
        for arm in ARMS:
            saved=pd.read_csv(output/f"predictions_{name}_{arm}.csv"); merged=pd.concat(rows[name][arm],ignore_index=True)
            # Index order and aggregate order are intentionally identical across sessions/rotations/seeds.
            if not saved.drop(columns=["p_0","p_1","p_2"]).equals(merged.drop(columns=["p_0","p_1","p_2"])):
                raise ValueError("Changed neural aggregate metadata")
            np.testing.assert_allclose(probabilities(saved),probabilities(merged),atol=1e-15,rtol=0)
            coverage(saved)
        for session in (1,2,3):
            chunk=json.loads((output/f"model_metrics_{name}_session{session}.json").read_text())
            expected=[json.loads((output/"models"/item["id"]/"metrics.json").read_text()) for item in index if item["id"].startswith(name+"_") and f"_session{session}_" in item["id"]]
            if chunk!=expected: raise ValueError("Changed bounded metric export")
    head_results={}
    for name in LINEAR:
        predicted={arm:pd.read_csv(output/f"predictions_{name}_{arm}.csv") for arm in ARMS}; max_head=0.; refitted=0; max_coef=0.
        for p in predicted.values(): coverage(p)
        for session in (1,2,3):
            for rotation in (0,1,2):
                records=json.loads((output/f"linear_{name}_session{session}_rotation{rotation}.json").read_text())
                if len(records)!=10: raise ValueError("Missing paired linear folds")
                for r in records:
                    idx=masks(table,folds[r["fold"]],session,rotation,r["arm"]); scaler=StandardScaler().fit(values[name][idx["train"]]); inputs=scaler.transform(values[name])
                    np.testing.assert_array_equal(scaler.mean_,r["scaler_mean"]); np.testing.assert_array_equal(scaler.scale_,r["scaler_scale"])
                    if [c["C"] for c in r["candidates"]]!=list(C_VALUES): raise ValueError("Changed C budget")
                    fits=[]
                    for candidate in r["candidates"]:
                        est=LogisticRegression(C=candidate["C"],class_weight="balanced",max_iter=2000,tol=1e-6,random_state=42)
                        est.fit(inputs[idx["train"]],COARSE[labels[idx["train"]]]); fits.append(est); refitted+=1
                        compare_metrics(common_metrics(labels[idx["validation"]],est.predict_proba(inputs[idx["validation"]]),"coarse3"),candidate["validation"])
                    chosen=max(range(4),key=lambda j:(r["candidates"][j]["validation"]["coarse3"]["balanced_accuracy"],-r["candidates"][j]["validation"]["coarse3"]["balanced_log_loss"]))
                    if r["candidates"][chosen]["C"]!=r["selected_C"]: raise ValueError("Incorrect C selection")
                    w=np.asarray(r["coefficients"]); b=np.asarray(r["intercept"]); est=fits[chosen]
                    np.testing.assert_allclose(w,est.coef_,atol=1e-9,rtol=1e-9); np.testing.assert_allclose(b,est.intercept_,atol=1e-9,rtol=1e-9)
                    max_coef=max(max_coef,float(np.max(np.abs(w-est.coef_))),float(np.max(np.abs(b-est.intercept_))))
                    p=softmax(inputs[idx["test"]]@w.T+b,axis=1); compare_metrics(common_metrics(labels[idx["test"]],p,"coarse3"),r["test"])
                    rows=predicted[r["arm"]]; rows=rows[(rows.source_session==session)&(rows.material_rotation==rotation)&(rows.fold==r["fold"])]
                    expected=frame(table,idx["test"],p,name,r["arm"],session,rotation,42,r["fold"])
                    if not rows.drop(columns=["p_0","p_1","p_2"]).reset_index(drop=True).equals(expected.drop(columns=["p_0","p_1","p_2"]).reset_index(drop=True)): raise ValueError("Changed linear trial metadata")
                    error=float(np.max(np.abs(p-probabilities(rows)))); max_head=max(max_head,error)
                    if error>1e-14: raise ValueError("Linear probability replay failed")
        head_results[name]={"selected_heads":90,"candidates_independently_refitted":refitted,"probability_rows":2160,"max_probability_error":max_head,"max_coefficient_refit_error":max_coef}
        print(f"Replayed all360candidates/90selected {name} heads",flush=True)
    expected=json.loads((output/"comparison.json").read_text())
    if analyze(output)!=expected: raise ValueError("Changed aggregate/bootstrap analysis")
    result={"passed":True,"neural_checkpoints_replayed":540,"neural_probability_rows":12960,"max_neural_probability_error":maximum,
            "paired_draw_groups":len(signatures),"initialization_groups":len(initializations),"linear":head_results,
            "scope":"Bound source/input/plan/material/fold metadata; exact shared val/test access, train-only scalers and balanced exposures,initial states,selected train/val/test checkpoint replays,all1440linear candidate refits,360coefficients,all21600probabilities and OOF/pairedbootstrap recomputation"}
    write_json(output/"verification.json",result); return result
