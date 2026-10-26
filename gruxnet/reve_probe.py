"""Pinned, source-reviewed, frozen REVE extraction and participant/session probes.

Downloaded author code and weights remain outside the public repository. The
adapter uses no HF remote-code loader, network calls or target-population fitting.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly
import torch
from safetensors.torch import load_file
from .data import COMMON_CHANNELS,digest,roots,seed_channels,sha256,write_json
from .prepare import preprocess_signal
from .temporal_controls import load,sequence_features,make_folds
from .session_controls import linear,verify_linear,analysis_table
from .train import seed_everything

REPO=Path(__file__).resolve().parents[1]
PINNED={"brain-bzh/reve-base":"dc2a075c287bb2f6c04ee5875bd79535a0f7dba6",
        "brain-bzh/reve-positions":"befa5b57a455b77cf302daf610c2e9ed8140bace"}
WEIGHTS={"reve-base":"8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe",
         "reve-positions":"4b793820b9df0998667deb6c8ce2dbb86b38221d5165c1844bc4971941b13f13"}


def audit_assets(assets):
    manifest=json.loads((assets/"download_manifest.json").read_text())
    if manifest["model_revisions"]!=PINNED or not manifest["weights_downloaded"]:
        raise ValueError("Expected pinned, complete official downloads")
    for row in manifest["files"]:
        if sha256(assets/row["local_file"])!=row["sha256"]:
            raise ValueError("Changed audited author asset")
    for repo,expected in WEIGHTS.items():
        if sha256(assets/repo/"model.safetensors")!=expected:
            raise ValueError("Unexpected checkpoint")
    card=assets/"reve-dataset/README.md"
    if sha256(card)!=manifest["pretraining_card"]["sha256"]:
        raise ValueError("Changed pretraining card")
    source_files=["reve-base/configuration_reve.py","reve-base/modeling_reve.py",
                  "reve-positions/configuration_bank.py","reve-positions/position_bank.py"]
    result={"model_revisions":PINNED,"weight_sha256":WEIGHTS,
            "source_hashes":{p:sha256(assets/p) for p in source_files},
            "download_manifest_sha256":sha256(assets/"download_manifest.json"),
            "review":"All four downloaded Python files manually reviewed before local import: configuration, tensor layers and electrode lookup; no shell, network or arbitrary external-file operations in these files. Tensor attention backend selection retained",
            "loading":"Explicit local Python package import after digest checks; safetensors load_file and strict state_dict loading. No HF remote-code execution or online model loading",
            "pretraining_card":manifest["pretraining_card"],
            "public_card_target_name_matches":{term:term.lower() in card.read_text().lower() for term in ("SEED","DEAP","GAMEEMO")},
            "target_independence":"No target corpus named in reviewed paper Appendix B or open-subset card. The card explicitly covers only the open subset; no full file/participant manifest or independently certified checkpoint exclusion. Absence of names is not proof of exclusion",
            "license":"Official pinned REVE Responsible Use License v1.0 permits this research. Weights and downloaded author code remain local; no model redistribution",
            "development_only":True}
    write_json(assets/"code_review.json",result)
    return result


def local_model(assets,pretrained):
    audit_assets(assets)
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    namespace="_gruxnet_reviewed_reve"
    package=types.ModuleType(namespace); package.__path__=[str(assets/"reve-base")]
    sys.modules[namespace]=package
    modules={}
    for name in ("configuration_reve","modeling_reve"):
        spec=importlib.util.spec_from_file_location(namespace+"."+name,assets/"reve-base"/(name+".py"))
        module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
        spec.loader.exec_module(module); modules[name]=module
    seed_everything(42)
    config=modules["configuration_reve"].ReveConfig(**json.loads((assets/"reve-base/config.json").read_text()))
    model=modules["modeling_reve"].Reve(config)
    if pretrained:
        model.load_state_dict(load_file(str(assets/"reve-base/model.safetensors")),strict=True)
    model.eval().requires_grad_(False)
    bank=load_file(str(assets/"reve-positions/model.safetensors"))
    names=json.loads((assets/"reve-positions/config.json").read_text())["position_names"]
    if bank["embedding"].shape!=(len(names),3) or len(set(names))!=len(names):
        raise ValueError("Invalid electrode bank")
    positions=bank["embedding"][[names.index(c) for c in COMMON_CHANNELS]].clone()
    if positions.shape!=(14,3): raise ValueError("Missing physical electrodes")
    return model,positions


def prepare(data_root,temporal_cache,cache):
    if cache.exists(): raise FileExistsError("Input cache exists")
    sequences,table,info=load(temporal_cache)
    root=roots(data_root)["SEEDIV"]; channels=seed_channels(root)
    for name,expected in info["metadata_sources"].items():
        if sha256(root/name)!=expected: raise ValueError("Changed source metadata")
    cache.mkdir(parents=True)
    signals=np.lib.format.open_memmap(cache/"prefix128.npy",mode="w+",dtype=np.float32,shape=(1080,14,5120))
    for source in info["source_files"]:
        path=root/source["file"]
        if sha256(path)!=source["sha256"]: raise ValueError("Changed raw source")
        values=loadmat(path)
        for i,row in table[table.source_file==source["file"]].iterrows():
            filtered,mask=preprocess_signal(values[row.source_key],200,channels,COMMON_CHANNELS)
            prefix=filtered[:,:5120]
            if not mask.all() or prefix.shape!=(14,5120): raise ValueError("Invalid prefix")
            windows=prefix.reshape(14,10,512).transpose(1,0,2)
            np.testing.assert_array_equal(sequence_features(windows),sequences[i])
            signals[i]=prefix
        print(f"Verified REVE raw prefixes: {source['file']}",flush=True)
    signals.flush(); del signals
    table.to_csv(cache/"trials.csv",index=False)
    record={"shape":[1080,14,5120],"dtype":"float32","temporal_cache_fingerprint":info["fingerprint"],
            "exact_sequence_feature_matches":1080,"channels":COMMON_CHANNELS,
            "prefix128_sha256":sha256(cache/"prefix128.npy"),"trials_sha256":sha256(cache/"trials.csv"),
            "filter_context":info["filter_context"]}
    record["fingerprint"]=digest(record); write_json(cache/"prepared.json",record)
    return record


def load_inputs(cache):
    info=json.loads((cache/"prepared.json").read_text()); check=dict(info); check.pop("fingerprint")
    if digest(check)!=info["fingerprint"]: raise ValueError("Changed input metadata")
    for filename,key in [("prefix128.npy","prefix128_sha256"),("trials.csv","trials_sha256")]:
        if sha256(cache/filename)!=info[key]: raise ValueError("Changed input cohort")
    return np.load(cache/"prefix128.npy",mmap_mode="r",allow_pickle=False),pd.read_csv(cache/"trials.csv"),info


def adapt(trials):
    values=resample_poly(np.asarray(trials,dtype=np.float32),25,16,axis=-1)
    if values.shape[1:]!=(14,8000) or not np.isfinite(values).all(): raise ValueError("Invalid fixed 200Hz observation")
    # Entire observation's own channel statistics; never pool people/trials.
    mean=values.mean(-1,keepdims=True,dtype=np.float64)
    scale=values.std(-1,keepdims=True,dtype=np.float64)
    values=np.clip((values-mean)/np.maximum(scale,1e-8),-15,15).astype(np.float32)
    return values.reshape(-1,14,10,800).transpose(0,2,1,3).reshape(-1,14,800)


def embed(model,positions,windows,device,batch_size):
    parts=[]
    with torch.inference_mode():
        for i in range(0,len(windows),batch_size):
            batch=torch.from_numpy(windows[i:i+batch_size].copy()).to(device)
            tokens=model(batch,positions.to(device).expand(len(batch),-1,-1))
            if tokens.shape[1:]!=(14,4,512): raise ValueError("Unexpected author encoder token shape")
            parts.append(tokens.mean((1,2)).cpu().numpy())
    return np.concatenate(parts)


def pilot(assets,cache):
    inputs,_,info=load_inputs(cache); windows=adapt(inputs[:1])
    model,positions=local_model(assets,True); model=model.to("cuda")
    results=[]; reference=None
    for batch in (1,2,4,10):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
        result=embed(model,positions,windows,"cuda",batch)
        torch.cuda.synchronize()
        if reference is None: reference=result
        error=float(np.max(np.abs(result-reference)))
        np.testing.assert_allclose(result,reference,atol=2e-5,rtol=2e-5)
        results.append({"batch_windows":batch,"elapsed_seconds":time.perf_counter()-start,
                        "peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated(),
                        "peak_reserved_cuda_bytes":torch.cuda.max_memory_reserved(),"max_batch_probability_free_embedding_difference":error})
    record={"label_access":"None; first fixed trial only, input feasibility and batching, no classifier fitted or score inspected",
            "input_fingerprint":info["fingerprint"],"parameters":sum(p.numel() for p in model.parameters()),
            "dtype":"float32; author SDPA backend selection, no autocast","device_name":torch.cuda.get_device_name(),
            "tests":results,"chosen_batch_windows":10,"reason":"Ten 4-second windows of one trial; fixed before outcome evaluation; memory fits 6GB"}
    write_json(assets/"feasibility.json",record)
    return record


def plan():
    return {"development_only":True,"research_question_change_approved":False,"checkpoint_revisions":PINNED,
            "encoder_conditions":["reve_pretrained","reve_random42"],"encoder_access":"Frozen eval encoder, no task gradient, no target-population calibration",
            "random_control":"Same official architecture, constructor seed42, frozen; one random encoder initialization, shared official positions",
            "input":"All1080 SEEDIV common14 trials, first40 seconds; exact prior bandpower prefix match. Offline full-original-trial filter4..40Hz then128Hz precedes prefix",
            "adapter":"Polyphase128->200Hz across each fixed prefix; per-observation per-channel population mean/std (float64), floor1e-8, clip+-15. Split into10 disjoint4s800-sample windows. Each window14electrodes*4author1s patches at180sample stride=56tokens; final60samples(0.30s) of each window unused by patcher,37s direct patch coverage within40s normalized observation",
            "pooling":"Mean final-layer channel/patch tokens to512/window, then arithmetic mean10windows to512/trial; no task attention-pooling parameters",
            "feasibility":"Batch10 windows, float32 without AMP, fixed after outcome-free first-input memory/batch-invariance pilot",
            "heads":"Training-only feature scaler and coarse3 class-balanced multinomial logistic; C.01/.1/1/10, validation3BA then balanced logloss, first tie",
            "protocols":"Same five9/3/3 participant folds: mixed all3sessions and each single source session. Single-source selection sees source only and tests each of3sessions; exact same target trials as session controls",
            "budget":"40 selected heads,160 candidates,8640 test probability rows across pretrained/random encoders; fixed encoder seed42",
            "metrics":"Primary3class BA/all1080, secondary conditionalbinary810; same-session diagonal, two cyclic unseen-source directions, average directions; mixed-session result separate",
            "uncertainty":"10000 paired participant-block draws seed20261006. Pretrained-minus-random contrasts in mixed/same/unseen protocols, unadjusted, fixedfolds/developmentcohort; one random initialization limits attribution",
            "limitations":"Published sources do not independently certify full checkpoint corpus exclusion. Adapter narrows pretraining bandwidth, uses trial instead of session zscore and mean-pools short windows. This is an authored frozen probe, not published FACED result replication or finetuning. No unseen-corpus claim"}


def extract(assets,cache,output,plan_path):
    if output.exists(): raise FileExistsError("Feature pack exists")
    declaration=json.loads(plan_path.read_text())
    if declaration!=plan(): raise ValueError("Changed probe declaration")
    inputs,table,info=load_inputs(cache); output.mkdir(parents=True)
    table.to_csv(output/"trials.csv",index=False); write_json(output/"plan.json",declaration)
    sources=["gruxnet/reve_probe.py","gruxnet/session_controls.py","gruxnet/temporal_controls.py","gruxnet/prepare.py","gruxnet/train.py"]
    records={}
    for name in ("reve_pretrained","reve_random42"):
        model,positions=local_model(assets,name=="reve_pretrained"); model=model.to("cuda")
        # State hash remains local only; confirms frozen parameters before/after inference.
        from hashlib import sha256 as hasher
        def state_digest():
            h=hasher()
            for key,value in model.state_dict().items(): h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before=state_digest(); features=np.empty((1080,512),dtype=np.float32)
        torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        for i in range(1080):
            features[i]=embed(model,positions,adapt(inputs[i:i+1]),"cuda",10).mean(0)
            if (i+1)%100==0: print(f"{name}: {i+1}/1080 trial embeddings",flush=True)
        after=state_digest()
        if before!=after: raise ValueError("Encoder changed during extraction")
        np.save(output/(name+".npy"),features,allow_pickle=False)
        records[name]={"sha256":sha256(output/(name+".npy")),"shape":list(features.shape),
                       "state_sha256":before,"state_unchanged":before==after,"elapsed_seconds":time.perf_counter()-start,
                       "peak_allocated_cuda_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_cuda_bytes":torch.cuda.max_memory_reserved()}
        del model; torch.cuda.empty_cache()
    config={"input_fingerprint":info["fingerprint"],"plan_sha256":sha256(plan_path),"trials_sha256":sha256(output/"trials.csv"),
            "download_manifest_sha256":sha256(assets/"download_manifest.json"),"code_review_sha256":sha256(assets/"code_review.json"),
            "source_hashes":{p:sha256(REPO/p) for p in sources},"features":records,"device":"cuda","torch_version":torch.__version__}
    write_json(output/"features.json",config)
    return config


def load_features(output):
    config=json.loads((output/"features.json").read_text()); table=pd.read_csv(output/"trials.csv")
    if sha256(output/"trials.csv")!=config["trials_sha256"] or sha256(output/"plan.json")!=config["plan_sha256"]:
        raise ValueError("Changed feature labels/plan")
    for path,expected in config["source_hashes"].items():
        if sha256(REPO/path)!=expected: raise ValueError("Changed bound probe implementation")
    features={}
    for name,record in config["features"].items():
        if sha256(output/(name+".npy"))!=record["sha256"]: raise ValueError("Changed features")
        features[name]=np.load(output/(name+".npy"),allow_pickle=False)
    return features,table,config


def run(output):
    features,table,_=load_features(output); folds=make_folds(table.subject_id.tolist())
    write_json(output/"folds.json",folds)
    for name,values in features.items(): linear(values,table,folds,output,name,include_mixed=True)
    return analyze(output)


def analyze(output):
    from .session_controls import protocol_views,probabilities,COARSE
    frames={name:pd.read_csv(output/f"predictions_{name}.csv") for name in ("reve_pretrained","reve_random42")}
    report=analysis_table(frames)
    subjects=sorted(frames["reve_pretrained"].subject_id.unique())
    draws=np.random.default_rng(20261006).integers(0,15,size=(10000,15))
    contrasts=[]
    for task in ("coarse3","binary"):
        key="mean_coarse3_BA" if task=="coarse3" else "mean_binary_BA"
        for protocol in ("mixed_sessions","same_session","mean_unseen"):
            scores={}
            for name,frame in frames.items():
                views=protocol_views(frame); blocks=[]
                for part in (["unseen_cycle1","unseen_cycle2"] if protocol=="mean_unseen" else [protocol]):
                    counts=[]
                    for subject in subjects:
                        rows=views[part][views[part].subject_id==subject]; y=rows.original_label.to_numpy(); p=probabilities(rows)
                        if task=="binary": keep=y!=0; y=(y[keep]==3).astype(int); p=p[keep,1:]
                        else: y=COARSE[y]
                        counts.append([[int((p.argmax(1)[y==c]==c).sum()),int((y==c).sum())] for c in range(p.shape[1])])
                    total=np.asarray(counts)[draws].sum(1); blocks.append((total[...,0]/total[...,1]).mean(1))
                scores[name]=np.mean(blocks,axis=0)
            delta=scores["reve_pretrained"]-scores["reve_random42"]
            contrasts.append({"task":task,"protocol":protocol,"comparison":"reve_pretrained minus reve_random42",
                              "mean_BA_difference":report["models"]["reve_pretrained"][protocol][key]-report["models"]["reve_random42"][protocol][key],
                              "paired_participant_percentile_95":np.quantile(delta,[.025,.975]).tolist()})
    report["pretraining_contrasts"]=contrasts
    write_json(output/"comparison.json",report)
    return report


def verify(assets,cache,output):
    features,table,config=load_features(output); inputs,input_table,info=load_inputs(cache)
    if not table.equals(input_table) or info["fingerprint"]!=config["input_fingerprint"]:
        raise ValueError("Feature/input cohort mismatch")
    audit_assets(assets)
    for filename,key in [("download_manifest.json","download_manifest_sha256"),("code_review.json","code_review_sha256")]:
        if sha256(assets/filename)!=config[key]: raise ValueError("Changed official assets/review")
    indices=[0,71,144,287,432,575,720,863,1008,1079]
    errors={}
    for name in features:
        model,positions=local_model(assets,name=="reve_pretrained"); model=model.to("cuda")
        actual=np.stack([embed(model,positions,adapt(inputs[i:i+1]),"cuda",10).mean(0) for i in indices])
        expected=features[name][indices]; error=float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5); errors[name]=error
        del model; torch.cuda.empty_cache()
    folds=make_folds(table.subject_id.tolist())
    if json.loads((output/"folds.json").read_text())!=folds: raise ValueError("Changed participant folds")
    heads={name:verify_linear(values,table,folds,output,name,include_mixed=True) for name,values in features.items()}
    previous=json.loads((output/"comparison.json").read_text())
    if analyze(output)!=previous: raise ValueError("Changed probe analysis")
    result={"passed":True,"selected_heads_replayed":40,"test_probability_rows":8640,
            "head_replays":heads,"embedding_replay_trial_indices":indices,"embedding_max_errors":errors,
            "scope":"All input/feature/source/official asset hashes,20 sampled embeddings across2encoders,40 selected head coefficients and source-only scalers/selection,full OOF coverage and paired bootstrap. Full1080embedding recomputation not repeated"}
    write_json(output/"verification.json",result)
    return result
