"""Bounded FACED development runner; no original-data I/O in this module.

All data mappings are lazy and accessed only after identity/quality/activation
guards. Preflight uses tagged synthetic fixtures; dataset activation is separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .eegnet_regression import EEGNetRegression, original_scale_prediction
from .faced_classical_contract import fit_source_ridge, predict_ridge, participant_film_mae
from .faced_classical_runner import estimate_priors
from .faced_inference_quality import POLICY_ID

READOUTS = ("eeg_only","context_residual")
RATES = (.0003,.001,.003)
SEEDS = (20261101,20261102)
CHECKPOINTS = (30,60,120)
STRONG_CONTROLS = ("global_median","coarse_valence_median","category_median","context_only","context_kernel")


def window_array(value):
    x = np.asarray(value)
    if x.shape != (7,30,512) or x.dtype.kind != "f" or not np.isfinite(x).all():
        raise ValueError("Invalid finite physical-unit trial windows")
    return x


def valid_targets(pairs, targets):
    y = np.array([targets[p] for p in pairs],dtype=np.float64)
    if y.shape != (len(pairs),) or not np.isfinite(y).all() or np.any((y<0)|(y>7)):
        raise ValueError("Invalid original-scale source targets")
    return y


def guard_bundle(programme, metadata, decisions):
    requested = [(r["subject"],r["vid"]) for r in programme["input_access"]["future_development_pairs"]]
    expected = set(requested)
    if set(metadata) != expected or set(decisions) != expected or len(expected)!=1260:
        raise PermissionError("Bundle identities differ before EEG/target access")
    if len(requested)!=1260 or [r["rotation"] for r in programme["rotation_manifests"]]!=[0,1,2]:
        raise PermissionError("Duplicate identities or changed rotation order")
    source_union=set()
    for rotation in programme["rotation_manifests"]:
        arms = rotation["arms"]
        source = {(r["subject"],r["vid"]) for r in arms["source_fit"]}
        unseen = {(r["subject"],r["vid"]) for r in arms["validation_unseen"]}
        familiar = {(r["subject"],r["vid"]) for r in arms["validation_familiar"]}
        if len(source)!=560 or len(unseen)!=120 or len(familiar)!=160 or not (source|unseen|familiar)<=expected:
            raise PermissionError("Changed development rectangles")
        for name,pairs,npeople,nclips in (("source_fit",source,70,8),("validation_unseen",unseen,20,6),("validation_familiar",familiar,20,8)):
            people={p[0] for p in pairs};clips={p[1] for p in pairs}
            if len(arms[name])!=len(pairs) or len(people)!=npeople or len(clips)!=nclips or pairs!={(p,v) for p in people for v in clips}:
                raise PermissionError("Nonrectangular or duplicate trial identities")
        if {p[0] for p in unseen}!={p[0] for p in familiar} or {p[1] for p in source}!={p[1] for p in familiar}:
            raise PermissionError("Familiar arm identity mismatch")
        if {p[0] for p in source}&{p[0] for p in unseen|familiar}:
            raise PermissionError("Participant source/evaluation overlap")
        if {metadata[p]["family"] for p in source}&{metadata[p]["family"] for p in unseen}:
            raise PermissionError("Source-title family overlap")
        source_union |= source
    if len(source_union)!=980:
        raise PermissionError("Changed source union")
    for pair in expected:
        decision=decisions[pair]
        role="source" if pair in source_union else "validation"
        if metadata[pair].get("role")!=role or decision.get("role")!=role or (decision.get("subject"),decision.get("vid"))!=pair:
            raise PermissionError("Source/validation technical role mismatch")
        if decision.get("policy_id")!=POLICY_ID or type(decision.get("eeg_available")) is not bool or not isinstance(decision.get("reasons"),list):
            raise PermissionError("Invalid quality decision")
        if decision["eeg_available"]==bool(decision["reasons"]):
            raise PermissionError("Contradictory quality decision")
        if pair in source_union and not decision["eeg_available"]:
            raise PermissionError("Source quality failure holds targets and fitting")
    return expected


def guard_controls(programme,controls):
    if set(controls)!={0,1,2}:
        raise PermissionError("Incomplete stronger control rotations")
    for manifest in programme["rotation_manifests"]:
        expected={(r["subject"],r["vid"]) for r in manifest["arms"]["validation_unseen"]}
        panel=controls[manifest["rotation"]]
        if set(panel)!=set(STRONG_CONTROLS):
            raise PermissionError("Missing stronger no-EEG control")
        for values in panel.values():
            if set(values)!=expected:
                raise PermissionError("Control identity mismatch")
            a=np.array([values[p] for p in sorted(expected)],dtype=float)
            if not np.isfinite(a).all() or np.any((a<0)|(a>7)):
                raise ValueError("Invalid bounded context prediction")


@dataclass
class SourceChannelScaler:
    mean: np.ndarray
    sd: np.ndarray
    source_pairs: tuple
    sd_floor_uv: float = 1e-6

    @classmethod
    def fit(cls, pairs, windows):
        pairs=tuple(pairs)
        if not pairs or len(set(pairs))!=len(pairs):
            raise ValueError("Invalid unique source normalization set")
        total=0;mean=np.zeros(30);m2=np.zeros(30)
        for pair in pairs:
            x=window_array(windows[pair]).astype(np.float64).transpose(1,0,2).reshape(30,-1)
            n=x.shape[1];mu=x.mean(axis=1);delta=mu-mean
            within=np.square(x-mu[:,None]).sum(axis=1)
            m2 += within+delta**2*total*n/(total+n)
            mean += delta*n/(total+n);total+=n
        sd=np.maximum(np.sqrt(np.maximum(m2/total,0)),1e-6)
        if not np.isfinite(mean).all() or not np.isfinite(sd).all():
            raise ValueError("Nonfinite source normalization")
        return cls(mean,sd,pairs)

    def transform(self, trial):
        values=(window_array(trial).astype(np.float64)-self.mean[None,:,None])/self.sd[None,:,None]
        result=values.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Normalization overflow")
        return result


def source_context(source_pairs, folds, features, targets, alpha):
    pairs=list(source_pairs);people={p[0] for p in pairs}
    flattened=[p for fold in folds for p in fold]
    if len(folds)!=5 or len(flattened)!=len(set(flattened)) or set(flattened)!=people:
        raise PermissionError("OOF participant roles changed")
    full=fit_source_ridge(pairs,features,targets,alpha)
    oof={};ledger=[]
    for fold,held_people in enumerate(folds):
        held=[p for p in pairs if p[0] in held_people]
        training=[p for p in pairs if p[0] not in held_people]
        if not held or not training or {p[0] for p in training}&{p[0] for p in held}:
            raise PermissionError("Invalid residual source separation")
        model=fit_source_ridge(training,features,targets,alpha)
        values=predict_ridge(model,held,features)
        oof.update(zip(held,values))
        ledger.append({"fold":fold,"source_pairs":training,"held_pairs":held,"source_identity_sha256":model["source_identity_sha256"]})
    if set(oof)!=set(pairs):
        raise PermissionError("Incomplete OOF residual coverage")
    return full,oof,ledger


@dataclass
class FitBudget:
    seconds_cap: float = 21600
    fit_cap: int = 20
    reserved_bytes_cap: int = 5500000000
    clock: object = time.monotonic
    fits: int = 0
    elapsed: float = 0
    started: float | None = None

    def check(self, device=None):
        running=0 if self.started is None else self.clock()-self.started
        if self.elapsed+running>=self.seconds_cap:
            raise TimeoutError("Fixed GPU-fit elapsed budget exhausted")
        if device is not None and torch.device(device).type=="cuda":
            if torch.cuda.max_memory_reserved(device)>self.reserved_bytes_cap:
                raise MemoryError("Declared CUDA reserved-memory ceiling exceeded")

    def begin(self):
        self.check()
        if self.started is not None or self.fits>=self.fit_cap:
            raise PermissionError("Fixed fit count or concurrent-fit guard")
        self.fits+=1;self.started=self.clock()

    def end(self):
        if self.started is not None:
            self.elapsed+=self.clock()-self.started;self.started=None


def training_mse_flags(baseline, mse60, mse120, seed_maes):
    values=[baseline,*mse60,*mse120,*seed_maes]
    flags=[]
    if len(mse60)!=2 or len(mse120)!=2 or len(seed_maes)!=2 or not np.isfinite(values).all() or min(values)<0:
        return ["invalid_or_nonfinite_learning_metrics"]
    if any(value>.8*baseline for value in mse120):flags.append("insufficient_source_learning")
    if any(last<.9*mid for mid,last in zip(mse60,mse120)):flags.append("budget_limited_convergence")
    if abs(seed_maes[0]-seed_maes[1])>.15:flags.append("validation_seed_divergence")
    if max(mse120)>2*max(min(mse120),1e-8):flags.append("source_seed_divergence")
    return flags


def trial_rows(pairs, values, targets, metadata):
    if len(pairs)!=len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid complete trial prediction")
    y=valid_targets(pairs,targets)
    return [{"subject":p[0],"vid":p[1],"family":metadata[p]["family"],"target":float(a),"prediction":float(b)} for p,a,b in zip(pairs,y,values)]


def select_schedule(records, unseen_pairs, targets, metadata):
    expected={(rate,seed) for rate in RATES for seed in SEEDS}
    keyed={(r["learning_rate"],r["seed"]):r for r in records}
    if len(records)!=6 or set(keyed)!=expected or any(set(r["checkpoints"])!=set(CHECKPOINTS) for r in records):
        raise PermissionError("Partial/changed candidate panel cannot select a winner")
    candidates=[]
    for rate in RATES:
        for epoch in CHECKPOINTS:
            values=np.stack([keyed[rate,seed]["checkpoints"][epoch]["validation_unseen_raw"] for seed in SEEDS])
            if values.shape!=(2,len(unseen_pairs)) or not np.isfinite(values).all():
                raise ValueError("Missing/nonfinite ensemble member")
            averaged=np.clip(values.mean(axis=0),0,7)
            error=participant_film_mae(trial_rows(unseen_pairs,averaged,targets,metadata))
            candidates.append({"learning_rate":rate,"epoch":epoch,"mae":error})
    minimum=min(c["mae"] for c in candidates)
    chosen=min((c for c in candidates if c["mae"]<=minimum+1e-8),key=lambda c:(c["epoch"],c["learning_rate"]))
    return chosen,candidates


def _batch(pairs, windows, scaler, device):
    values=np.stack([scaler.transform(windows[p]) for p in pairs])
    return torch.as_tensor(values,device=device)


def predict_trials(model, pairs, windows, scaler, offsets, decisions, device):
    result=np.array([offsets[p] for p in pairs],dtype=float)
    eligible=[i for i,p in enumerate(pairs) if decisions[p]["eeg_available"]]
    # Unavailable EEG is never fetched, including as an unused dummy batch row.
    model.eval()
    with torch.no_grad():
        for start in range(0,len(eligible),8):
            indices=eligible[start:start+8];batch=[pairs[i] for i in indices]
            normalized=model(_batch(batch,windows,scaler,device))
            offset=torch.as_tensor([offsets[p] for p in batch],dtype=normalized.dtype,device=device)
            result[indices]=original_scale_prediction(normalized,offset).cpu().numpy()
    if not np.isfinite(result).all():raise ValueError("Nonfinite prediction cannot be imputed")
    return result


def fit_trial_model(source, arms, windows, targets, offsets, scaler, decisions,
                    *, rate, seed, device, budget, event=lambda value:None, epochs=120, checkpoints=CHECKPOINTS,
                    source_inference_offsets=None):
    """Train on original trials only. Smaller schedules are synthetic-test fixtures."""
    if rate not in RATES or seed not in SEEDS or not source or epochs<=0 or max(checkpoints)>epochs:
        raise ValueError("Undeclared optimizer/schedule")
    y=valid_targets(source,targets)
    source_offset=np.array([offsets["source_fit"][p] for p in source],dtype=float)
    normalized_target=(y-source_offset)/7
    if not np.isfinite(normalized_target).all():raise ValueError("Nonfinite residual target")
    baseline=float(np.mean((y-source_offset)**2))
    budget.begin();history=[];saved={}
    try:
        torch.manual_seed(seed)
        if torch.device(device).type=="cuda":
            # Initialize the allocator context before its first peak-stat reset.
            torch.cuda.init();torch.cuda.set_device(torch.device(device))
            torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        model=EEGNetRegression().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=.01)
        shuffle=np.random.default_rng(seed)
        for epoch in range(1,epochs+1):
            budget.check(device);model.train();summed=0.
            for index in range(0,len(source),8):
                # Construct one fresh deterministic permutation per epoch below.
                if index==0:order=shuffle.permutation(len(source))
                selection=order[index:index+8];pairs=[source[i] for i in selection]
                x=_batch(pairs,windows,scaler,device)
                target=torch.as_tensor(normalized_target[selection],dtype=x.dtype,device=device)
                optimizer.zero_grad(set_to_none=True);prediction=model(x)
                loss=torch.mean((prediction-target)**2)
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Missing/nonfinite gradient")
                optimizer.step();model.constrain();budget.check(device)
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite updated parameter")
                summed+=float(loss.detach())*len(pairs)
            history.append({"epoch":epoch,"normalized_training_mse":summed/len(source)})
            event({"type":"epoch","epoch":epoch,"history":history[-1]})
            if epoch in checkpoints:
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                predictions={arm:predict_trials(model,pairs,windows,scaler,offsets[arm],decisions,device) for arm,pairs in arms.items()}
                raw=predictions["source_fit"]
                saved[epoch]={**{arm+"_raw":value for arm,value in predictions.items()},
                              "training_target_source_mse":float(np.mean((y-raw)**2)),"state_dict":state}
                if source_inference_offsets is not None:
                    # Same EEG output, full-source context rather than its OOF training offset.
                    inference=raw-source_offset+np.array([source_inference_offsets[p] for p in source],dtype=float)
                    if not np.isfinite(inference).all():raise ValueError("Invalid source inference offset")
                    saved[epoch]["source_inference_raw"]=inference
                    saved[epoch]["source_inference_mse"]=float(np.mean((y-inference)**2))
                event({"type":"checkpoint","epoch":epoch,**saved[epoch]})
        budget.check(device)
        return {"learning_rate":rate,"seed":seed,"history":history,"checkpoints":saved,"starting_control_mse":baseline}
    except Exception as error:
        event({"type":"failure","error":type(error).__name__+": "+str(error),"history":history})
        raise
    finally:
        if torch.device(device).type=="cuda":torch.cuda.synchronize(device)
        budget.end()


def require_activation(activation, expected_plan_sha256, code_bindings):
    if not isinstance(activation,dict) or activation.get("status")!="active_development_diagnostic" or activation.get("preflight_plan_sha256")!=expected_plan_sha256:
        raise PermissionError("A separately frozen development activation is required")
    if activation.get("code_bindings")!=code_bindings or activation.get("confirmation_access") is not False or activation.get("research_question_changed") is not False:
        raise PermissionError("Activation code/confirmation/paper contract mismatch")


def run_diagnostic(programme, metadata, decisions, windows, targets, context_features,
                   controls, *, activation, expected_plan_sha256, code_bindings, device,
                   event=lambda value:None, budget=None):
    require_activation(activation,expected_plan_sha256,code_bindings)
    guard_bundle(programme,metadata,decisions)
    guard_controls(programme,controls)
    if torch.device(device).type!="cuda":raise PermissionError("Declared dataset diagnostic requires measured CUDA budget")
    budget=budget or FitBudget();selections={};outputs=[]
    if (budget.fit_cap,budget.seconds_cap,budget.reserved_bytes_cap,budget.fits,budget.started)!=(20,21600,5500000000,0,None):
        raise PermissionError("Changed declared resource budget or partial resume")
    for manifest in programme["rotation_manifests"]:
        rotation=manifest["rotation"]
        arms={name:[(r["subject"],r["vid"]) for r in rows] for name,rows in manifest["arms"].items()}
        source=arms["source_fit"]
        # Source signal/scaling occurs before target access and any source fit.
        scaler=SourceChannelScaler.fit(source,windows)
        alpha={0:.1,1:100.,2:100.}[rotation]
        context,oof,ledger=source_context(source,programme["source_participant_oof_folds"],context_features,targets,alpha)
        priors=estimate_priors(source,targets,metadata)
        offsets={"eeg_only":{arm:{p:priors["global"] for p in pairs} for arm,pairs in arms.items()},
                 "context_residual":{"source_fit":oof}}
        for arm in ("validation_unseen","validation_familiar"):
            offsets["context_residual"][arm]=dict(zip(arms[arm],predict_ridge(context,arms[arm],context_features)))
        source_inference={"eeg_only":offsets["eeg_only"]["source_fit"],
                          "context_residual":dict(zip(source,predict_ridge(context,source,context_features)))}
        event({"type":"source_prepared","rotation":rotation,"source_pairs":source,"channel_mean_uv":scaler.mean,"channel_sd_uv":scaler.sd,"context_alpha":alpha,"residual_oof_ledger":ledger})
        for readout in READOUTS:
            rates=RATES if rotation==0 else (selections[readout]["learning_rate"],)
            records=[]
            for rate in rates:
                for seed in SEEDS:
                    event({"type":"fit_begin","rotation":rotation,"readout":readout,"learning_rate":rate,"seed":seed})
                    fit=fit_trial_model(source,arms,windows,targets,offsets[readout],scaler,decisions,rate=rate,seed=seed,device=device,budget=budget,
                        event=lambda item:event({"rotation":rotation,"readout":readout,"learning_rate":rate,"seed":seed,**item}),
                        source_inference_offsets=source_inference[readout])
                    records.append(fit)
            if rotation==0:
                chosen,candidates=select_schedule(records,arms["validation_unseen"],targets,metadata)
                selections[readout]=chosen
                event({"type":"selection","readout":readout,"chosen":chosen,"all_candidates":candidates})
            chosen=selections[readout];rate=chosen["learning_rate"];epoch=chosen["epoch"]
            members=[r for r in records if r["learning_rate"]==rate]
            if {r["seed"] for r in members}!=set(SEEDS) or len(members)!=2:
                raise PermissionError("Partial transfer ensemble")
            paired={};seed_maes=[]
            for arm,pairs in arms.items():
                raw=np.stack([r["checkpoints"][epoch][arm+"_raw"] for r in members]).mean(axis=0)
                rows=trial_rows(pairs,np.clip(raw,0,7),targets,metadata)
                paired[arm]=rows
                event({"type":"ensemble_predictions","rotation":rotation,"readout":readout,"arm":arm,"rows":rows,"fallback_pairs":[p for p in pairs if not decisions[p]["eeg_available"]]})
            source_raw=np.stack([r["checkpoints"][epoch]["source_inference_raw"] for r in members]).mean(axis=0)
            event({"type":"source_inference_predictions","rotation":rotation,"readout":readout,
                   "rows":trial_rows(source,np.clip(source_raw,0,7),targets,metadata)})
            for r in members:
                seed_maes.append(participant_film_mae(trial_rows(arms["validation_unseen"],np.clip(r["checkpoints"][epoch]["validation_unseen_raw"],0,7),targets,metadata)))
            warning=training_mse_flags(members[0]["starting_control_mse"],
                [r["checkpoints"][60]["training_target_source_mse"] for r in members],
                [r["checkpoints"][120]["training_target_source_mse"] for r in members],seed_maes)
            candidate_controls=controls[rotation]
            control_maes={name:participant_film_mae(trial_rows(arms["validation_unseen"],[values[p] for p in arms["validation_unseen"]],targets,metadata)) for name,values in candidate_controls.items()}
            strongest=min(control_maes,key=lambda name:(control_maes[name],STRONG_CONTROLS.index(name)))
            joint_mae=participant_film_mae(paired["validation_unseen"])
            gain=control_maes[strongest]-joint_mae
            outputs.append({"rotation":rotation,"readout":readout,"schedule":chosen,"learning_flags":warning,"strongest_context":strongest,"control_maes":control_maes,"gain":gain,"joint_mae":joint_mae})
    if budget.fits!=20:raise PermissionError("Incomplete declared diagnostic")
    leads={readout:all(not r["learning_flags"] and r["gain"]>0 for r in outputs if r["readout"]==readout) and np.mean([r["gain"] for r in outputs if r["readout"]==readout])>=.15 for readout in READOUTS}
    return {"rotations":outputs,"selection":selections,"fits":budget.fits,"charged_fit_seconds":budget.elapsed,"engineering_leads":leads,"confirmation_access":False,"research_question_changed":False}
