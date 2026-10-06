"""Exploratory crossed participant/material sensitivity, conditional on fitted models."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.data import digest,sha256,write_json
from gruxnet.session_controls import protocol_views,probabilities,COARSE


def plan():
    return {"development_only":True,"research_question_change_approved":False,
            "timing":"Added after initial session results were inspected; before REVE classifier outcomes. Exploratory sensitivity, not original preregistration",
            "units":"15 participants crossed with72 published-design material keys(session,trialposition); original video hashes unavailable",
            "resampling":"10000 paired draws seed20261006. Resample15 participants, then independently6 material keys with replacement in each of12 session/native-emotion strata. Same draw weights across models/protocols",
            "averaging":"Average correctness over encoder training seeds and two unseen-source directions for each participant/material before resampling; keep labels/population and source checkpoints fixed",
            "contrasts":"Same minus unseen reported as unseen-minus-same per model; Transformer-minus-MLP same/unseen; pretrained-minus-random mixed/same/unseen; both common3 and conditionalbinary",
            "limitations":"Unadjusted percentile intervals conditional on these3sessions, fixed training folds/checkpoints and observed clip proxies; not causal identification, refit uncertainty, new session population or confirmed film identity"}


def crossed_weights(table,n=10000,seed=20261006):
    table=table.copy(); table["material"]=table.session.astype(str)+":"+table.trial_id.str.rsplit(':',n=1).str[-1]
    if (table.groupby("material").original_label.nunique()!=1).any(): raise ValueError("Material labels differ by participant")
    materials=table[["material","session","original_label"]].drop_duplicates().sort_values("material").reset_index(drop=True)
    subjects=sorted(table.subject_id.unique())
    if len(subjects)!=15 or len(materials)!=72: raise ValueError("Unexpected cohort")
    rng=np.random.default_rng(seed)
    subject_weights=np.zeros((n,15),dtype=int); selections=rng.integers(0,15,size=(n,15))
    np.add.at(subject_weights,(np.repeat(np.arange(n),15),selections.ravel()),1)
    material_weights=np.zeros((n,72),dtype=int)
    for _,rows in materials.groupby(["session","original_label"]):
        if len(rows)!=6: raise ValueError("Unbalanced original material classes")
        choices=rng.choice(rows.index.to_numpy(),size=(n,6),replace=True)
        np.add.at(material_weights,(np.repeat(np.arange(n),6),choices.ravel()),1)
    return subjects,materials,subject_weights,material_weights


def protocol_scores(frame,subjects,materials,subject_weights,material_weights,task):
    frame=frame.copy(); y=frame.original_label.to_numpy(); p=probabilities(frame)
    frame["material"]=frame.session.astype(str)+":"+frame.trial_id.str.rsplit(':',n=1).str[-1]
    frame["correct"]=(p[:,1:].argmax(1)==(y==3)) if task=="binary" else (p.argmax(1)==COARSE[y])
    matrix=frame.groupby(["subject_id","material"]).correct.mean().unstack().loc[subjects,materials.material].to_numpy(dtype=float)
    if not np.isfinite(matrix).all(): raise ValueError("Incomplete crossed test coverage")
    labels=materials.original_label.to_numpy(); labels=(labels==3).astype(int) if task=="binary" else COARSE[labels]
    included=materials.original_label.to_numpy()!=0 if task=="binary" else np.ones(72,dtype=bool)
    scores=[]; point=[]
    for c in range(2 if task=="binary" else 3):
        mask=(labels==c)&included
        correct=subject_weights@matrix[:,mask]
        weights=material_weights[:,mask]
        scores.append((correct*weights).sum(1)/(subject_weights.sum(1)*weights.sum(1)))
        point.append(float(matrix[:,mask].mean()))
    return float(np.mean(point)),np.mean(scores,axis=0)


def analyze(session_run,reve_run,output,plan_path):
    if json.loads(plan_path.read_text())!=plan(): raise ValueError("Changed sensitivity declaration")
    references={}
    for folder in (session_run,reve_run):
        if not json.loads((folder/"verification.json").read_text())["passed"]:
            raise ValueError("Primary study has not passed replay")
        references.update(json.loads((folder/"comparison.json").read_text())["models"])
    frames={name:pd.concat([pd.read_csv(session_run/f"predictions_{name}_source{s}.csv") for s in (1,2,3)],ignore_index=True)
            for name in ("mean_mlp","transformer")}
    frames.update({name:pd.read_csv(session_run/f"predictions_{name}.csv") for name in ("bandpower","duration")})
    frames.update({name:pd.read_csv(reve_run/f"predictions_{name}.csv") for name in ("reve_pretrained","reve_random42")})
    table=frames["duration"][["trial_id","subject_id","original_label","session"]].drop_duplicates()
    subjects,materials,sw,mw=crossed_weights(table)
    summaries={}; scored={}
    for name,frame in frames.items():
        views=protocol_views(frame)
        views["mean_unseen"]=pd.concat([views["unseen_cycle1"],views["unseen_cycle2"]],ignore_index=True)
        summaries[name]={}; scored[name]={}
        for protocol in ("same_session","mean_unseen",*(["mixed_sessions"] if "mixed_sessions" in views else [])):
            summaries[name][protocol]={}; scored[name][protocol]={}
            for task in ("coarse3","binary"):
                point,draws=protocol_scores(views[protocol],subjects,materials,sw,mw,task)
                key="mean_coarse3_BA" if task=="coarse3" else "mean_binary_BA"
                if abs(point-references[name][protocol][key])>1e-12:
                    raise ValueError("Crossed summary differs from verified primary point estimate")
                summaries[name][protocol][task]={"BA":point,"crossed_percentile_95":np.quantile(draws,[.025,.975]).tolist()}
                scored[name][protocol][task]=draws
    contrasts=[]
    def contrast(a,pa,b,pb,task):
        difference=scored[a][pa][task]-scored[b][pb][task]
        contrasts.append({"model_a":a,"protocol_a":pa,"model_b":b,"protocol_b":pb,"task":task,
                          "BA_difference":summaries[a][pa][task]["BA"]-summaries[b][pb][task]["BA"],
                          "crossed_percentile_95":np.quantile(difference,[.025,.975]).tolist()})
    for task in ("coarse3","binary"):
        for name in frames: contrast(name,"mean_unseen",name,"same_session",task)
        for protocol in ("same_session","mean_unseen"):
            contrast("transformer",protocol,"mean_mlp",protocol,task)
        for protocol in ("mixed_sessions","same_session","mean_unseen"):
            contrast("reve_pretrained",protocol,"reve_random42",protocol,task)
    paths=list(session_run.glob("predictions*.csv"))+list(reve_run.glob("predictions*.csv"))
    paths.extend(folder/name for folder in (session_run,reve_run) for name in ("comparison.json","verification.json"))
    report={"plan":plan(),"plan_sha256":sha256(plan_path),"source_sha256":sha256(Path(__file__)),
            "prediction_hashes":{str(p.resolve()):sha256(p) for p in paths},"subject_weight_digest":digest(sw.tolist()),
            "material_weight_digest":digest(mw.tolist()),"models":summaries,"contrasts":contrasts}
    output.mkdir(parents=True,exist_ok=True); write_json(output/"comparison.json",report)
    return report


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__); parser.add_argument("action",choices=["plan","analyze","verify"])
    for name in ("plan","session-run","reve-run","output"): parser.add_argument("--"+name,type=Path)
    a=parser.parse_args()
    if a.action=="plan":
        if a.plan.exists(): raise FileExistsError("Existing sensitivity declaration")
        write_json(a.plan,plan()); print("Saved crossed-material sensitivity declaration")
    else:
        expected=json.loads((a.output/"comparison.json").read_text()) if a.action=="verify" else None
        result=analyze(a.session_run,a.reve_run,a.output,a.plan)
        if expected is not None and expected!=result: raise ValueError("Changed sensitivity replay")
        if a.action=="verify": write_json(a.output/"verification.json",{"passed":True,"scope":"Input/source/plan hashes and complete deterministic crossed bootstrap recomputation"})
        print(json.dumps({"models":len(result["models"]),"contrasts":len(result["contrasts"]),"action":a.action}))
