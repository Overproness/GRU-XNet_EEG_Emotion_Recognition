"""Audit design feasibility across corrected corpora without fitting a model."""
from argparse import ArgumentParser
from hashlib import sha256
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gruxnet.data import write_json


def audit(cache,output):
    metadata=json.loads((cache/"prepared.json").read_text())
    path=cache/"trials.csv"
    if sha256(path.read_bytes()).hexdigest()!=metadata["trials_sha256"]:
        raise ValueError("Changed previously verified corrected trial metadata")
    table=pd.read_csv(path)
    if len(table)!=2167 or table.trial_id.duplicated().any():
        raise ValueError("Unexpected corrected retained population")
    rows=[]
    for name,part in table.groupby("dataset",sort=True):
        part=part.copy()
        if name=="SEEDIV":
            part["material"]=part.trial_id.str.split(":").str[2:].str.join(":")
        else:
            part["material"]=part.trial_id.str.rsplit(":",n=1).str[-1]
        counts=part.groupby(["material","label"]).size().unstack(fill_value=0)
        for material,count in counts.iterrows():
            group=part[part.material==material]
            rows.append({"dataset":name,"material":material,"negative_trials":int(count.get(0,0)),
                         "positive_trials":int(count.get(1,0)),"participants":int(group.subject_id.nunique()),
                         "window_count_min":int(group.windows.min()),"window_count_max":int(group.windows.max())})
    summary=pd.DataFrame(rows)
    output.mkdir(parents=True,exist_ok=True)
    summary.to_csv(output/"material_label_counts.csv",index=False)
    game=summary[summary.dataset=="GAMEEMO"].set_index("material")
    assignments=[]
    for validation,test in itertools.permutations(sorted(game.index),2):
        training=[g for g in game.index if g not in (validation,test)]
        def both(keys):
            return bool((game.loc[keys,["negative_trials","positive_trials"]].sum()>0).all())
        assignments.append({"train":sorted(training),"validation":[validation],"test":[test],
                            "both_classes_train":both(training),"both_classes_validation":both([validation]),
                            "both_classes_test":both([test])})
    # Cohort-level eligibility is necessary, not sufficient for participant-disjoint fold eligibility.
    result={"development_only":True,"models_fitted":0,
            "bound_trial_metadata_sha256":metadata["trials_sha256"],
            "bound_feature_metadata_sha256":sha256((cache/"prepared.json").read_bytes()).hexdigest(),
            "audit_script_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),
            "population":{name:{"retained_trials":int(len(part)),"participants":int(part.subject_id.nunique()),
                                "conditions":int(summary[summary.dataset==name].material.nunique())}
                          for name,part in table.groupby("dataset")},
            "gameemo_all_12_two_train_one_validation_one_test_assignments":assignments,
            "eligible_gameemo_cohort_assignments":sum(all(r[f"both_classes_{role}"] for role in ("train","validation","test")) for r in assignments),
            "scope":"Retained corrected binary cohort only; neutral/midpoint exclusions inherited. No neural outcomes, model selection, waveform audit or unseen-corpus evaluation. GAMEEMO game IDs are conditions, not identical audiovisual trajectories. Cohort class coverage does not guarantee coverage in held-out participant folds."}
    write_json(output/"feasibility.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.cache,args.output)
