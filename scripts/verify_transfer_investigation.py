"""Reconstruct all feature-MLP selections, exact exposures and test probabilities."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.data import digest,sha256,write_json
from gruxnet.deap_control import candidate_is_better
from gruxnet.train import scores,seed_everything
from gruxnet.transfer_controls import CONDITIONS,FeatureMLP,balanced_batches,frame_for,load_inputs,predict
from scripts.native_seediv_diagnostic import make_folds


def verify(output):
    config=json.loads((output/"config.json").read_text())
    if "target_dataset" in config:
        from gruxnet.transfer_extension import target_mapping
        with target_mapping(config["target_dataset"]):
            return verify_selected(output)
    return verify_selected(output)


def verify_selected(output):
    config=json.loads((output/"config.json").read_text())
    for name,expected in config["source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[1]/"gruxnet"/name)!=expected:
            raise ValueError("Evaluation source changed: "+name)
    assert sha256(output/"plan.json")==config["plan_sha256"]
    plan=json.loads((output/"plan.json").read_text())
    if "target_dataset" in config:
        from gruxnet.transfer_extension import load_extension_inputs,participant_folds
        from gruxnet.transfer_controls import DATASET_INDEX
        assert config["conditions"]==CONDITIONS
        assert config["dataset_index"]==DATASET_INDEX
        x,target,z,sources,binding=load_extension_inputs(config)
        expected_folds=participant_folds(target.subject_id.tolist())
    else:
        x,target,z,sources,binding=load_inputs(Path(config["pack"]),Path(config["common_cache"]))
        expected_folds=make_folds(target.subject_id.tolist())
    assert binding==config["input_binding"]
    folds=json.loads((output/"folds.json").read_text())
    assert folds==expected_folds and list(CONDITIONS)==plan["conditions"]
    all_saved=pd.read_csv(output/"trial_predictions.csv")
    reconstructed=[]
    reconstructed_models=[]
    count_models,count_checkpoints,count_predictions=0,0,0
    device=torch.device(config["hardware"]["device"])
    for seed in plan["seeds"]:
        for fold in folds:
            target_streams=[]
            initials=[]
            for condition in CONDITIONS:
                run=output/"models"/f"{condition}_seed{seed}_fold{fold['fold']}"
                declaration=json.loads((run/"config.json").read_text())
                result=json.loads((run/"metrics.json").read_text())
                masks,table,features,scaler=frame_for(x,target,z,sources,fold,condition)
                pd.testing.assert_frame_equal(pd.read_csv(run/"training_trials.csv"),table)
                for name,key in [("training_trials.csv","training_trials_sha256"),("normalizer.npz","normalizer_sha256")]:
                    assert sha256(run/name)==declaration[key]
                with np.load(run/"normalizer.npz",allow_pickle=False) as normalizer:
                    np.testing.assert_array_equal(normalizer["mean"],scaler.mean_)
                    np.testing.assert_array_equal(normalizer["scale"],scaler.scale_)
                assert declaration["scaler_subjects"]==sorted(target[masks["train"]].subject_id.unique())
                assert declaration["max_steps"]==600*len(CONDITIONS[condition])
                assert declaration["target_presentations_max"]==36000
                batches=balanced_batches(table,CONDITIONS[condition],declaration["max_steps"],seed,fold["fold"])
                assert hashlib.sha256(batches.tobytes()).hexdigest()==declaration["sampling_sha256"]
                quota=60//(2*len(CONDITIONS[condition]))
                target_streams.append((batches[:,:quota].reshape(-1),batches[:,quota:2*quota].reshape(-1)))
                seed_everything(seed+1000*fold["fold"])
                torch.set_num_threads(2)
                model=FeatureMLP(condition=="joint_heads").to(device)
                backbone=digest({k:v.detach().cpu().tolist() for k,v in model.backbone.state_dict().items()})
                head=digest({k:v.detach().cpu().tolist() for k,v in model.heads[0].state_dict().items()})
                assert backbone==declaration["initial_backbone_sha256"] and head==declaration["initial_target_head_sha256"]
                initials.append((backbone,head))
                history=json.loads((run/"history.json").read_text())
                assert [row["step"] for row in history]==list(range(25,declaration["max_steps"]+1,25))
                validation_x=torch.from_numpy(scaler.transform(x[masks["validation"]])).to(device)
                validation_d=torch.zeros(len(validation_x),dtype=torch.long,device=device)
                training_x=torch.from_numpy(scaler.transform(x[masks["train"]])).to(device)
                training_d=torch.zeros(len(training_x),dtype=torch.long,device=device)
                test_x=torch.from_numpy(scaler.transform(x[masks["test"]])).to(device)
                test_d=torch.zeros(len(test_x),dtype=torch.long,device=device)
                saved=pd.read_csv(run/"test_trial_predictions.csv")
                assert sha256(run/"test_trial_predictions.csv")==result["predictions_sha256"]
                for budget,metric in result["budgets"].items():
                    budget_steps=600 if budget=="primary" else declaration["max_steps"]
                    best_score,best_loss,best_step=-1.,float("inf"),None
                    for record in history:
                        if record["step"]>budget_steps:break
                        score=record["validation"]["balanced_accuracy"]
                        loss=record["validation_balanced_log_loss"]
                        if candidate_is_better(score,loss,best_score,best_loss):
                            best_score,best_loss,best_step=score,loss,record["step"]
                    assert metric["selected_step"]==best_step
                    assert sha256(run/f"{budget}.pt")==metric["checkpoint_sha256"]
                    checkpoint=torch.load(run/f"{budget}.pt",map_location=device,weights_only=False)
                    assert checkpoint["step"]==best_step
                    model.load_state_dict(checkpoint["state_dict"])
                    val=predict(model,validation_x,validation_d)
                    assert scores(target.label[masks["validation"]],val)==metric["validation"]
                    training=predict(model,training_x,training_d)
                    assert scores(target.label[masks["train"]],training)==metric["train_target"]
                    probabilities=predict(model,test_x,test_d)
                    rows=saved[saved.budget==budget].reset_index(drop=True)
                    assert rows.trial_id.tolist()==target[masks["test"]].trial_id.tolist()
                    np.testing.assert_array_equal(rows.label,target.label[masks["test"]])
                    np.testing.assert_allclose(rows.positive_probability,probabilities,atol=1e-7,rtol=0)
                    assert scores(rows.label,probabilities)==metric["test"]
                    reconstructed.append(rows)
                    count_checkpoints+=1
                    count_predictions+=len(rows)
                count_models+=1
                reconstructed_models.append(result)
            assert len(set(initials))==1
            for negative,positive in target_streams[1:]:
                np.testing.assert_array_equal(negative,target_streams[0][0])
                np.testing.assert_array_equal(positive,target_streams[0][1])
    actual=pd.concat(reconstructed,ignore_index=True)
    assert reconstructed_models==json.loads((output/"model_metrics.json").read_text())
    pd.testing.assert_frame_equal(actual,all_saved)
    report=json.loads((output/"neural_comparison.json").read_text())
    for (condition,budget),rows in all_saved.groupby(["condition","budget"]):
        entry=report[f"{condition}:{budget}"]
        for seed,group in rows.groupby("seed"):
            assert not group.trial_id.duplicated().any() and set(group.trial_id)==set(target.trial_id)
            assert scores(group.label,group.positive_probability)==entry["per_seed"][str(seed)]
        values=[m["balanced_accuracy"] for m in entry["per_seed"].values()]
        assert entry["mean_seed_BA"]==float(np.mean(values))
        assert entry["std_seed_BA"]==float(np.std(values,ddof=1))
    result={"passed":True,"models_checked":count_models,"selected_checkpoints_checked":count_checkpoints,
            "test_probabilities_reproduced":count_predictions,"training_only_scalers_refitted":True,
            "paired_initializations_equal":True,"exact_target_exposure_streams_equal":True,
            "validation_selections_reconstructed":True,"all_aggregate_metrics_reproduced":True,
            "predictions_sha256":sha256(output/"trial_predictions.csv"),
            "verifier_sha256":sha256(Path(__file__)),
            "scope":"Selected checkpoint replay and metrics, frozen scaler refits, source-participant exclusions, sampling streams and history selection. Does not rerun every optimizer update or independently regenerate original features."}
    write_json(output/"verification.json",result)
    print(json.dumps(result))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    verify(parser.parse_args().output)
