"""Paired development summaries and a standalone figure; no model selection."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.data import write_json


def analyze(output):
    predictions=pd.read_csv(output/"trial_predictions.csv")
    report=json.loads((output/"neural_comparison.json").read_text())
    linear=json.loads((output/"linear_comparison.json").read_text())
    metrics=json.loads((output/"model_metrics.json").read_text())
    seeds=sorted(int(value) for value in predictions.seed.unique())
    subjects=sorted(predictions.subject_id.unique())
    subject_scores={}
    for (condition,budget),rows in predictions.groupby(["condition","budget"]):
        subject_scores[f"{condition}:{budget}"]={s:float(np.mean([
            balanced_accuracy_score(t.label,t.positive_probability>=.5)
            for _,t in group.groupby("seed")])) for s,group in rows.groupby("subject_id")}
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(subjects),size=(10000,len(subjects)))
    comparisons={}
    pairs=[(f"{name}:{budget}","single:primary") for budget in ["primary","exposure"]
           for name in ["plus_deap","plus_game","joint","joint_heads"]]
    pairs.extend([(f"joint_heads:{budget}",f"joint:{budget}") for budget in ["primary","exposure"]])
    for condition,reference in pairs:
        difference=np.array([subject_scores[condition][s]-subject_scores[reference][s] for s in subjects])
        seed_differences={str(seed):report[condition]["per_seed"][str(seed)]["balanced_accuracy"]-
                                   report[reference]["per_seed"][str(seed)]["balanced_accuracy"] for seed in seeds}
        comparisons[f"{condition} minus {reference}"]={"mean_BA_difference":float(difference.mean()),
            "paired_subject_percentile_95":np.quantile(difference[draws].mean(1),[.025,.975]).tolist(),
            "per_seed_BA_difference":seed_differences}
    record={"development_only":True,"paired_comparisons":comparisons,"subjects":len(subjects),"seeds":seeds,
            "bootstrap":"Average paired per-subject differences over initializations, then resample 15 subjects 10,000 times",
            "limitations":"One fixed fold grouping, overlapping training sets, three initializations. Interval describes held-out participant variation, not all training or protocol-selection uncertainty. No confirmatory significance or causal-mechanism claim."}
    write_json(output/"paired_comparison.json",record)
    behavior={}
    for condition in ["single","plus_deap","plus_game","joint","joint_heads"]:
        for budget in ["primary","exposure"]:
            selected=[m["budgets"][budget] for m in metrics if m["condition"]==condition and budget in m["budgets"]]
            if not selected:continue
            behavior[f"{condition}:{budget}"]={
                "mean_train_target_BA":float(np.mean([m["train_target"]["balanced_accuracy"] for m in selected])),
                "mean_validation_BA":float(np.mean([m["validation"]["balanced_accuracy"] for m in selected])),
                "mean_selected_step":float(np.mean([m["selected_step"] for m in selected]))}
    write_json(output/"training_behavior.json",behavior)
    names=["single","plus_deap","plus_game","joint","joint_heads"]
    labels=["SEED-IV only","+ DEAP","+ GAMEEMO","+ both, shared","+ both, heads"]
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,budget,title in zip(axes,["primary","exposure"],["Same total updates: 600","Available target exposure: 36,000 presentations"]):
        keys=[f"{name}:{budget}" if name!="single" else "single:primary" for name in names]
        values=[100*report[k]["mean_seed_BA"] for k in keys]
        deviations=[100*report[k]["std_seed_BA"] for k in keys]
        ax.bar(np.arange(5),values,yerr=deviations,capsize=4,color=["#466e9a","#82a589","#cc994f","#a0708d","#6c86ad"],alpha=.9)
        for i,value in enumerate(values):
            ax.text(i,20,f"{value:.2f}%",ha="center",color="white",fontweight="bold",fontsize=10)
        if budget=="primary":
            ax.scatter(np.arange(4),[100*linear[n]["test_out_of_fold"]["balanced_accuracy"] for n in names[:4]],
                       marker="D",s=30,color="black",label="Anchored linear control",zorder=4)
        ax.axhline(50,color="gray",linestyle="--",label="Chance BA")
        ax.set_xticks(np.arange(5),labels,rotation=18,ha="right",fontsize=9)
        ax.set_ylim(0,90)
        ax.set_title(title,fontsize=11)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("SEED-IV held-out trial balanced accuracy (%)")
    axes[0].legend(fontsize=8,loc="upper left")
    fig.suptitle("Feature-MLP development controls: mean ± SD over 3 initializations, 5 participant folds",fontsize=12)
    fig.tight_layout()
    fig.savefig(output/"neural_transfer_comparison.png",dpi=180)
    plt.close(fig)
    print(json.dumps({"mean_neural_BA":{k:v["mean_seed_BA"] for k,v in report.items()},
                      "linear_BA":{k:v["test_out_of_fold"]["balanced_accuracy"] for k,v in linear.items()},
                      "elapsed_training_seconds":sum(m["elapsed_seconds"] for m in metrics),
                      "maximum_peak_allocated_mib":max(m["peak_cuda_allocated_mib"] for m in metrics)}))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    analyze(parser.parse_args().output)
