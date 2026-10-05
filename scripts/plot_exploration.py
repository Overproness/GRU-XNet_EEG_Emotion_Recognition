"""Standalone development figures; no inference, fitting, selection, or paper changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[2]
RUNS = WORKSPACE / "publication_runs"
native = json.loads((RUNS / "native_seediv_diagnostic/comparison.json").read_text())
joint = json.loads((RUNS / "joint_seediv_diagnostic/comparison.json").read_text())
fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
conditions = [(["native4_native62","native4_common14"],["62 electrodes","14 electrodes"],native,.25,"Native four-class task"),
              (["binary_native62","binary_common14"],["62 electrodes","14 electrodes"],native,.5,"Coarse binary task"),
              (["single","joint_global","joint_by_dataset"],["SEED-IV only","Joint global\nscaler","Joint per-dataset\nscalers"],joint,.5,"Binary pooling control")]
for ax,(names,labels,report,chance,title) in zip(axes,conditions):
    values = [100*report[name]["test_out_of_fold"]["balanced_accuracy"] for name in names]
    ax.bar(np.arange(len(names)),values,color=["#426c9c","#c98940","#68a389"][:len(names)],alpha=.85)
    for index,name in enumerate(names):
        points = [100*fold["test"]["balanced_accuracy"] for fold in report[name]["folds"]]
        ax.scatter(index+np.linspace(-.11,.11,5),points,s=19,color="black",zorder=3,label="Participant folds" if index==0 else None)
        ax.text(index,12,f"{values[index]:.2f}%",ha="center",fontsize=10,color="white",fontweight="bold")
    ax.axhline(100*chance,color="gray",linestyle="--",label="Chance balanced accuracy")
    ax.set_xticks(np.arange(len(names)),labels,fontsize=9)
    ax.set_title(title,fontsize=11)
    ax.set_ylim(0,90)
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Held-out trial balanced accuracy (%)")
axes[0].legend(fontsize=8,loc="upper left")
fig.suptitle("SEED-IV exploratory controls: fixed participant folds, classical trial features",fontsize=12)
fig.tight_layout()
fig.savefig(RUNS / "native_seediv_diagnostic/comparison.png",dpi=180)
plt.close(fig)

# Descriptive paired subject uncertainty; training folds share participants.
saved=pd.read_csv(RUNS / "joint_seediv_diagnostic/trial_predictions.csv")
from sklearn.metrics import balanced_accuracy_score
subject_scores={c: {s:float(balanced_accuracy_score(t.label,t.prediction)) for s,t in rows.groupby("subject_id")}
                for c,rows in saved.groupby("condition")}
rng=np.random.default_rng(42)
subjects=sorted(subject_scores["single"])
draws=rng.integers(0,len(subjects),size=(10000,len(subjects)))
comparison={}
for name in ["joint_global","joint_by_dataset"]:
    difference=np.array([subject_scores[name][s]-subject_scores["single"][s] for s in subjects])
    comparison[name]={"balanced_accuracy_difference":float(difference.mean()),
                      "paired_subject_bootstrap_percentile_95":np.quantile(difference[draws].mean(1),[.025,.975]).tolist()}
output={"development_only":True,"comparisons":comparison,
        "scope":"10,000 paired test-subject resamples. Within each subject the task has identical 36/18 class counts, so mean subject BA equals pooled trial BA.",
        "limitations":"Only one predefined fold grouping; training sets overlap across folds. Intervals describe test-subject variability, not all training/selection uncertainty, and do not establish neural negative transfer or mechanism."}
(RUNS / "joint_seediv_diagnostic/paired_comparison.json").write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
print(json.dumps(output))
