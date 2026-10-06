"""Make standalone scientific figures from the verified fixed-duration controls."""
from argparse import ArgumentParser
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    report = json.loads((output/"comparison.json").read_text())
    names = list(report["conditions"])
    def label(name):
        return name.replace("mean_mlp","MLP").replace("transformer","Transformer").replace("linear","Logistic").replace("_"," / ")
    fig, axes = plt.subplots(1,2,figsize=(13,6.5),sharey=True)
    y = np.arange(len(names))
    for ax,task,key,chance in zip(axes,["coarse3","binary"],["mean_coarse3_BA","mean_binary_BA"],[1/3,.5]):
        values = [report["conditions"][name][key] if key in report["conditions"][name]
                  else report["conditions"][name][task]["balanced_accuracy"] for name in names]
        ax.barh(y,100*np.array(values),color=["#247e9e" if not name.startswith("linear") else "#929aa5" for name in names])
        ax.axvline(100*chance,color="black",linestyle="--",linewidth=1,label="chance BA")
        for index,value in enumerate(values):
            ax.text(100*value+.6,index,f"{100*value:.2f}",va="center",fontsize=9)
        ax.set_xlim(0,85)
        ax.set_xlabel("Out-of-fold trial balanced accuracy (%)")
        ax.set_title("Three-class valence (all 1,080 trials)" if task=="coarse3" else "Conditional binary valence (810 trials)")
        ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(y,labels=[label(n) for n in names])
    axes[0].invert_yaxis()
    fig.suptitle("SEED-IV development controls: fixed 40-second inputs, participant-separated folds",fontsize=12)
    fig.text(.5,.01,"Neural bars average three initializations; logistic controls use one deterministic fit per fold. Duration diagnostic uses full-trial length only.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(output/"comparison.png",dpi=180)
    plt.close(fig)
    pairs = report["paired_comparisons"]
    fig, axes = plt.subplots(1,2,figsize=(14,8),sharey=True)
    labels = []
    for ax,task in zip(axes,["coarse3","binary"]):
        selected = [p for p in pairs if p["task"]==task]
        labels = [label(p["a"])+" minus "+label(p["b"]) for p in selected]
        for i,pair in enumerate(selected):
            mean = pair["mean_BA_difference"]*100
            low,high = np.array(pair["paired_subject_percentile_95"])*100
            ax.plot([low,high],[i,i],color="#247e9e",linewidth=2)
            ax.plot(mean,i,"o",color="#193c54")
        ax.axvline(0,color="black",linestyle="--",linewidth=1)
        ax.grid(axis="x",alpha=.2)
        ax.set_xlabel("Paired balanced-accuracy difference (percentage points)")
        ax.set_title("Common three-class task" if task=="coarse3" else "Common binary task")
    axes[0].set_yticks(np.arange(len(labels)),labels=labels,fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Factor contrasts: participant bootstrap 95% intervals",fontsize=12)
    fig.text(.5,.01,"10,000 paired participant draws, averaged across three initialization seeds; exploratory and unadjusted for 24 comparisons.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.95))
    fig.savefig(output/"paired_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    plot(args.output)
