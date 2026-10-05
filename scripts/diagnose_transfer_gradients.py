"""Training-only covariate/gradient diagnostics; does not train or select checkpoints."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256,write_json
from gruxnet.train import scores,seed_everything
from gruxnet.transfer_controls import DATASET_INDEX,FeatureMLP,frame_for,load_inputs


def diagnose(output,plan):
    config=json.loads((output/"config.json").read_text())
    x,target,z,sources,_=load_inputs(Path(config["pack"]),Path(config["common_cache"]))
    folds=json.loads((output/"folds.json").read_text())
    device=torch.device(config["hardware"]["device"])
    records=[]
    for seed in [42,43,44]:
        for fold in folds:
            masks,table,features,scaler=frame_for(x,target,z,sources,fold,"joint")
            transformed=scaler.transform(features)
            for condition in ["initial","single","joint","joint_heads"]:
                seed_everything(seed+1000*fold["fold"])
                torch.set_num_threads(2)
                model=FeatureMLP(condition=="joint_heads").to(device)
                checksum=None
                if condition!="initial":
                    path=output/"models"/f"{condition}_seed{seed}_fold{fold['fold']}"/"primary.pt"
                    checkpoint=torch.load(path,map_location=device,weights_only=False)
                    model.load_state_dict(checkpoint["state_dict"])
                    checksum=sha256(path)
                model.eval()
                gradients,domains={},{}
                for dataset in ["SEEDIV","DEAP","GAMEEMO"]:
                    mask=(table.dataset==dataset).to_numpy()
                    inputs=torch.from_numpy(transformed[mask]).to(device)
                    labels=torch.tensor(table.label[mask].to_numpy(),dtype=torch.long,device=device)
                    ids=torch.full((len(inputs),),DATASET_INDEX[dataset],dtype=torch.long,device=device)
                    model.zero_grad(set_to_none=True)
                    logits=model(inputs,ids)
                    per_trial=nn.functional.cross_entropy(logits,labels,reduction="none")
                    loss=torch.stack([per_trial[labels==c].mean() for c in [0,1]]).mean()
                    loss.backward()
                    gradient=torch.cat([p.grad.detach().flatten() for p in model.backbone.parameters()]).cpu().numpy()
                    gradients[dataset]=gradient
                    domains[dataset]={"training_trials":int(mask.sum()),"class_balanced_loss":float(loss.detach()),
                                      "backbone_gradient_norm":float(np.linalg.norm(gradient)),
                                      "mean_feature_offset_l2":float(np.linalg.norm(transformed[mask].mean(0))),
                                      "training_metrics":scores(labels.cpu().numpy(),logits.detach().softmax(-1)[:,1].cpu().numpy())}
                cosines={}
                for a,b in [("SEEDIV","DEAP"),("SEEDIV","GAMEEMO"),("DEAP","GAMEEMO")]:
                    norm=float(np.linalg.norm(gradients[a])*np.linalg.norm(gradients[b]))
                    cosines[f"{a}:{b}"]=float(np.dot(gradients[a],gradients[b])/norm) if norm>0 else None
                records.append({"seed":seed,"fold":fold["fold"],"condition":condition,
                                "checkpoint_sha256":checksum,"domains":domains,"gradient_cosines":cosines})
    result={"development_only":True,"training_only":True,"plan_sha256":sha256(plan),"records":records,
            "scope":"Complete original training trials; no validation/test inputs. Gradients of class-balanced loss in eval mode on shared backbone.",
            "limitations":"Descriptive association, not causal evidence about label semantics or a new method; no tuning/retraining follows these diagnostics."}
    (output/"gradient_plan.json").write_bytes(plan.read_bytes())
    write_json(output/"gradient_diagnostics.json",result)
    summary={}
    for condition in ["initial","single","joint","joint_heads"]:
        items=[r for r in records if r["condition"]==condition]
        summary[condition]={key:{"mean_cosine":float(np.mean([r["gradient_cosines"][key] for r in items])),
                                 "negative_fraction":float(np.mean([r["gradient_cosines"][key]<0 for r in items]))}
                            for key in ["SEEDIV:DEAP","SEEDIV:GAMEEMO"]}
    write_json(output/"gradient_summary.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    diagnose(args.output,args.plan)
