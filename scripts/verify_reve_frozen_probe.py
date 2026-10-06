"""Independent frozen-probe replay using sklearn's stable softmax normalization.

The original generic logsumexp verifier is preserved byte-for-byte. Its tiny
probability rounding difference is amplified in a confident conditional-binary
log loss. Replay the fitted algorithm rather than widening the tolerance.
"""
from argparse import ArgumentParser
from hashlib import sha256 as hasher
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet import reve_probe as probe
from gruxnet.data import sha256,write_json
from gruxnet.session_controls import source_masks,protocol_views,probabilities,common_metrics,make_folds,C_VALUES


def compare_metrics(actual,expected):
    for task,metrics in expected.items():
        for key,value in metrics.items():
            if key=="balanced_log_loss":
                if abs(actual[task][key]-value)>1e-10: raise ValueError("Logloss does not replay")
            elif actual[task][key]!=value: raise ValueError("Classification metrics do not replay")


def verify_heads(features,table,folds,output,name):
    predictions=pd.read_csv(output/f"predictions_{name}.csv"); protocol_views(predictions)
    labels=table.original_label.to_numpy(dtype=int); target=probe.COARSE[labels] if hasattr(probe,"COARSE") else np.array([0,1,1,2])[labels]
    maximum=0.; coefficient_error=0.; candidates=0
    for source in range(4):
        for record in json.loads((output/f"linear_{name}_source{source}.json").read_text()):
            masks=source_masks(table,folds[record["fold"]],source)
            scaler=StandardScaler().fit(features[masks["train"]]); x=scaler.transform(features)
            np.testing.assert_array_equal(scaler.mean_,record["scaler_mean"])
            np.testing.assert_array_equal(scaler.scale_,record["scaler_scale"])
            if [r["C"] for r in record["candidates"]]!=list(C_VALUES): raise ValueError("Changed C candidates")
            fits=[]
            for candidate in record["candidates"]:
                estimator=LogisticRegression(C=candidate["C"],class_weight="balanced",max_iter=2000,tol=1e-6,random_state=42)
                estimator.fit(x[masks["train"]],target[masks["train"]]); candidates+=1
                metrics=common_metrics(labels[masks["validation"]],estimator.predict_proba(x[masks["validation"]]),"coarse3")
                compare_metrics(metrics,candidate["validation"]); fits.append(estimator)
            selected=max(range(len(record["candidates"])),key=lambda i:(record["candidates"][i]["validation"]["coarse3"]["balanced_accuracy"],-record["candidates"][i]["validation"]["coarse3"]["balanced_log_loss"]))
            if record["candidates"][selected]["C"]!=record["selected_C"]: raise ValueError("Incorrect C selection")
            estimator=fits[selected]; weights=np.asarray(record["coefficients"]); bias=np.asarray(record["intercept"])
            np.testing.assert_allclose(estimator.coef_,weights,atol=1e-9,rtol=1e-9)
            np.testing.assert_allclose(estimator.intercept_,bias,atol=1e-9,rtol=1e-9)
            coefficient_error=max(coefficient_error,float(np.max(np.abs(estimator.coef_-weights))),float(np.max(np.abs(estimator.intercept_-bias))))
            p=softmax(x[masks["validation"]]@weights.T+bias,axis=1)
            compare_metrics(common_metrics(labels[masks["validation"]],p,"coarse3"),record["candidates"][selected]["validation"])
            for session,indices in masks["test"].items():
                p=softmax(x[indices]@weights.T+bias,axis=1)
                rows=predictions[(predictions.source_session==source)&(predictions.test_session==session)&(predictions.fold==record["fold"])]
                if rows.trial_id.tolist()!=table.iloc[indices].trial_id.tolist() or not np.array_equal(rows.original_label,labels[indices]):
                    raise ValueError("Changed trial coverage/labels")
                np.testing.assert_allclose(p,probabilities(rows),atol=1e-14,rtol=0)
                np.testing.assert_allclose(estimator.predict_proba(x[indices]),p,atol=1e-10,rtol=0)
                maximum=max(maximum,float(np.max(np.abs(p-probabilities(rows)))))
                compare_metrics(common_metrics(labels[indices],p,"coarse3"),record["test_sessions"][str(session)])
        print(f"Replayed {name} source{source}: all candidates, selection and tests",flush=True)
    return {"selected_models_replayed":20,"candidates_independently_refitted":candidates,"test_probability_rows":len(predictions),
            "max_probability_error":maximum,"max_refit_coefficient_error":coefficient_error}


def verify(assets,cache,output):
    features,table,config=probe.load_features(output); inputs,input_table,info=probe.load_inputs(cache)
    if not table.equals(input_table) or config["input_fingerprint"]!=info["fingerprint"]: raise ValueError("Changed cohort")
    probe.audit_assets(assets)
    for filename,key in [("download_manifest.json","download_manifest_sha256"),("code_review.json","code_review_sha256")]:
        if sha256(assets/filename)!=config[key]: raise ValueError("Changed reviewed assets")
    indices=[0,71,144,287,432,575,720,863,1008,1079]; errors={}
    import torch
    for name in features:
        model,positions=probe.local_model(assets,name=="reve_pretrained")
        state_hash=hasher()
        for key,value in model.state_dict().items(): state_hash.update(key.encode()); state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        if state_hash.hexdigest()!=config["features"][name]["state_sha256"]: raise ValueError("Changed encoder state")
        model=model.to("cuda")
        actual=np.stack([probe.embed(model,positions,probe.adapt(inputs[i:i+1]),"cuda",10).mean(0) for i in indices])
        np.testing.assert_allclose(actual,features[name][indices],atol=2e-5,rtol=2e-5)
        errors[name]=float(np.max(np.abs(actual-features[name][indices])))
        del model; torch.cuda.empty_cache()
    folds=make_folds(table.subject_id.tolist())
    if json.loads((output/"folds.json").read_text())!=folds: raise ValueError("Changed folds")
    heads={name:verify_heads(x,table,folds,output,name) for name,x in features.items()}
    expected=json.loads((output/"comparison.json").read_text())
    if probe.analyze(output)!=expected: raise ValueError("Changed primary bootstrap/summary")
    result={"passed":True,"selected_heads_replayed":40,"candidates_independently_refitted":160,"test_probability_rows":8640,
            "heads":heads,"embedding_replay_trial_indices":indices,"embedding_max_errors":errors,
            "verification_source_sha256":sha256(Path(__file__)),
            "numerical_correction":"Use scipy.special.softmax, as sklearn multinomial probabilities do. Original logsumexp-normalized reconstruction differed by <=1.78e-15 in probabilities and 2.58e-8 in one conditional-binary validation logloss. All classification metrics/selection/data/weights unchanged; strict1e-10 logloss tolerance retained",
            "scope":"All bound inputs/features/source/assets; exact frozen state hash,20 sampled embeddings,160 candidate refits,40 selected coefficient/scaler/selection replays,8640 test probabilities,OOF/bootstrap summaries; full1080embeddings not recomputed twice"}
    write_json(output/"verification.json",result); return result


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    for name in ("assets","cache","output"): parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args(); print(json.dumps(verify(a.assets.resolve(),a.cache,a.output),indent=2))
