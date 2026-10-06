"""Predeclared repeated participant/material controls; previous sources stay frozen."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from .data import digest, sha256, write_json
from .prepare import load_prepared
from .temporal_controls import COARSE, C_VALUES, TemporalControl, common_metrics, infer, load, metric, sequence_features
from .material_controls import assignment, masks as seed_masks, state_digest
from .reve_probe import load_features
from .train import seed_everything

REPO = Path(__file__).resolve().parents[1]
ARMS = ("exposed", "unexposed")
NEURAL = ("mean_mlp", "transformer")
GROUPS = {1: {"participant_seed": 20261007, "material_seed": 20261017},
          2: {"participant_seed": 20261008, "material_seed": 20261018}}
SOURCE_FILES = ["gruxnet/grouped_material_controls.py", "gruxnet/material_controls.py",
                "gruxnet/temporal_controls.py", "gruxnet/session_controls.py", "gruxnet/train.py",
                "gruxnet/data.py", "gruxnet/prepare.py", "gruxnet/reve_probe.py",
                "scripts/native_seediv_diagnostic.py", "scripts/repeated_material_controls.py"]


def plan():
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "development_only": True,
            "research_question_change_approved": False, "groupings": GROUPS,
            "seediv": "Two additional independently seeded participant/material groupings. All1080 trials; same3 sessions,3 material rotations,5 participant folds,9/3/3 people,108/12/24 trials. Repeat both original architectures and all4 linear representations. Reuse original grouping only at neural initialization42 for like-for-like grouping sensitivity; original3-seed mean retained as prior evidence.",
            "deap": "All1264 spreadsheet-rated trials,32 participants,40 Experiment_id video keys; exclude exactly5. Both labels can occur for a video. Two fixed groupings;8 folds of24/4/4 people;5 material rotations. Permute sorted40 videos into5 blocks of8. Test block r; validation r+1; common training block r+2; exposed train r and r+2; unexposed train r+2 and r+3, modulo5. Validation videos unseen in both arms. Each retained trial tested exactly once per grouping.",
            "deap_matching": "Before fitting, independently for each training participant/binary label retain min(eligible exposed,eligible unexposed) trials in each arm; deterministic SHA256 order of group,rotation,fold,participant,label,trial_id (no features/test scores). Exact same participant/class counts, source-only scalers. Missing groups may retain zero for a class; whole train/validation/test must each contain both classes. Declare infeasible splits and stop rather than choose a successful split seed. All test video keys must actually occur in exposed training after matching; all training people must be represented.",
            "observation": "Same first10 non-overlapping4-second windows,common14 physical electrodes,absolute log Welch4-band power56 features. DEAP cache regenerated from hash-checked waveform windows; raw source hashes and spreadsheet checked. Offline full-trial4..40Hz filter context, DEAP384 baseline samples removed. No augmentation/target-population calibration.",
            "models": "Existing meanMLP/temporal transformer backbones. SEED coarse3 heads unchanged19079/19075; DEAP replaces3-class head with fresh2-class head before optimization. No pretrained DEAP encoder experiment in this phase. Bandpower logistic for both corpora; frozen pretrained/randomREVE and duration diagnostic repeated only for SEED. DEAP training-label-only material-prior diagnostic: Laplace1 smoothed per-video positive rate, unseen video falls back to smoothed global training rate, threshold tie negative; no EEG.",
            "neural_initializations": [42], "neural_fits": {"SEEDIV": 360, "DEAP": 320, "total": 680},
            "training": "600 balanced60-trial AdamWupdates,lr.001,wd.01,clip1; SEED20/class,DEAP30/class; validation every25,maxBA then minbalancedlogloss,firsttie; deterministicCUDAmathSDPA,noAMP/scheduler/earlystop. Init42+1000fold+10000rotation, reset stochasticRNG+1000000. Canonical participant/native-emotion/rank streams paired across SEED arms; participant/binary-class/rank across DEAP arms. Initial states paired across arms/groupings/sessions within architecture and output size. One initialization intentionally holds optimizer randomness fixed while varying groupings; does not add independent people/videos.",
            "linear": "Same splits; training-only StandardScaler,balanced LogisticRegression,C.01/.1/1/10,max_iter2000,tol1e-6,random_state42; validationBA/logloss firsttie. SEED4*2groups*3sessions*3rotations*5folds*2arms=720 selected; DEAP1*2groups*5rotations*8folds*2arms=160 selected; total880 heads/3520 candidates. Material-prior has160 nonselected diagnostic fits, noC search.",
            "evaluation": "Report every grouping,model,arm and each session/rotation/fold; aggregate original out-of-fold trial predictions, never average unequal DEAP fold BAs or select a grouping from test. PrimarySEEDcoarse3; secondaryconditionalbinary excludes270neutral with unchanged massfloor/tie; primaryDEAPbinary individual self-report. Combine groupings by averaging correctness per observed person/material cell; repeats are paired development sensitivity, not independent replications/new subjects.",
            "uncertainty": "10000 paired subject-only and crossedsubject/material percentile draws,seed20261006; SEED15people and6videos per12session/nativeemotion strata; DEAP32people and40videos unstratified because video has no single emotional class. Compute weighted class numerators/denominators on actual observed cells so missing16midpoints and individual labels remain respected. Same weights for all arms/models/groupings. Conditionalfixedfit/groupings/cohorts; unadjusted exploratory intervals, no equivalence or causal identity claim.",
            "contrasts": "Unexposed-minus-exposed all6SEED models and4DEAP models; transformer-minus-MLP botharms; SEED pretrained-minus-random botharms. Report allgroup contrasts and combined, SEED2tasks/DEAP1, not cherry-picked significant subsets.",
            "gate": "No research-question/manuscript change. Present complete evidence before proposing pivot; require author approval and archive then-current paper immediately before adoption."}


def folds(subjects, seed, size):
    subjects = sorted(set(subjects))
    if len(subjects) % size: raise ValueError("Unequal participant groups")
    order = np.random.default_rng(seed).permutation(subjects).tolist()
    groups = [order[i:i+size] for i in range(0, len(order), size)]
    return [{"fold": i, "test": group, "validation": groups[(i+1) % len(groups)],
             "train": sorted(set(subjects)-set(group)-set(groups[(i+1) % len(groups)]))}
            for i, group in enumerate(groups)]


def annotate(table, dataset, group):
    table = table.copy()
    table["material_key"] = table.trial_id.str.rsplit(":", n=1).str[-1]
    if dataset == "SEEDIV": table["material_key"] = table.session.astype(str)+":"+table.material_key
    rng = np.random.default_rng(GROUPS[group]["material_seed"]); ranks = {}
    if dataset == "SEEDIV":
        materials = table[["material_key", "session", "original_label"]].drop_duplicates()
        if len(table) != 1080 or len(materials) != 72: raise ValueError("Incorrect SEED cohort")
        for _, part in materials.groupby(["session", "original_label"], sort=True):
            keys = sorted(part.material_key)
            if len(keys) != 6: raise ValueError("Missing native-emotion materials")
            ranks.update({keys[int(i)]: r for r, i in enumerate(rng.permutation(6))})
        table["label"] = COARSE[table.original_label.to_numpy(dtype=int)]
        table["training_class"] = table.original_label.astype(int)
    else:
        keys = sorted(table.material_key.unique())
        if len(table) != 1264 or len(keys) != 40 or table.subject_id.nunique() != 32: raise ValueError("Incorrect DEAP cohort")
        ranks = {keys[int(i)]: r for r, i in enumerate(rng.permutation(40))}
        table["training_class"] = table.label.astype(int)
        if not np.array_equal(table.label.to_numpy(), (table.original_label.to_numpy()>5).astype(int)) or table.original_label.eq(5).any():
            raise ValueError("DEAP must use corrected participant ratings")
    table["material_rank"] = table.material_key.map(ranks)
    if table.trial_id.duplicated().any(): raise ValueError("Duplicate original trials")
    return table


def deap_roles(rotation, arm):
    if rotation not in range(5) or arm not in ARMS: raise ValueError("Unknown DEAP condition")
    return {"test": [rotation], "validation": [(rotation+1) % 5],
            "train": [(rotation+2) % 5, rotation if arm == "exposed" else (rotation+3) % 5]}


def matched_deap(table, fold, rotation, group):
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(fold[a]) & set(fold[b]): raise ValueError("Participant leakage")
    results = {}
    for arm in ARMS:
        roles = deap_roles(rotation, arm)
        results[arm] = {part: np.flatnonzero(table.subject_id.isin(fold[part]) & (table.material_rank//8).isin(blocks))
                        for part, blocks in roles.items()}
    selected = {a: [] for a in ARMS}
    pools_by_arm = {a: {k: part.index.to_numpy(dtype=int).tolist() for k, part in table.iloc[results[a]["train"]].groupby(["subject_id", "label"])} for a in ARMS}
    for subject in sorted(fold["train"]):
        for label in (0, 1):
            pools = {a: pools_by_arm[a].get((subject, label), []) for a in ARMS}
            n = min(len(p) for p in pools.values())
            for a in ARMS:
                # Material/participant metadata only; no EEG features or held-out outcomes.
                ranked = sorted(pools[a], key=lambda i: digest([group, rotation, fold["fold"], subject, label, table.at[i, "trial_id"]]))
                selected[a].extend(ranked[:n])
    for arm in ARMS:
        results[arm]["train"] = np.asarray(sorted(selected[arm]), dtype=int)
        for part, idx in results[arm].items():
            if set(table.iloc[idx].label) != {0, 1}: raise ValueError(f"Infeasible class coverage:{group}/{rotation}/{fold['fold']}/{arm}/{part}")
        training = table.iloc[results[arm]["train"]]
        if set(training.subject_id) != set(fold["train"]): raise ValueError("Training participant lost during matching")
        keys = {p: set(table.iloc[idx].material_key) for p, idx in results[arm].items()}
        if keys["validation"] & (keys["train"] | keys["test"]): raise ValueError("Validation video leakage")
        if arm == "unexposed" and keys["train"] & keys["test"]: raise ValueError("Unseen video leaked")
        if arm == "exposed" and not keys["test"].issubset(keys["train"]): raise ValueError("Exposed video absent after matching")
    for part in ("validation", "test"): np.testing.assert_array_equal(results[ARMS[0]][part], results[ARMS[1]][part])
    counts = [table.iloc[results[a]["train"]].groupby(["subject_id", "label"]).size() for a in ARMS]
    pd.testing.assert_series_equal(*counts)
    return results


def indices(table, fold, session, rotation, arm, dataset, group):
    return seed_masks(table, fold, session, rotation, arm) if dataset == "SEEDIV" else matched_deap(table, fold, rotation, group)[arm]


def paired_batches(table, train, seed, classes, updates=600):
    rows = table.iloc[train].copy(); rows["position"] = train
    rows = rows.sort_values(["subject_id", "training_class", "trial_id"])
    rows["rank"] = rows.groupby(["subject_id", "training_class"]).cumcount()
    keys = {r.position: f"{r.subject_id}:{r.training_class}:{r.rank}" for r in rows.itertuples()}
    pools = [rows.loc[rows.label == c, "position"].to_numpy() for c in range(classes)]
    if any(len(p) == 0 for p in pools): raise ValueError("Missing source class")
    rng = np.random.default_rng(seed)
    batches = np.stack([np.concatenate([rng.choice(p, 60//classes, replace=True) for p in pools]) for _ in range(updates)])
    return batches, digest([[keys[int(i)] for i in row] for row in batches])


def new_model(name, dataset):
    model = TemporalControl(name, "coarse3")
    if dataset == "DEAP": model.head = nn.Linear(model.head.in_features, 2)
    return model


def metrics(table, idx, p, dataset):
    return common_metrics(table.iloc[idx].original_label.to_numpy(dtype=int), p, "coarse3") if dataset == "SEEDIV" else {"binary": metric(table.iloc[idx].label.to_numpy(dtype=int), p)}


def key(value, dataset):
    value = value["coarse3" if dataset == "SEEDIV" else "binary"]
    return value["balanced_accuracy"], -value["balanced_log_loss"]


def prepare_deap(common_cache, cache):
    if cache.exists(): raise FileExistsError("Existing DEAP temporal cache")
    manifest, common = load_prepared(common_cache, verify_files=False)
    lineage = [r for r in json.loads((common_cache/"lineage.json").read_text()) if r["dataset"] == "DEAP"]
    sources = {r["source"]: r["source_sha256"] for r in lineage}
    for name, checksum in sources.items():
        if sha256(Path(name)) != checksum: raise ValueError("Changed raw DEAP source")
    rating = common["source_audit"]["deap_label_recovery"]
    if sha256(Path(rating["file"])) != rating["sha256"]: raise ValueError("Changed rating spreadsheet")
    sequences = []; rows = []
    for r in sorted(lineage, key=lambda r: r["trial_id"]):
        path = common_cache/r["cache_file"]
        if sha256(path) != r["cache_sha256"]: raise ValueError("Changed DEAP waveform")
        windows = np.load(path, allow_pickle=False)
        if windows.shape != (15, 14, 512): raise ValueError("Expected baseline-excluded60s trial")
        declared = manifest[manifest.trial_id == r["trial_id"]].sort_values("start_sample")
        if len(declared) != 15 or not np.array_equal(declared.start_sample.to_numpy(), np.arange(15)*512): raise ValueError("Changed window order")
        if not declared.label.eq(r["label"]).all(): raise ValueError("Changed cached binary label")
        # CSV parsing can round a spreadsheet-derived continuous rating by one ulp.
        np.testing.assert_allclose(declared.original_label.to_numpy(), r["original_label"], atol=1e-12, rtol=0)
        sequences.append(sequence_features(windows[:10]))
        rows.append({k: r[k] for k in ("trial_id", "subject_id", "original_label", "label", "windows")})
    x = np.stack(sequences); table = pd.DataFrame(rows); table["session"] = 1
    if x.shape != (1264, 10, 56): raise ValueError("Incorrect DEAP observation population")
    cache.mkdir(parents=True); np.save(cache/"sequences.npy", x, allow_pickle=False); table.to_csv(cache/"trials.csv", index=False)
    info = {"dataset": "DEAP", "common_cache_fingerprint": common["cache_fingerprint"], "shape": list(x.shape),
            "sequences_sha256": sha256(cache/"sequences.npy"), "trials_sha256": sha256(cache/"trials.csv"),
            "source_files": [{"file": Path(n).name, "sha256": h} for n, h in sources.items()], "rating_metadata_sha256": rating["sha256"],
            "source_waveforms_checked": 1264, "source_files_checked": 32, "input": plan()["observation"],
            "first_party_signal_authentication": "Outstanding; these source hashes bind the previously audited Kaggle mirror, not independently authenticated first-party signals"}
    info["fingerprint"] = digest(info); write_json(cache/"prepared.json", info)
    return info


def load_deap(cache):
    info = json.loads((cache/"prepared.json").read_text()); check = dict(info); fingerprint = check.pop("fingerprint")
    if digest(check) != fingerprint: raise ValueError("Changed DEAP cache metadata")
    for filename in ("sequences.npy", "trials.csv"):
        k = "sequences_sha256" if filename.endswith("npy") else "trials_sha256"
        if sha256(cache/filename) != info[k]: raise ValueError("Changed DEAP features")
    return np.load(cache/"sequences.npy", allow_pickle=False), pd.read_csv(cache/"trials.csv"), info


def features(dataset, cache, reve_run=None):
    x, table, info = load(cache) if dataset == "SEEDIV" else load_deap(cache)
    values = {"bandpower": x.mean(1)}
    if dataset == "SEEDIV":
        frozen, other, _ = load_features(reve_run)
        if not table.equals(other) or not json.loads((reve_run/"verification.json").read_text())["passed"]: raise ValueError("Unverified frozen features")
        values.update(duration=table.windows.to_numpy()[:, None].astype(float), **frozen)
    if not np.isfinite(x).all(): raise ValueError("Nonfinite temporal input")
    return x, table, info, values


def cells(table, dataset, group):
    fs = folds(table.subject_id.tolist(), GROUPS[group]["participant_seed"], 3 if dataset == "SEEDIV" else 4)
    return fs, [(s, r, f) for s in ((1, 2, 3) if dataset == "SEEDIV" else (1,))
                for r in range(3 if dataset == "SEEDIV" else 5) for f in fs]


def feasibility(original, dataset):
    records = []
    for group in GROUPS:
        table = annotate(original, dataset, group); _, conditions = cells(table, dataset, group); tested = []
        for session, rotation, fold in conditions:
            paired = [indices(table, fold, session, rotation, a, dataset, group) for a in ARMS]
            for p in ("validation", "test"): np.testing.assert_array_equal(paired[0][p], paired[1][p])
            signatures = [paired_batches(table, p["train"], 42+1000*fold["fold"]+10000*rotation, 3 if dataset == "SEEDIV" else 2, updates=2)[1] for p in paired]
            if signatures[0] != signatures[1]: raise ValueError("Unpaired canonical sampling")
            tested.extend(table.iloc[paired[0]["test"]].trial_id)
            records.append({"dataset": dataset, "group": group, "session": session, "rotation": rotation, "fold": fold["fold"],
                            "train": len(paired[0]["train"]), "validation": len(paired[0]["validation"]), "test": len(paired[0]["test"]),
                            "train_class_counts": table.iloc[paired[0]["train"]].label.value_counts().sort_index().to_dict(),
                            "validation_class_counts": table.iloc[paired[0]["validation"]].label.value_counts().sort_index().to_dict(),
                            "test_class_counts": table.iloc[paired[0]["test"]].label.value_counts().sort_index().to_dict()})
        if len(tested) != len(table) or len(set(tested)) != len(table) or set(tested) != set(table.trial_id): raise ValueError("Incomplete once-per-trial OOF")
    return records


def configuration(dataset, info, reve_run, plan_path, device, previous):
    config = {"dataset": dataset, "input_fingerprint": info["fingerprint"], "plan_sha256": sha256(plan_path),
              "source_hashes": {s: sha256(REPO/s) for s in SOURCE_FILES}, "torch_version": torch.__version__,
              "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else "CPU"}
    if dataset == "SEEDIV":
        config["reve_metadata_sha256"] = sha256(reve_run/"features.json"); config["reve_verification_sha256"] = sha256(reve_run/"verification.json")
        names = ["config.json", "verification.json"]+[f"predictions_{m}_{a}.csv" for m in (*NEURAL, "bandpower", "duration", "reve_pretrained", "reve_random42") for a in ARMS]
        config["previous_evidence_sha256"] = {n: sha256(previous/n) for n in names}
        if not json.loads((previous/"verification.json").read_text())["passed"]: raise ValueError("Previous study not verified")
        old_config = json.loads((previous/"config.json").read_text())
        if any(sha256(REPO/name) != checksum for name, checksum in old_config["source_hashes"].items()): raise ValueError("Previously frozen source changed")
    return config


def frame(table, idx, p, name, arm, session, rotation, fold, group):
    rows = table.iloc[idx][["trial_id", "subject_id", "session", "original_label", "label", "material_key", "material_rank"]].copy()
    for k, v in {"model": name, "arm": arm, "source_session": session, "material_rotation": rotation, "fold": fold, "group": group, "seed": 42}.items(): rows[k] = v
    for c in range(p.shape[1]): rows[f"p_{c}"] = p[:, c]
    return rows


def probabilities(rows):
    return rows[[c for c in rows if c.startswith("p_")]].to_numpy(dtype=float)


def fit_neural(x, table, dataset, group, session, rotation, fold, arm, name, output, device):
    identifier = f"{name}_{arm}_group{group}_session{session}_rotation{rotation}_fold{fold['fold']}"
    folder = output/"models"/identifier; folder.mkdir(parents=True, exist_ok=True)
    idx = indices(table, fold, session, rotation, arm, dataset, group); init = 42+1000*fold["fold"]+10000*rotation
    batches, signature = paired_batches(table, idx["train"], init, 3 if dataset == "SEEDIV" else 2)
    if (folder/"metrics.json").exists():
        record = json.loads((folder/"metrics.json").read_text())
        if record["config_sha256"] != sha256(output/"config.json"): raise ValueError("Changed resumed experiment")
        return record, pd.read_csv(folder/"predictions.csv")
    scaler = StandardScaler().fit(x[idx["train"]].reshape(-1, 56))
    data = torch.from_numpy(scaler.transform(x.reshape(-1, 56)).reshape(x.shape).astype(np.float32)).to(device)
    target = torch.from_numpy(table.label.to_numpy(dtype=np.int64)).to(device)
    seed_everything(init); model = new_model(name, dataset).to(device); initial = state_digest(model.state_dict())
    seed_everything(init+1000000); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best, state, selected = None, None, None; history = []
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        for step, batch in enumerate(batches, 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(data[batch]), target[batch]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            if step % 25 == 0:
                val = metrics(table, idx["validation"], infer(model, data[idx["validation"]]), dataset)
                history.append({"step": step, "training_loss": float(loss.item()), "validation": val})
                if best is None or key(val, dataset) > best: best, state, selected = key(val, dataset), copy.deepcopy(model.state_dict()), step
        model.load_state_dict(state); training = metrics(table, idx["train"], infer(model, data[idx["train"]]), dataset)
        p = infer(model, data[idx["test"]]); tested = metrics(table, idx["test"], p, dataset)
    rows = frame(table, idx["test"], p, name, arm, session, rotation, fold["fold"], group)
    rows.to_csv(folder/"predictions.csv", index=False); write_json(folder/"history.json", history)
    torch.save({"state": {k: v.cpu() for k, v in state.items()}, "mean": scaler.mean_, "scale": scaler.scale_, "selected_step": selected}, folder/"selected.pt")
    record = {"id": identifier, "model": name, "arm": arm, "group": group, "session": session, "rotation": rotation, "fold": fold["fold"],
              "training": training, "test": tested, "selected_step": selected, "parameters": sum(p.numel() for p in model.parameters()),
              "initial_state_sha256": initial, "canonical_draw_signature": signature, "batch_digest": digest(batches.tolist()),
              "elapsed_seconds": time.perf_counter()-start, "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
              "config_sha256": sha256(output/"config.json"), "checkpoint_sha256": sha256(folder/"selected.pt"),
              "history_sha256": sha256(folder/"history.json"), "predictions_sha256": sha256(folder/"predictions.csv")}
    write_json(folder/"metrics.json", record); del model, optimizer, state, data, target
    return record, rows


def material_prior(table, train, tested):
    source = table.iloc[train]; global_p = (source.label.sum()+1)/(len(source)+2)
    per_video = source.groupby("material_key").label.agg(["sum", "count"])
    lookup = ((per_video["sum"]+1)/(per_video["count"]+2)).to_dict()
    pos = np.array([lookup.get(k, global_p) for k in table.iloc[tested].material_key])
    return np.stack([1-pos, pos], axis=1)


def fit_linears(values, table, dataset, group, output):
    _, conditions = cells(table, dataset, group)
    for name in (*values, *(("material_prior",) if dataset == "DEAP" else ())):
        parts = {a: [] for a in ARMS}; records = []
        for session, rotation, fold in conditions:
            for arm in ARMS:
                idx = indices(table, fold, session, rotation, arm, dataset, group)
                record = {"model": name, "group": group, "arm": arm, "session": session, "rotation": rotation, "fold": fold["fold"]}
                if name == "material_prior":
                    p = material_prior(table, idx["train"], idx["test"]); record["selection"] = "No hyperparameters; training labels only"
                else:
                    scaler = StandardScaler().fit(values[name][idx["train"]]); inputs = scaler.transform(values[name]); candidates = []; estimators = []
                    for c in C_VALUES:
                        est = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, tol=1e-6, random_state=42)
                        est.fit(inputs[idx["train"]], table.iloc[idx["train"]].label); estimators.append(est)
                        candidates.append({"C": c, "validation": metrics(table, idx["validation"], est.predict_proba(inputs[idx["validation"]]), dataset)})
                    chosen = max(range(4), key=lambda j: key(candidates[j]["validation"], dataset)); est = estimators[chosen]
                    p = est.predict_proba(inputs[idx["test"]]); record.update(selected_C=C_VALUES[chosen], candidates=candidates,
                        coefficients=est.coef_.tolist(), intercept=est.intercept_.tolist(), scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist())
                record["test"] = metrics(table, idx["test"], p, dataset); records.append(record)
                parts[arm].append(frame(table, idx["test"], p, name, arm, session, rotation, fold["fold"], group))
        # Per-session/per-rotation coefficient chunks remain below the public evidence size cap.
        for session in sorted(table.session.unique()):
            for rotation in range(3 if dataset == "SEEDIV" else 5):
                write_json(output/f"linear_{name}_group{group}_session{session}_rotation{rotation}.json", [r for r in records if r["session"] == session and r["rotation"] == rotation])
        for arm in ARMS: pd.concat(parts[arm], ignore_index=True).to_csv(output/f"predictions_{name}_{arm}_group{group}.csv", index=False)
        print(f"Completed {dataset} group{group} {name} ({len(records)} heads)", flush=True)


def run(dataset, cache, reve_run, previous, output, plan_path, device="cuda"):
    if (output/"verification.json").exists(): raise FileExistsError("Completed experiment")
    declaration = json.loads(plan_path.read_text()); current = json.loads(json.dumps(plan()))
    if {k: v for k, v in declaration.items() if k != "created_utc"} != {k: v for k, v in current.items() if k != "created_utc"}: raise ValueError("Changed frozen declaration")
    x, original, info, values = features(dataset, cache, reve_run); output.mkdir(parents=True, exist_ok=True)
    config = configuration(dataset, info, reve_run, plan_path, device, previous)
    if (output/"config.json").exists() and json.loads((output/"config.json").read_text()) != config: raise ValueError("Changed resumed config")
    write_json(output/"config.json", config); write_json(output/"plan.json", declaration); write_json(output/"feasibility.json", feasibility(original, dataset))
    index = []; count = 0; total = 360 if dataset == "SEEDIV" else 320
    for group in GROUPS:
        table = annotate(original, dataset, group); fs, conditions = cells(table, dataset, group)
        write_json(output/f"folds_group{group}.json", fs)
        table[["material_key", "material_rank"]].drop_duplicates().sort_values("material_key").to_csv(output/f"material_assignments_group{group}.csv", index=False)
        for name in NEURAL:
            parts = {a: [] for a in ARMS}; records = []
            for session, rotation, fold in conditions:
                for arm in ARMS:
                    record, rows = fit_neural(x, table, dataset, group, session, rotation, fold, arm, name, output, device)
                    records.append(record); parts[arm].append(rows); count += 1
                    index.append({"id": record["id"], "metrics_sha256": sha256(output/"models"/record["id"]/"metrics.json")})
                    if count % 10 == 0: print(f"{dataset} neural {count}/{total}:group{group}/{name}", flush=True)
            write_json(output/f"model_metrics_{name}_group{group}.json", records)
            for arm in ARMS: pd.concat(parts[arm], ignore_index=True).to_csv(output/f"predictions_{name}_{arm}_group{group}.csv", index=False)
        fit_linears(values, table, dataset, group, output)
    write_json(output/"model_index.json", index)
    return {"completed": True, "neural_fits": total, "analysis_after_independent_verification": True}


def check_metrics(actual, expected):
    for task, values in expected.items():
        for k, v in values.items():
            if k == "balanced_log_loss":
                if abs(actual[task][k]-v) > 1e-10: raise ValueError("Log loss failed replay")
            elif actual[task][k] != v: raise ValueError("Classification metric failed replay")


def check_rows(actual, expected, tolerance):
    columns = [c for c in actual if not c.startswith("p_")]
    pd.testing.assert_frame_equal(actual[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_exact=True)
    error = float(np.max(np.abs(probabilities(actual)-probabilities(expected))))
    if error > tolerance: raise ValueError("Probabilities failed replay")
    return error


def check_coverage(rows, original):
    for _, part in rows.groupby(["model", "arm", "group", "seed"]):
        if len(part) != len(original) or part.trial_id.duplicated().any() or set(part.trial_id) != set(original.trial_id): raise ValueError("OOF coverage failed")


def verify(dataset, cache, reve_run, previous, output, device="cuda"):
    x, original, info, values = features(dataset, cache, reve_run); config = json.loads((output/"config.json").read_text())
    if configuration(dataset, info, reve_run, output/"plan.json", config["device"], previous) != config: raise ValueError("Changed config/source/input bindings")
    if json.loads((output/"feasibility.json").read_text()) != json.loads(json.dumps(feasibility(original, dataset))): raise ValueError("Changed feasibility record")
    tables = {g: annotate(original, dataset, g) for g in GROUPS}; folded = {g: cells(t, dataset, g)[0] for g, t in tables.items()}
    for g, table in tables.items():
        if json.loads((output/f"folds_group{g}.json").read_text()) != folded[g]: raise ValueError("Changed folds")
        expected = table[["material_key", "material_rank"]].drop_duplicates().sort_values("material_key").reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_csv(output/f"material_assignments_group{g}.csv"), expected, check_exact=True)
    index = json.loads((output/"model_index.json").read_text()); required = 360 if dataset == "SEEDIV" else 320
    expected_ids = {f"{name}_{arm}_group{g}_session{s}_rotation{r}_fold{f['fold']}"
                    for g, table in tables.items() for s, r, f in cells(table, dataset, g)[1] for name in NEURAL for arm in ARMS}
    if len(index) != required or {r["id"] for r in index} != expected_ids: raise ValueError("Incomplete neural experiment")
    signatures = {}; initializations = {}; collected = {}; max_neural = 0.
    for number, item in enumerate(index, 1):
        folder = output/"models"/item["id"]; record = json.loads((folder/"metrics.json").read_text())
        if sha256(folder/"metrics.json") != item["metrics_sha256"] or record["config_sha256"] != sha256(output/"config.json"): raise ValueError("Changed model record")
        for filename, k in (("selected.pt", "checkpoint_sha256"), ("history.json", "history_sha256"), ("predictions.csv", "predictions_sha256")):
            if sha256(folder/filename) != record[k]: raise ValueError("Changed fitted artifact")
        g = record["group"]; table = tables[g]; fold = folded[g][record["fold"]]
        idx = indices(table, fold, record["session"], record["rotation"], record["arm"], dataset, g)
        init = 42+1000*record["fold"]+10000*record["rotation"]
        batches, signature = paired_batches(table, idx["train"], init, 3 if dataset == "SEEDIV" else 2)
        if signature != record["canonical_draw_signature"] or digest(batches.tolist()) != record["batch_digest"]: raise ValueError("Changed sampling")
        signatures.setdefault((g, record["session"], record["rotation"], record["fold"]), set()).add(signature)
        seed_everything(init); model = new_model(record["model"], dataset).to(device)
        if state_digest(model.state_dict()) != record["initial_state_sha256"]: raise ValueError("Initialization failed replay")
        if sum(p.numel() for p in model.parameters()) != record["parameters"]: raise ValueError("Changed architecture")
        initializations.setdefault((record["model"], record["rotation"], record["fold"]), set()).add(record["initial_state_sha256"])
        checkpoint = torch.load(folder/"selected.pt", map_location="cpu", weights_only=False); model.load_state_dict(checkpoint["state"])
        scaler = StandardScaler().fit(x[idx["train"]].reshape(-1, 56))
        np.testing.assert_array_equal(scaler.mean_, checkpoint["mean"]); np.testing.assert_array_equal(scaler.scale_, checkpoint["scale"])
        data = torch.from_numpy(scaler.transform(x.reshape(-1, 56)).reshape(x.shape).astype(np.float32)).to(device)
        history = json.loads((folder/"history.json").read_text())
        if [h["step"] for h in history] != list(range(25, 601, 25)): raise ValueError("Changed available update budget")
        best = max(history, key=lambda h: key(h["validation"], dataset))
        if best["step"] != record["selected_step"] or checkpoint["selected_step"] != record["selected_step"]: raise ValueError("Incorrect checkpoint choice")
        with sdpa_kernel(SDPBackend.MATH):
            check_metrics(metrics(table, idx["validation"], infer(model, data[idx["validation"]]), dataset), best["validation"])
            check_metrics(metrics(table, idx["train"], infer(model, data[idx["train"]]), dataset), record["training"])
            p = infer(model, data[idx["test"]]); check_metrics(metrics(table, idx["test"], p, dataset), record["test"])
        saved = pd.read_csv(folder/"predictions.csv"); expected = frame(table, idx["test"], p, record["model"], record["arm"], record["session"], record["rotation"], record["fold"], g)
        max_neural = max(max_neural, check_rows(saved, expected, 1e-6))
        collected.setdefault((record["model"], record["arm"], g), []).append(saved); del model, data
        if number % 80 == 0: print(f"Replayed {dataset} neural {number}/{required}", flush=True)
    if any(len(v) != 1 for v in (*signatures.values(), *initializations.values())): raise ValueError("Unpaired streams or initial states")
    for (name, arm, g), rows in collected.items():
        actual = pd.read_csv(output/f"predictions_{name}_{arm}_group{g}.csv")
        check_rows(actual, pd.concat(rows, ignore_index=True), 1e-15); check_coverage(actual, original)
    for g in GROUPS:
        for name in NEURAL:
            saved = json.loads((output/f"model_metrics_{name}_group{g}.json").read_text())
            expected = [json.loads((output/"models"/i["id"]/"metrics.json").read_text()) for i in index if i["id"].startswith(name+"_") and f"_group{g}_" in i["id"]]
            if saved != expected: raise ValueError("Changed metric chunks")
    selected_heads = 0; candidates = 0; diagnostic_heads = 0; max_linear = 0.; max_coefficient = 0.
    for g, table in tables.items():
        for name in (*values, *(("material_prior",) if dataset == "DEAP" else ())):
            saved = {a: pd.read_csv(output/f"predictions_{name}_{a}_group{g}.csv") for a in ARMS}
            replayed = {a: [] for a in ARMS}
            for p in saved.values(): check_coverage(p, original)
            for session in sorted(table.session.unique()):
                for rotation in range(3 if dataset == "SEEDIV" else 5):
                    records = json.loads((output/f"linear_{name}_group{g}_session{session}_rotation{rotation}.json").read_text())
                    expected_cells = {(a, f["fold"]) for a in ARMS for f in folded[g]}
                    if len(records) != len(expected_cells) or {(r["arm"], r["fold"]) for r in records} != expected_cells: raise ValueError("Missing linear cell")
                    for r in records:
                        idx = indices(table, folded[g][r["fold"]], session, rotation, r["arm"], dataset, g)
                        if name == "material_prior":
                            p = material_prior(table, idx["train"], idx["test"]); diagnostic_heads += 1
                        else:
                            scaler = StandardScaler().fit(values[name][idx["train"]]); inputs = scaler.transform(values[name])
                            np.testing.assert_array_equal(scaler.mean_, r["scaler_mean"]); np.testing.assert_array_equal(scaler.scale_, r["scaler_scale"])
                            if [v["C"] for v in r["candidates"]] != list(C_VALUES): raise ValueError("Changed C budget")
                            estimators = []
                            for c in r["candidates"]:
                                est = LogisticRegression(C=c["C"], class_weight="balanced", max_iter=2000, tol=1e-6, random_state=42)
                                est.fit(inputs[idx["train"]], table.iloc[idx["train"]].label); estimators.append(est); candidates += 1
                                check_metrics(metrics(table, idx["validation"], est.predict_proba(inputs[idx["validation"]]), dataset), c["validation"])
                            chosen = max(range(4), key=lambda j: key(r["candidates"][j]["validation"], dataset)); est = estimators[chosen]
                            if C_VALUES[chosen] != r["selected_C"]: raise ValueError("Incorrect C choice")
                            w = np.asarray(r["coefficients"]); b = np.asarray(r["intercept"])
                            np.testing.assert_allclose(w, est.coef_, atol=1e-9, rtol=1e-9); np.testing.assert_allclose(b, est.intercept_, atol=1e-9, rtol=1e-9)
                            max_coefficient = max(max_coefficient, float(np.max(np.abs(w-est.coef_))), float(np.max(np.abs(b-est.intercept_))))
                            if dataset == "DEAP":
                                pos = expit((inputs[idx["test"]]@w.T+b)[:, 0]); p = np.stack([1-pos, pos], axis=1)
                            else: p = softmax(inputs[idx["test"]]@w.T+b, axis=1)
                            selected_heads += 1
                        check_metrics(metrics(table, idx["test"], p, dataset), r["test"])
                        rows = saved[r["arm"]]; rows = rows[(rows.source_session == session) & (rows.material_rotation == rotation) & (rows.fold == r["fold"])]
                        expected = frame(table, idx["test"], p, name, r["arm"], session, rotation, r["fold"], g)
                        max_linear = max(max_linear, check_rows(rows, expected, 1e-14)); replayed[r["arm"]].append(expected)
            for a in ARMS:
                # Stored rows have condition-major order; complete replay checks original IDs once.
                if len(pd.concat(replayed[a])) != len(original): raise ValueError("Incomplete linear replay")
    required_heads = 720 if dataset == "SEEDIV" else 160
    if selected_heads != required_heads or candidates != 4*required_heads or diagnostic_heads != (160 if dataset == "DEAP" else 0): raise ValueError("Wrong completed head count")
    result = {"passed": True, "dataset": dataset, "neural_fits_replayed": required, "selected_linear_heads": selected_heads,
              "candidates_independently_refitted": candidates, "diagnostic_heads_replayed": diagnostic_heads,
              "maximum_neural_probability_error": max_neural, "maximum_linear_probability_error": max_linear,
              "maximum_coefficient_refit_error": max_coefficient, "groupings": list(GROUPS), "initializations": [42],
              "config_sha256": sha256(output/"config.json"), "scope": "Input/source bindings, exact folds/material matching/OOF coverage, paired streams/initial states, all selected neural checkpoint train/validation/test replay and24-checkpoint selection, every linear candidate refit and coefficients/probabilities. Frozen embeddings reused from earlier verified extraction; no new first-party data authentication."}
    write_json(output/"verification.json", result)
    return result


def read_predictions(output, previous, dataset):
    names = (*NEURAL, "bandpower", "duration", "reve_pretrained", "reve_random42") if dataset == "SEEDIV" else (*NEURAL, "bandpower", "material_prior")
    frames = {}
    for name in names:
        frames[name] = {}
        for arm in ARMS:
            parts = [pd.read_csv(output/f"predictions_{name}_{arm}_group{g}.csv") for g in GROUPS]
            if dataset == "SEEDIV":
                old = pd.read_csv(previous/f"predictions_{name}_{arm}.csv"); old = old[old.seed == 42].copy()
                old["label"] = COARSE[old.original_label.to_numpy(dtype=int)]; old["group"] = 0
                parts.insert(0, old)
            frames[name][arm] = pd.concat(parts, ignore_index=True)
    return frames


def weights(table, dataset, draws=10000, seed=20261006):
    subjects = sorted(table.subject_id.unique()); materials = sorted(table.material_key.unique())
    rng = np.random.default_rng(seed)
    sw = rng.multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=draws).astype(float)
    if dataset == "DEAP": mw = rng.multinomial(40, np.full(40, 1/40), size=draws).astype(float)
    else:
        metadata = table[["material_key", "session", "original_label"]].drop_duplicates().set_index("material_key").loc[materials]
        mw = np.zeros((draws, len(materials)), dtype=float)
        for _, part in metadata.groupby(["session", "original_label"], sort=True):
            positions = [materials.index(k) for k in part.index]
            mw[:, positions] = rng.multinomial(6, np.full(6, 1/6), size=draws)
    return subjects, materials, sw, mw


def bootstrap(rows, dataset, task, subjects, materials, sw, mw):
    rows = rows.copy(); p = probabilities(rows)
    if dataset == "SEEDIV" and task == "binary":
        pos = p[:, 2]/np.maximum(p[:, 1:].sum(1), 1e-12)
        rows["correct"] = np.stack([1-pos, pos], axis=1).argmax(1) == (rows.original_label.to_numpy() == 3)
        rows["target"] = (rows.original_label == 3).astype(int); rows = rows[rows.original_label != 0]
    else:
        rows["correct"] = p.argmax(1) == rows.label.to_numpy(dtype=int); rows["target"] = rows.label
    if rows.groupby(["subject_id", "material_key"]).target.nunique().gt(1).any(): raise ValueError("Labels changed across repeated predictions")
    cells = rows.groupby(["subject_id", "material_key"]).agg(correct=("correct", "mean"), target=("target", "first"))
    points = []; values = []
    for c in range(3 if task == "coarse3" else 2):
        included = cells.target.eq(c).astype(float).unstack().reindex(index=subjects, columns=materials).fillna(0).to_numpy()
        correct = (cells.correct*cells.target.eq(c)).unstack().reindex(index=subjects, columns=materials).fillna(0).to_numpy()
        denominator = ((sw@included)*mw).sum(1)
        if (denominator <= 0).any(): raise ValueError("Bootstrap draw lost class coverage")
        values.append(((sw@correct)*mw).sum(1)/denominator); points.append(correct.sum()/included.sum())
    return float(np.mean(points)), np.mean(values, axis=0)


def analyze(dataset, output, previous, write=True):
    if not json.loads((output/"verification.json").read_text())["passed"]: raise ValueError("Numerical verification required before analysis")
    frames = read_predictions(output, previous, dataset); table = next(iter(frames.values()))["exposed"].drop_duplicates("trial_id")
    subjects, materials, sw, mw = weights(table, dataset); fixed = np.ones_like(mw)
    tasks = ("coarse3", "binary") if dataset == "SEEDIV" else ("binary",)
    result = {"development_only": True, "research_question_change_approved": False, "dataset": dataset, "models": {}, "contrasts": []}
    scopes = ["combined", *[str(g) for g in sorted(table_group for table_group in next(iter(frames.values()))["exposed"].group.unique())]]
    distributions = {}
    for scope in scopes:
        distributions[scope] = {}; report = {}
        for name, arms in frames.items():
            report[name] = {}; distributions[scope][name] = {}
            for arm, all_rows in arms.items():
                rows = all_rows if scope == "combined" else all_rows[all_rows.group == int(scope)]
                check_coverage(rows, table); item = {}; distributions[scope][name][arm] = {}
                for task in tasks:
                    point, crossed = bootstrap(rows, dataset, task, subjects, materials, sw, mw)
                    _, participant = bootstrap(rows, dataset, task, subjects, materials, sw, fixed)
                    per_group = {str(g): (common_metrics(part.original_label.to_numpy(dtype=int), probabilities(part), "coarse3") if dataset == "SEEDIV" else {"binary": metric(part.label.to_numpy(dtype=int), probabilities(part))}) for g, part in rows.groupby("group")}
                    if abs(point-np.mean([v[task]["balanced_accuracy"] for v in per_group.values()])) > 1e-12: raise ValueError("Bootstrap point and per-group BA disagree")
                    item[task] = {"mean_BA": point, "crossed_percentile_95": np.quantile(crossed, [.025, .975]).tolist(), "participant_percentile_95": np.quantile(participant, [.025, .975]).tolist()}
                    distributions[scope][name][arm][task] = (crossed, participant)
                item["per_group_metrics"] = per_group; report[name][arm] = item
        result["models"][scope] = report
        pairs = [(name, "unexposed", name, "exposed") for name in frames]
        pairs += [("transformer", a, "mean_mlp", a) for a in ARMS]
        if dataset == "SEEDIV": pairs += [("reve_pretrained", a, "reve_random42", a) for a in ARMS]
        for task in tasks:
            for a, aa, b, ba in pairs:
                ca, pa = distributions[scope][a][aa][task]; cb, pb = distributions[scope][b][ba][task]
                result["contrasts"].append({"scope": scope, "model_a": a, "arm_a": aa, "model_b": b, "arm_b": ba, "task": task,
                    "BA_difference": report[a][aa][task]["mean_BA"]-report[b][ba][task]["mean_BA"],
                    "crossed_percentile_95": np.quantile(ca-cb, [.025, .975]).tolist(), "participant_percentile_95": np.quantile(pa-pb, [.025, .975]).tolist()})
    result["bootstrap"] = {"draws": 10000, "seed": 20261006, "subject_weight_digest": digest(sw.tolist()), "material_weight_digest": digest(mw.tolist()),
                           "scope": "Paired subject and crossed subject/material percentiles with observed-cell class denominators; average correctness across groupings per person/video. Reused people/videos and fixed fitted models/groupings, not independent replications. Unadjusted exploratory intervals."}
    result["scope"] = "Both arms same test/validation participants/trials and matched training participant/class counts; source-only fitting. Exposure changes training clip content/order/difficulty, not a causal video-identity intervention. Different individual vs stimulus labels across corpora; no pooled/unseen-corpus transfer claim. One new optimizer initialization; no test-based grouping selection."
    if write: write_json(output/"comparison.json", result)
    return result
