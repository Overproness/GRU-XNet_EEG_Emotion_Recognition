"""Separately activated FACED development-only context and pairing diagnosis.

No work occurs at import. The runner opens existing declared caches only after
input bindings and a source-fixture checkpoint replay gate pass. It never opens
original recordings or original ratings, and fits no neural network.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .faced_classical_contract import ALPHAS, context_features, fit_source_ridge, predict_ridge, participant_film_mae
from .faced_eegnet_bridge import authoritative_bundle, array_sha256, guard_cached_identity
from .faced_cached_targets import cached_original_target
from .faced_classical_runner import estimate_priors, prior_predictions

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
OLD_QUALITY = BASE / "faced_inference_quality_2026-10-11"
OLD_ARTIFACT = BASE / "faced_artifact_control_2026-10-11"
OLD_NEURAL = BASE / "faced_eegnet_diagnostic_v2_2026-10-11"
OLD_PRIVATE = REPO.parent / "publication_runs/faced_artifact_control_2026-10-11"
PROGRAMME = BASE / "faced_development_protocol_2026-10-11/plan.json"
MATERIAL_SALT = "FACED_DIAG_CONTEXT_2026-10-12"
OLD_CONTEXT_ALPHAS = {0: .1, 1: 100., 2: 100.}
SWAP_SEEDS = tuple(2026101200 + i for i in range(32))
FIT_CAP = 612
TIME_CAP = 1800
ARMS = ("validation_unseen", "validation_familiar")
SCHEMES = ("person_same_film_acquisition", "film_same_person_coarse_valence")
PRIOR_NAMES = ("global_median", "coarse_valence_median", "category_median")
COMPARATOR_PRIORITY = ("category_only", "full_context") + PRIOR_NAMES


def required_binding_paths():
    names = [PROGRAMME, BASE / "faced_backup_2026-10-10/participant_reservation.json",
             BASE / "faced_codebooks_2026-10-10/audit.json", OLD_ARTIFACT / "source_records.json",
             OLD_ARTIFACT / "validation_records.json", OLD_QUALITY / "signal_audit.json",
             OLD_QUALITY / "predictions.csv", OLD_QUALITY / "comparison.json",
             OLD_NEURAL / "checkpoint_predictions.csv", OLD_NEURAL / "ensemble_predictions.csv",
             OLD_NEURAL / "selection.json"]
    names += [REPO / x for x in ("gruxnet/faced_development_diagnosis.py", "gruxnet/faced_classical_contract.py",
              "gruxnet/faced_eegnet_bridge.py", "gruxnet/faced_cached_targets.py", "gruxnet/faced_classical_runner.py",
              "gruxnet/faced_eegnet_runner.py")]
    return [p.relative_to(REPO).as_posix() for p in names]


def load(path):
    return json.loads(Path(path).read_bytes())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata_bundle():
    programme = load(PROGRAMME)
    reservation = load(BASE / "faced_backup_2026-10-10/participant_reservation.json")
    catalogue = {r["clip"]: r for r in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
    records = load(OLD_ARTIFACT / "source_records.json")["records"] + load(OLD_ARTIFACT / "validation_records.json")["records"]
    decisions = {(r["subject"], r["vid"]): r for r in load(OLD_QUALITY / "signal_audit.json")["trial_decisions"]}
    record_map, metadata = authoritative_bundle(programme, reservation, catalogue, records, decisions)
    return programme, record_map, metadata, decisions


def crossed_source_folds(source_pairs, metadata, person_folds, *, salt=MATERIAL_SALT):
    """Twenty source-only folds; held person AND held title absent from fitting."""
    source = list(source_pairs)
    if len(source) != len(set(source)) or any(p not in metadata or metadata[p]["role"] != "source" for p in source):
        raise PermissionError("Duplicate, protected or nonsource CV identity")
    people = {p[0] for p in source}
    clips = {p[1] for p in source}
    if len(people) != 70 or len(clips) != 8 or set(source) != {(s, v) for s in people for v in clips}:
        raise PermissionError("Source CV requires the fixed 70 by 8 rectangle")
    flattened = [s for fold in person_folds for s in fold]
    if len(person_folds) != 5 or any(len(fold) != 14 for fold in person_folds) or len(flattened) != len(set(flattened)) or set(flattened) != people:
        raise PermissionError("Changed source participant folds")
    families = {v: {metadata[p]["family"] for p in source if p[1] == v} for v in clips}
    if any(len(x) != 1 for x in families.values()) or len(set(next(iter(x)) for x in families.values())) != 8:
        raise PermissionError("Source material folds require eight distinct title families")
    ordered = sorted(clips, key=lambda v: (hashlib.sha256(f"{salt}/{v}".encode()).hexdigest(), v))
    material_folds = [ordered[i::4] for i in range(4)]
    result = []
    for person_fold, held_people in enumerate(person_folds):
        for material_fold, held_clips in enumerate(material_folds):
            training = [p for p in source if p[0] not in held_people and p[1] not in held_clips]
            held = [p for p in source if p[0] in held_people and p[1] in held_clips]
            if len(training) != 336 or len(held) != 28 or {p[0] for p in training} & {p[0] for p in held} or {metadata[p]["family"] for p in training} & {metadata[p]["family"] for p in held}:
                raise PermissionError("Crossed source-fold exclusion failed")
            result.append({"person_fold": person_fold, "material_fold": material_fold,
                           "held_clips": held_clips, "training": training, "held": held})
    if Counter(p for f in result for p in f["held"]) != Counter(source):
        raise PermissionError("Crossed held rectangle not covered exactly once")
    return result


def balanced_decomposition(pairs, targets, metadata):
    """Descriptive person/title/additive-residual sums of squares, not noise."""
    pairs = list(pairs)
    if any(p not in metadata or p not in targets for p in pairs):
        raise PermissionError("Unknown target identity")
    people = sorted({p[0] for p in pairs})
    titles = sorted({metadata[p]["family"] for p in pairs})
    cells = defaultdict(list)
    for p in pairs:
        if p not in metadata or p not in targets:
            raise PermissionError("Unknown target identity")
        cells[p[0], metadata[p]["family"]].append(p)
    if len(pairs) != len(set(pairs)) or len(cells) != len(people) * len(titles) or any(len(v) != 1 for v in cells.values()):
        raise PermissionError("Decomposition requires one rating per complete person/title cell")
    y = np.array([[targets[cells[s, f][0]] for f in titles] for s in people], dtype=float)
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 7)):
        raise ValueError("Invalid cached original-scale targets")
    mean = y.mean()
    person = y.mean(axis=1) - mean
    title = y.mean(axis=0) - mean
    residual = y - mean - person[:, None] - title[None, :]
    ss = {"person": float(len(titles) * np.square(person).sum()),
          "title": float(len(people) * np.square(title).sum()),
          "additive_residual": float(np.square(residual).sum())}
    total = float(np.square(y - mean).sum())
    if abs(sum(ss.values()) - total) > 1e-8 * max(1., total):
        raise ValueError("Balanced decomposition identity failed")
    return {"participants": len(people), "title_families": len(titles), "trials": len(pairs),
            "grand_mean": float(mean), "person_means": dict(zip(people, map(float, y.mean(axis=1)))),
            "title_means": dict(zip(titles, map(float, y.mean(axis=0)))), "total_ss": total, "sum_squares": ss,
            "fractions": {k: (v / total if total else None) for k, v in ss.items()},
            "interpretation": "Descriptive additive decomposition. Residual includes interaction and unexplained variation; it is not measurement noise or rating reliability."}


def donor_assignments(pairs, metadata, decisions, scheme, seed):
    """Validation-only strict derangements; singleton and unavailable unchanged."""
    pairs = list(pairs)
    if scheme not in SCHEMES or len(pairs) != len(set(pairs)):
        raise PermissionError("Unknown scheme or duplicate recipient")
    groups = defaultdict(list)
    result = {}
    for p in pairs:
        if p not in metadata or p not in decisions or metadata[p]["role"] != "validation":
            raise PermissionError("Pairing recipient outside declared validation")
        if type(decisions[p]["eeg_available"]) is not bool:
            raise ValueError("Nonboolean availability")
        result[p] = {"donor": p, "eligible": False, "stratum_size": 0,
                     "reason": "unavailable_eeg" if not decisions[p]["eeg_available"] else "singleton_stratum"}
        if decisions[p]["eeg_available"]:
            group = (p[1], metadata[p]["acquisition_group"]) if scheme == SCHEMES[0] else (p[0], metadata[p]["assigned_valence"])
            groups[group].append(p)
    rng = np.random.default_rng(seed)
    for group in sorted(groups, key=lambda x: tuple(str(k) for k in x)):
        members = sorted(groups[group])
        if len(members) == 1:
            result[members[0]]["stratum_size"] = 1
            continue
        indices = np.arange(len(members))
        for _ in range(10000):
            permutation = rng.permutation(len(members))
            if np.all(permutation != indices):
                break
        else:
            raise RuntimeError("Could not construct fixed strict derangement")
        for i, p in enumerate(members):
            donor = members[int(permutation[i])]
            result[p] = {"donor": donor, "eligible": True, "stratum_size": len(members), "reason": "switched"}
    if set(result) != set(pairs) or any(d["donor"] not in set(pairs) for d in result.values()):
        raise PermissionError("Donor left the recipient arm")
    return result


def mixed_joint_features(pairs, donors, context, spectral, decisions):
    """Retain RECIPIENT context while using only eligible DONOR EEG."""
    result = {}
    for p in pairs:
        if decisions[p]["eeg_available"]:
            donor = donors[p]["donor"]
            if donor not in decisions or not decisions[donor]["eeg_available"]:
                raise PermissionError("Unavailable EEG donor")
            result[p] = np.r_[context[p], spectral[donor]]
    return result


def residual_swap_predictions(pairs, donors, offsets, corrections, decisions):
    """Corrections are averages of two seed RAW outputs minus donor offset."""
    result = {}
    for p in pairs:
        value = offsets[p]
        if decisions[p]["eeg_available"]:
            donor = donors[p]["donor"]
            if not decisions[donor]["eeg_available"] or donor not in corrections:
                raise PermissionError("Missing/failed neural correction donor")
            value += corrections[donor]
        result[p] = float(np.clip(value, 0, 7))
    return result


def average_raw_correction(raw_values, original_offset):
    values = np.asarray(raw_values, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or not np.isfinite(original_offset):
        raise ValueError("Require two finite unclipped neural seed predictions")
    return float(np.mean(values - original_offset))


def loss_metrics(rows):
    if not rows:
        return None
    return {"participant_title_mae": participant_film_mae(rows),
            "trial_mae": float(np.mean([abs(r["target"] - r["prediction"]) for r in rows])),
            "trials": len(rows), "participants": len({r["subject"] for r in rows}),
            "title_families": len({r["family"] for r in rows})}


def repeat_loss_summary(original_rows, repeated_rows, eligible):
    """Average losses across repeats, never average predictions before loss."""
    identity = lambda r: (r["subject"], int(r["vid"]))
    originals = {identity(r): r for r in original_rows}
    if len(originals) != len(original_rows) or set(eligible) != set(originals):
        raise PermissionError("Changed comparison recipient rectangle")
    original_full = loss_metrics(original_rows)
    original_eligible = loss_metrics([r for r in original_rows if eligible[identity(r)]])
    repeats = []
    for rows in repeated_rows:
        keyed = {identity(r): r for r in rows}
        if set(keyed) != set(originals) or len(keyed) != len(rows):
            raise PermissionError("Changed repeated recipient rectangle")
        if any(keyed[p]["target"] != originals[p]["target"] or keyed[p]["family"] != originals[p]["family"] for p in originals):
            raise PermissionError("Recipient target/title changed in swap")
        all_metrics = loss_metrics(rows)
        subset = loss_metrics([r for r in rows if eligible[identity(r)]])
        repeats.append({"full": all_metrics, "eligible": subset,
                        "full_delta": all_metrics["participant_title_mae"] - original_full["participant_title_mae"],
                        "eligible_delta": None if subset is None else subset["participant_title_mae"] - original_eligible["participant_title_mae"]})
    return {"original_full": original_full, "original_eligible": original_eligible,
            "repeats": repeats, "mean_full_delta": float(np.mean([r["full_delta"] for r in repeats])),
            "mean_eligible_delta": None if original_eligible is None else float(np.mean([r["eligible_delta"] for r in repeats])),
            "eligible_trials": sum(eligible.values()), "unperturbed_trials": len(eligible) - sum(eligible.values()),
            "loss_aggregation": "Each repeat is scored independently, then its losses are averaged. No averaged-prediction scoring or formal null test.",
            "subset_limitation": "Eligible-only coverage may be nonrectangular; no complete-crossed bootstrap or independent-confirmation claim."}


def crossed_loss_interval(original_rows, repeated_rows, *, draws=4000, seed=425401):
    """Crossed resampling of mean-repeat loss deltas on the FULL rectangle."""
    originals = {(r["subject"], int(r["vid"])): r for r in original_rows}
    deltas = defaultdict(list)
    for rows in repeated_rows:
        if {(r["subject"], int(r["vid"])) for r in rows} != set(originals):
            raise PermissionError("Changed full-panel loss rectangle")
        for r in rows:
            original = originals[r["subject"], int(r["vid"])]
            delta = abs(r["target"] - r["prediction"]) - abs(original["target"] - original["prediction"])
            deltas[r["subject"], r["family"]].append(delta)
    people = sorted({p for p, _ in deltas})
    families = sorted({f for _, f in deltas})
    if len(deltas) != len(people) * len(families):
        raise PermissionError("Crossed interval requires the complete person/title rectangle")
    matrix = np.asarray([[np.mean(deltas[p, f]) for f in families] for p in people])
    rng = np.random.default_rng(seed)
    people_draw = rng.integers(len(people), size=(draws, len(people)))
    family_draw = rng.integers(len(families), size=(draws, len(families)))
    distribution = matrix[people_draw[:, :, None], family_draw[:, None, :]].mean(axis=(1, 2))
    return {"mean_loss_delta": float(matrix.mean()), "draws": draws, "seed": seed,
            "descriptive_95_percent_interval": np.quantile(distribution, [.025, .975]).tolist(),
            "scope": "Full participant/title rectangle; mean repeat losses. Reused development, descriptive uncertainty only; not an independent or causal null test."}


def require_replay_gate(gate, activation):
    """Reject absent or mismatched gate before any numeric cache interpretation."""
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("source_fixture_replay_passed") is not True:
        raise PermissionError("Passing independent source-fixture checkpoint replay required")
    if gate.get("confirmation_access") is not False or gate.get("validation_forward_replay_performed") is not False:
        raise PermissionError("Unexpected replay scope or protected access")
    if gate.get("checkpoint_csv_sha256") != activation["bindings"].get((OLD_NEURAL / "checkpoint_predictions.csv").relative_to(REPO).as_posix()):
        raise PermissionError("Replay gate does not bind the activated raw checkpoint CSV")
    return gate


class FitLedger:
    def __init__(self, private_dir, *, time_cap=TIME_CAP):
        self.private_dir = Path(private_dir)
        self.started = time.monotonic()
        self.time_cap = time_cap
        self.records = []

    def check(self):
        if time.monotonic() - self.started > self.time_cap:
            raise TimeoutError("Declared development CPU time cap exceeded")

    def fit(self, pairs, features, targets, alpha, **metadata):
        self.check()
        if len(self.records) >= FIT_CAP:
            raise PermissionError("Fixed 612-fit cap exceeded")
        fitted = fit_source_ridge(pairs, features, targets, alpha)
        path = self.private_dir / f"model_{len(self.records):03d}.npz"
        with path.open("xb") as f:
            np.savez_compressed(f, mean=fitted["scaler"].mean_, scale=fitted["scaler"].scale_,
                                variance=fitted["scaler"].var_, coef=fitted["estimator"].coef_,
                                intercept=np.asarray(fitted["estimator"].intercept_))
        self.records.append({"index": len(self.records), **metadata, "alpha": alpha,
                             "source_rows": len(pairs), "source_pairs": pairs,
                             "source_identity_sha256": fitted["source_identity_sha256"],
                             "private_state_name": path.name, "private_state_sha256": sha256(path)})
        return fitted


def _guarded_quality_targets(path, metadata, programme):
    targets = {}
    old = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p, y = cached_original_target(row, metadata, programme, "quality")
            if y is None:
                raise PermissionError("Permutation row in canonical quality source")
            if p in targets and targets[p] != y:
                raise ValueError("Repeated cached original targets disagree")
            targets[p] = y
            k, arm, model = int(row["rotation"]), row["arm"], row["model"]
            if arm in ARMS and model in ("context_only", "context_plus_spectral_eeg"):
                intended = {(r["subject"], r["vid"]) for r in programme["rotation_manifests"][k]["arms"][arm]}
                if p not in intended:
                    raise PermissionError("Legacy control row outside declared arm")
                value = float(row["prediction"])
                alpha = float(row["alpha"])
                if not np.isfinite(value) or not 0 <= value <= 7:
                    raise ValueError("Invalid cached control prediction")
                key = (k, arm, model, alpha, p)
                if key in old:
                    raise PermissionError("Duplicate legacy control row")
                old[key] = value
    if set(targets) != set(metadata):
        raise PermissionError("Incomplete canonical development targets")
    return targets, old


def _load_spectral_features(record_map, metadata, decisions):
    context, spectral, receipts = {}, {}, []
    source = sorted(p for p in metadata if metadata[p]["role"] == "source")
    validation = sorted(p for p in metadata if metadata[p]["role"] == "validation")
    for role, pairs in (("source", source), ("validation", validation)):
        if role == "validation" and len(spectral) != 980:
            raise PermissionError("Complete authenticated source features before validation cache access")
        for p in pairs:
            vector = context_features(metadata[p])
            if array_sha256(vector) != record_map[p]["context_feature_sha256"]:
                raise ValueError("Reconstructed context bytes differ")
            context[p] = vector
            if not decisions[p]["eeg_available"]:
                if role == "source":
                    raise PermissionError("Failed source cache holds diagnostic")
                receipts.append({"subject": p[0], "vid": p[1], "eeg_available": False, "feature_cache_opened": False})
                continue
            path = OLD_PRIVATE / f"{p[0]}_clip{p[1]:02d}_features.npz"
            digest = sha256(path)
            if digest != record_map[p]["feature_cache_sha256"]:
                raise ValueError("Frozen private spectral cache changed")
            with np.load(path, allow_pickle=False) as cache:
                if set(cache.files) != {"context", "spectral"}:
                    raise PermissionError("Unexpected private feature arrays")
                x = cache["spectral"]
                if x.shape != (150,) or x.dtype != np.dtype("float64") or not np.isfinite(x).all() or not np.array_equal(cache["context"], vector):
                    raise ValueError("Invalid or mismatched bound features")
                if array_sha256(x) != record_map[p]["spectral_feature_sha256"]:
                    raise ValueError("Bound spectral vector hash differs")
                spectral[p] = x.copy()
            receipts.append({"subject": p[0], "vid": p[1], "eeg_available": True,
                             "feature_cache_opened": True, "feature_cache_sha256": digest})
    if len(context) != 1260 or len(spectral) != 1245 or sum(not r["feature_cache_opened"] for r in receipts) != 15:
        raise PermissionError("Changed complete feature/QC coverage")
    return context, spectral, receipts


def _row(p, prediction, targets, metadata, **fields):
    return {**fields, "subject": p[0], "vid": p[1], "family": metadata[p]["family"],
            "acquisition_group": metadata[p]["acquisition_group"], "target": targets[p], "prediction": float(prediction)}


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refuse empty declared output")
    with Path(path).open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def persist_partial_trace(public_dir, ledger, cv_rows, prediction_rows, donor_rows, swap_rows):
    """Retain identity/fit traces of failed attempts under distinct filenames."""
    public_dir = Path(public_dir)
    save(public_dir / "partial_model_manifest.json", {"fits": ledger.records, "new_ridge_fits": len(ledger.records),
         "new_neural_fits": 0, "status": "partial_unqualified_attempt", "resumption_authorized": False})
    for name, rows in (("partial_cv_predictions.csv", cv_rows), ("partial_predictions.csv", prediction_rows),
                       ("partial_donor_assignments.csv", donor_rows), ("partial_swapped_predictions.csv", swap_rows)):
        if rows:
            _write_csv(public_dir / name, rows)


def cached_bool(value):
    if value is True or value in ("True", "true", "1"):
        return True
    if value is False or value in ("False", "false", "0"):
        return False
    raise ValueError("Invalid cached availability/fallback boolean")


def _neural_corrections(programme, metadata, targets, context_offsets, decisions):
    choice = load(OLD_NEURAL / "selection.json")["selection"]["context_residual"]
    if choice["learning_rate"] != .003 or choice["epoch"] != 120:
        raise PermissionError("Frozen selected residual schedule changed")
    selected = {}
    for rotation in programme["rotation_manifests"]:
        k = rotation["rotation"]
        selected[k] = (0.003, 120)
    raw = defaultdict(dict)
    with (OLD_NEURAL / "checkpoint_predictions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = guard_cached_identity(row, metadata)
            if row["acquisition_group"] != metadata[p]["acquisition_group"]:
                raise PermissionError("Checkpoint acquisition mismatch")
            k, arm = int(row["rotation"]), row["arm"]
            if k not in selected:
                raise PermissionError("Unknown checkpoint rotation")
            if row["readout"] != "context_residual" or arm not in ARMS:
                continue
            rate, epoch = float(row["learning_rate"]), int(row["epoch"])
            if (rate, epoch) != selected[k]:
                continue
            expected = {(r["subject"], r["vid"]) for r in programme["rotation_manifests"][k]["arms"][arm]}
            if p not in expected:
                raise PermissionError("Selected checkpoint leaves arm")
            if cached_bool(row["eeg_available"]) != decisions[p]["eeg_available"] or cached_bool(row["fallback"]) != (not decisions[p]["eeg_available"]):
                raise PermissionError("Selected checkpoint availability/fallback changed")
            seed = int(row["seed"])
            if seed not in (20261101, 20261102) or seed in raw[k, arm, p]:
                raise PermissionError("Duplicate or unexpected neural seed")
            if float(row["target"]) != targets[p]:
                raise ValueError("Selected neural cached target differs")
            value = float(row["prediction_raw"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite selected raw neural prediction")
            raw[k, arm, p][seed] = value
    corrections, predictions = {}, {}
    for rotation in programme["rotation_manifests"]:
        k = rotation["rotation"]
        for arm in ARMS:
            panel = [(r["subject"], r["vid"]) for r in rotation["arms"][arm]]
            corrections[k, arm], predictions[k, arm] = {}, {}
            for p in panel:
                if set(raw[k, arm, p]) != {20261101, 20261102}:
                    raise PermissionError("Incomplete selected two-seed neural panel")
                values = [raw[k, arm, p][seed] for seed in (20261101, 20261102)]
                corrections[k, arm][p] = average_raw_correction(values, context_offsets[k][p])
                predictions[k, arm][p] = float(np.clip(np.mean(values), 0, 7))
    seen, checks = set(), []
    with (OLD_NEURAL / "ensemble_predictions.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = guard_cached_identity(row, metadata)
            if row["acquisition_group"] != metadata[p]["acquisition_group"]:
                raise PermissionError("Neural ensemble acquisition mismatch")
            if row["readout"] != "context_residual" or row["arm"] not in ARMS:
                continue
            k, arm = int(row["rotation"]), row["arm"]
            if k not in selected or (float(row["selected_learning_rate"]), int(row["selected_epoch"])) != selected[k]:
                raise PermissionError("Changed selected ensemble schedule")
            key = k, arm, p
            if key in seen or p not in predictions[k, arm]:
                raise PermissionError("Duplicate or wrong selected ensemble identity")
            seen.add(key)
            if cached_bool(row["eeg_available"]) != decisions[p]["eeg_available"] or cached_bool(row["fallback"]) != (not decisions[p]["eeg_available"]):
                raise PermissionError("Selected ensemble availability/fallback changed")
            if float(row["target"]) != targets[p] or abs(float(row["prediction"]) - predictions[k, arm][p]) > 2e-9:
                raise ValueError("Selected raw seeds do not reproduce the legacy neural ensemble")
            checks.append(abs(float(row["prediction"]) - predictions[k, arm][p]))
    if len(seen) != 840:
        raise PermissionError("Incomplete legacy neural ensemble reproduction")
    return corrections, predictions, {"rows": len(seen), "maximum_absolute_error": max(checks), "passed": True}


def run(public_dir, private_dir, activation, neural_replay_gate=None):
    """Execute only an externally declared, bound, development diagnostic once.

    Root wrapper owns declaration and independently verifies replay gate bindings.
    Required activation: bindings, new_fit_cap=612, time_cap_seconds<=1800,
    confirmation_access=false, research_question_changed=false,
    new_original_rating_decoding=0, original_data_access=false.
    """
    public_dir, private_dir = Path(public_dir), Path(private_dir)
    if activation.get("confirmation_access") is not False or activation.get("research_question_changed") is not False or activation.get("new_original_rating_decoding") != 0 or activation.get("original_data_access") is not False:
        raise PermissionError("Activation permits undeclared/protected data access")
    if activation.get("new_fit_cap") != FIT_CAP or not 0 < activation.get("time_cap_seconds", 0) <= TIME_CAP:
        raise PermissionError("Changed diagnostic fit/time cap")
    for name, expected in activation["bindings"].items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()) or sha256(path) != expected:
            raise PermissionError("Activated input/code binding changed")
    require_replay_gate(neural_replay_gate, activation)
    if not set(required_binding_paths()) <= set(activation["bindings"]):
        raise PermissionError("Mandatory metadata, outcome-cache or implementation binding absent")
    if any((public_dir / n).exists() for n in ("execution.json", "cv_predictions.csv", "model_manifest.json")) or private_dir.exists():
        raise PermissionError("Preserve previous or partial diagnostic; no resumption overwrite")
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=False)
    ledger = FitLedger(private_dir, time_cap=activation["time_cap_seconds"])
    cv_rows, cv_records, prediction_rows, selection_records, prior_records, strongest = [], [], [], [], [], {}
    pairing_records, donor_rows, swap_rows = [], [], []
    try:
        programme, record_map, metadata, decisions = metadata_bundle()
        frozen_folds = {str(r["rotation"]): crossed_source_folds([(q["subject"], q["vid"]) for q in r["arms"]["source_fit"]], metadata, programme["source_participant_oof_folds"]) for r in programme["rotation_manifests"]}
        if json.loads(json.dumps(frozen_folds)) != activation.get("source_crossed_folds"):
            raise PermissionError("Runtime source folds differ from the outcome-blind activation")
        if activation.get("swap_seeds") != list(SWAP_SEEDS) or activation.get("material_salt") != MATERIAL_SALT:
            raise PermissionError("Runtime seed/material plan differs from activation")
        context, spectral, receipts = _load_spectral_features(record_map, metadata, decisions)
        save(public_dir / "feature_cache_reuse.json", {"records": receipts, "private_feature_arrays_read": 1245,
             "unavailable_private_arrays_read": 0, "source_gate_passed": True, "original_files_opened": 0,
             "source_feature_caches_authenticated": 980, "fresh_full_source_qc_window_replay": False,
             "separate_source_fixture_replay_gate_passed": True,
             "source_fixture_pairs_by_rotation": sum(len(x) for x in neural_replay_gate["fixture_ids"].values())})
        targets, old_predictions = _guarded_quality_targets(OLD_QUALITY / "predictions.csv", metadata, programme)
        save(public_dir / "target_reuse.json", {"canonical_cached_targets": len(targets), "source": (OLD_QUALITY / "predictions.csv").relative_to(REPO).as_posix(),
             "sha256": sha256(OLD_QUALITY / "predictions.csv"), "new_original_rating_decoding": 0, "confirmation_access": False})
        decomposition = {role: balanced_decomposition(sorted(p for p in metadata if metadata[p]["role"] == role), targets, metadata) for role in ("source", "validation")}
        save(public_dir / "target_decomposition.json", decomposition)
        category = {p: x[:8].copy() for p, x in context.items()}
        maps = {"category_only": category, "full_context": context}
        old_offsets, anchor_predictions, reproduction = {}, {}, []
        previous = load(OLD_QUALITY / "comparison.json")
        with threadpool_limits(limits=2, user_api="blas"):
            for rotation in programme["rotation_manifests"]:
                k = rotation["rotation"]
                arms = {a: [(r["subject"], r["vid"]) for r in rows] for a, rows in rotation["arms"].items()}
                source = arms["source_fit"]
                folds = crossed_source_folds(source, metadata, programme["source_participant_oof_folds"])
                source_scores = {}
                prior_cv = {name: [] for name in PRIOR_NAMES}
                for fold in folds:
                    fitted_priors = estimate_priors(fold["training"], targets, metadata)
                    prior_records.append({"rotation": k, "stage": "crossed_source_cv", "person_fold": fold["person_fold"], "material_fold": fold["material_fold"], "source_pairs": fold["training"], "fitted": fitted_priors})
                    values = prior_predictions(fitted_priors, fold["held"], metadata)
                    for name in PRIOR_NAMES:
                        rows = [_row(p, value, targets, metadata, rotation=k, model=name, alpha=None, person_fold=fold["person_fold"], material_fold=fold["material_fold"]) for p, value in zip(fold["held"], values[name])]
                        prior_cv[name].extend(rows)
                        cv_rows.extend(rows)
                full_priors = estimate_priors(source, targets, metadata)
                prior_records.append({"rotation": k, "stage": "full_source", "source_pairs": source, "fitted": full_priors})
                for name in PRIOR_NAMES:
                    source_scores[name] = participant_film_mae(prior_cv[name])
                for arm in ARMS:
                    values = prior_predictions(full_priors, arms[arm], metadata)
                    for name in PRIOR_NAMES:
                        prediction_rows.extend(_row(p, value, targets, metadata, rotation=k, model=name, arm=arm, alpha=None, eeg_available=decisions[p]["eeg_available"], fallback=False) for p, value in zip(arms[arm], values[name]))
                for name, features in maps.items():
                    candidates = []
                    for alpha in ALPHAS:
                        held_rows = []
                        for fold in folds:
                            fitted = ledger.fit(fold["training"], features, targets, alpha, rotation=k, model=name, stage="crossed_source_cv", person_fold=fold["person_fold"], material_fold=fold["material_fold"])
                            values = predict_ridge(fitted, fold["held"], features)
                            rows = [_row(p, value, targets, metadata, rotation=k, model=name, alpha=alpha, person_fold=fold["person_fold"], material_fold=fold["material_fold"]) for p, value in zip(fold["held"], values)]
                            held_rows.extend(rows)
                            cv_rows.extend(rows)
                        mae = participant_film_mae(held_rows)
                        mse = float(np.mean([(r["target"] - r["prediction"]) ** 2 for r in held_rows]))
                        candidates.append({"alpha": alpha, "participant_title_mae": mae, "trial_mse": mse})
                    chosen = min(range(len(candidates)), key=lambda i: (candidates[i]["participant_title_mae"], candidates[i]["alpha"]))
                    alpha = candidates[chosen]["alpha"]
                    source_scores[name] = candidates[chosen]["participant_title_mae"]
                    fitted = ledger.fit(source, features, targets, alpha, rotation=k, model=name, stage="selected_full_source")
                    selections = {"rotation": k, "model": name, "selected_alpha": alpha, "candidates": candidates, "selection_data": "crossed source-person/title CV only", "validation_selection": False}
                    selection_records.append(selections)
                    for arm in ARMS:
                        values = predict_ridge(fitted, arms[arm], features)
                        prediction_rows.extend(_row(p, value, targets, metadata, rotation=k, model=name, arm=arm, alpha=alpha, eeg_available=decisions[p]["eeg_available"], fallback=False) for p, value in zip(arms[arm], values))
                strongest[k] = {"model": min(COMPARATOR_PRIORITY, key=lambda name: (source_scores[name], COMPARATOR_PRIORITY.index(name))), "source_cv_mae": source_scores, "tie_priority": list(COMPARATOR_PRIORITY), "selection_data": "crossed source-person/title CV only"}
                old_context = ledger.fit(source, context, targets, OLD_CONTEXT_ALPHAS[k], rotation=k, model="old_full_context", stage="legacy_anchor_reproduction")
                joint_features = {p: np.r_[context[p], spectral[p]] for p in spectral}
                old_alpha = previous["rotations"][k]["selected"]["context_plus_spectral_eeg"]["alpha"]
                old_joint = ledger.fit(source, joint_features, targets, old_alpha, rotation=k, model="old_joint_spectral", stage="legacy_anchor_reproduction")
                old_offsets[k] = {}
                anchor_predictions[k, "joint_spectral"] = {}
                for arm in ARMS:
                    pairs = arms[arm]
                    old_offsets[k].update(zip(pairs, predict_ridge(old_context, pairs, context)))
                    available = [p for p in pairs if decisions[p]["eeg_available"]]
                    joint = dict(zip(available, predict_ridge(old_joint, available, joint_features)))
                    joint.update({p: old_offsets[k][p] for p in pairs if not decisions[p]["eeg_available"]})
                    anchor_predictions[k, "joint_spectral"].update(joint)
                    for model, alpha, predicted in (("context_only", OLD_CONTEXT_ALPHAS[k], old_offsets[k]), ("context_plus_spectral_eeg", old_alpha, joint)):
                        errors = [abs(predicted[p] - old_predictions[k, arm, model, alpha, p]) for p in pairs]
                        if max(errors) > 2e-9:
                            raise ValueError("Legacy validation predictions not reproduced tightly")
                        reproduction.append({"rotation": k, "arm": arm, "model": model, "rows": len(pairs), "max_absolute_error": max(errors)})
                    for p in pairs:
                        prediction_rows.append(_row(p, joint[p], targets, metadata, rotation=k, model="joint_spectral", arm=arm, alpha=old_alpha, eeg_available=decisions[p]["eeg_available"], fallback=not decisions[p]["eeg_available"]))
                cv_records.append({"rotation": k, "folds": folds})
                print({"phase": "source_context_cv_complete", "rotation": k, "fits": len(ledger.records), "elapsed_seconds": round(time.monotonic() - ledger.started, 1)}, flush=True)
            if len(ledger.records) != FIT_CAP or len(cv_rows) != 21840 or len(prior_records) != 63:
                raise PermissionError("Incomplete fixed fit/CV coverage")
            corrections, neural_original, neural_reproduction = _neural_corrections(programme, metadata, targets, old_offsets, decisions)
            save(public_dir / "legacy_prediction_reproduction.json", {"passed": True, "panels": reproduction, "neural_ensemble": neural_reproduction, "tolerance": 2e-9})
            for rotation in programme["rotation_manifests"]:
                k = rotation["rotation"]
                source = [(r["subject"], r["vid"]) for r in rotation["arms"]["source_fit"]]
                old_alpha = previous["rotations"][k]["selected"]["context_plus_spectral_eeg"]["alpha"]
                state_record = next(r for r in ledger.records if r["rotation"] == k and r["model"] == "old_joint_spectral")
                # Recover a lightweight prediction function from the exact fitted
                # coefficient state; no extra estimator fit is performed.
                with np.load(private_dir / state_record["private_state_name"], allow_pickle=False) as f:
                    mean, scale, coef, intercept = (f[x].copy() for x in ("mean", "scale", "coef", "intercept"))
                for arm in ARMS:
                    pairs = [(r["subject"], r["vid"]) for r in rotation["arms"][arm]]
                    original = {"joint_spectral": {p: anchor_predictions[k, "joint_spectral"][p] for p in pairs}, "eegnet_residual": neural_original[k, arm]}
                    identity_donors = {p: {"donor": p} for p in pairs}
                    reconstructed = residual_swap_predictions(pairs, identity_donors, old_offsets[k], corrections[k, arm], decisions)
                    if max(abs(reconstructed[p] - original["eegnet_residual"][p]) for p in pairs) > 2e-9:
                        raise ValueError("Raw neural correction reconstruction differs")
                    for p in pairs:
                        prediction_rows.append(_row(p, original["eegnet_residual"][p], targets, metadata, rotation=k, model="eegnet_residual", arm=arm, alpha=None, eeg_available=decisions[p]["eeg_available"], fallback=not decisions[p]["eeg_available"]))
                    for scheme in SCHEMES:
                        repeated = {name: [] for name in original}
                        eligibility = None
                        for repeat, seed in enumerate(SWAP_SEEDS):
                            ledger.check()
                            assignments = donor_assignments(pairs, metadata, decisions, scheme, seed)
                            now_eligible = {p: assignments[p]["eligible"] for p in pairs}
                            if eligibility is not None and now_eligible != eligibility:
                                raise PermissionError("Eligibility changed across fixed repeats")
                            eligibility = now_eligible
                            for p in pairs:
                                d = assignments[p]
                                donor_rows.append({"rotation": k, "arm": arm, "scheme": scheme, "repeat": repeat, "seed": seed, "subject": p[0], "vid": p[1], "donor_subject": d["donor"][0], "donor_vid": d["donor"][1], "eligible": d["eligible"], "stratum_size": d["stratum_size"], "reason": d["reason"]})
                            mixed = mixed_joint_features(pairs, assignments, context, spectral, decisions)
                            joint_predictions = {p: float(np.clip(((mixed[p] - mean) / scale) @ coef + intercept, 0, 7)) if p in mixed else old_offsets[k][p] for p in pairs}
                            neural_predictions = residual_swap_predictions(pairs, assignments, old_offsets[k], corrections[k, arm], decisions)
                            for anchor, values in (("joint_spectral", joint_predictions), ("eegnet_residual", neural_predictions)):
                                rows = [_row(p, values[p], targets, metadata, rotation=k, anchor=anchor, arm=arm, scheme=scheme, repeat=repeat, seed=seed, donor_subject=assignments[p]["donor"][0], donor_vid=assignments[p]["donor"][1], eligible=assignments[p]["eligible"], eeg_available=decisions[p]["eeg_available"], fallback=not decisions[p]["eeg_available"]) for p in pairs]
                                repeated[anchor].append(rows)
                                swap_rows.extend(rows)
                        for anchor, values in original.items():
                            originals = [_row(p, values[p], targets, metadata) for p in pairs]
                            report = repeat_loss_summary(originals, repeated[anchor], eligibility)
                            report["full_panel_crossed_interval"] = crossed_loss_interval(originals, repeated[anchor])
                            families = {}
                            for family in sorted({metadata[p]["family"] for p in pairs}):
                                members = {p for p in pairs if metadata[p]["family"] == family}
                                families[family] = repeat_loss_summary([r for r in originals if (r["subject"], r["vid"]) in members], [[r for r in repeat_rows if (r["subject"], r["vid"]) in members] for repeat_rows in repeated[anchor]], {p: eligibility[p] for p in members})
                            pairing_records.append({"rotation": k, "arm": arm, "scheme": scheme, "anchor": anchor, **report, "per_family": families})
            if len(donor_rows) != 53760 or len(swap_rows) != 107520 or len(pairing_records) != 24:
                raise PermissionError("Incomplete fixed pairing rectangle")
            _write_csv(public_dir / "cv_predictions.csv", cv_rows)
            _write_csv(public_dir / "predictions.csv", prediction_rows)
            _write_csv(public_dir / "donor_assignments.csv", donor_rows)
            _write_csv(public_dir / "swapped_predictions.csv", swap_rows)
            save(public_dir / "source_cv.json", {"material_salt": MATERIAL_SALT, "material_hash_format": "{salt}/{vid}", "rotations": cv_records, "selections": selection_records, "strongest_comparators": strongest, "validation_selection": False, "alpha_tie_rule": "MAE, then smaller alpha"})
            save(public_dir / "prior_manifest.json", {"constructions": prior_records, "source_prior_constructions": len(prior_records), "prediction_rules": list(PRIOR_NAMES)})
            save(public_dir / "model_manifest.json", {"fits": ledger.records, "new_ridge_fits": len(ledger.records), "new_neural_fits": 0})
            comparisons = []
            for k in range(3):
                for arm in ARMS:
                    comparison = {"rotation": k, "arm": arm}
                    for name in COMPARATOR_PRIORITY + ("joint_spectral", "eegnet_residual"):
                        comparison[name] = loss_metrics([r for r in prediction_rows if r["rotation"] == k and r["arm"] == arm and r["model"] == name])
                    comparison["source_selected_comparator"] = strongest[k]["model"]
                    legacy_rows = [_row((r["subject"], r["vid"]), old_offsets[k][r["subject"], r["vid"]], targets, metadata) for r in programme["rotation_manifests"][k]["arms"][arm]]
                    comparison["legacy_full_context_offset"] = loss_metrics(legacy_rows)
                    comparator_mae = comparison[strongest[k]["model"]]["participant_title_mae"]
                    comparison["eeg_gain_over_source_selected_comparator"] = {name: comparator_mae - comparison[name]["participant_title_mae"] for name in ("joint_spectral", "eegnet_residual")}
                    comparison["eeg_gain_over_legacy_context_offset"] = {name: comparison["legacy_full_context_offset"]["participant_title_mae"] - comparison[name]["participant_title_mae"] for name in ("joint_spectral", "eegnet_residual")}
                    comparisons.append(comparison)
            save(public_dir / "comparison.json", {"context_panels": comparisons, "pairing_panels": pairing_records,
                 "new_ridge_fits": FIT_CAP, "new_neural_fits": 0, "paired_seed_repeats": 32,
                 "adaptively_reused_development": True, "independent_confirmation": False,
                 "source_fixture_forward_replay": True, "validation_forward_replay_performed": False,
                 "neural_scope": "Qualified cached raw-validation correction swaps; independent state-to-window forward replay was limited to source fixtures.",
                 "whole_trial_coverage_tested": False, "confirmation_access": False, "research_question_changed": False})
        ledger.check()
        save(public_dir / "execution.json", {"status": "completed_development_diagnosis", "new_ridge_fits": len(ledger.records), "new_neural_fits": 0,
             "elapsed_seconds": time.monotonic() - ledger.started, "cv_prediction_rows": len(cv_rows), "prediction_rows": len(prediction_rows),
             "donor_rows": len(donor_rows), "swapped_prediction_rows": len(swap_rows), "original_files_opened": 0,
             "new_original_rating_decoding": 0, "confirmation_access": False, "research_question_changed": False})
        return load(public_dir / "comparison.json")
    except BaseException as exc:
        persist_partial_trace(public_dir, ledger, cv_rows, prediction_rows, donor_rows, swap_rows)
        if not (public_dir / "execution.json").exists():
            save(public_dir / "execution.json", {"status": "held_or_failed", "error_type": type(exc).__name__, "error": str(exc),
                 "new_ridge_fits": len(ledger.records), "new_neural_fits": 0, "elapsed_seconds": time.monotonic() - ledger.started,
                 "original_files_opened": 0, "new_original_rating_decoding": 0, "confirmation_access": False, "research_question_changed": False})
        raise
