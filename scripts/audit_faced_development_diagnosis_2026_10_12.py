"""Independent public-only numerical audit; no import-time I/O or fitting.

Run only after the separately activated diagnostic is complete. No private
arrays, fitted states, original recordings or original ratings are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
PUBLIC = BASE / "faced_target_context_pairing_2026-10-12"
QUALITY = BASE / "faced_inference_quality_2026-10-11"
NEURAL = BASE / "faced_eegnet_diagnostic_v2_2026-10-11"
ARMS = ("validation_unseen", "validation_familiar")
PRIORS = ("global_median", "coarse_valence_median", "category_median")
CONTEXT = ("category_only", "full_context")
ANCHORS = ("joint_spectral", "eegnet_residual")
PRIORITY = CONTEXT + PRIORS
ALPHAS = (.1, 1., 10., 100., 1000.)
SCHEMES = ("person_same_film_acquisition", "film_same_person_coarse_valence")


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, label, tolerance=2e-9):
    if not np.isfinite([actual, expected]).all() or abs(actual - expected) > tolerance:
        raise AssertionError(f"Numerical mismatch: {label}: {actual!r} versus {expected!r}")


def boolean(value):
    if value not in ("True", "False"):
        raise AssertionError("Invalid CSV boolean")
    return value == "True"


def identity(row, metadata):
    pair = row["subject"], int(row["vid"])
    if pair not in metadata or row.get("family", metadata[pair]["family"]) != metadata[pair]["family"]:
        raise PermissionError("Identity/family guard before numerical conversion")
    if row.get("acquisition_group", metadata[pair]["acquisition_group"]) != metadata[pair]["acquisition_group"]:
        raise PermissionError("Acquisition identity guard before numerical conversion")
    return pair


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def score(panel):
    groups = defaultdict(list)
    for r in panel:
        groups[r["subject"], r["family"]].append(abs(r["target"] - r["prediction"]))
    people = defaultdict(list)
    for (person, _), values in groups.items():
        people[person].append(float(np.mean(values)))
    return None if not people else float(np.mean([np.mean(v) for v in people.values()]))


def summarize(panel):
    return {"participant_title_mae": score(panel), "trials": len(panel),
            "participants": len({r["subject"] for r in panel}),
            "title_families": len({r["family"] for r in panel})}


def check_metric(actual, panel, label):
    expected = summarize(panel)
    for key in ("trials", "participants", "title_families"):
        if actual[key] != expected[key]:
            raise AssertionError(f"Changed metric denominator: {label}/{key}")
    close(actual["participant_title_mae"], expected["participant_title_mae"], label)
    if "trial_mae" in actual:
        close(actual["trial_mae"], np.mean([abs(r["target"] - r["prediction"]) for r in panel]), label + "/trial")


def source_prior(training, targets, metadata):
    y = [targets[p] for p in training]
    result = {"global": float(np.median(y)), "coarse": {}, "category": {}, "video": {}}
    for field, key in (("assigned_valence", "coarse"), ("assigned_category", "category"), ("vid", "video")):
        groups = defaultdict(list)
        for p in training:
            groups[str(p[1] if field == "vid" else metadata[p][field])].append(targets[p])
        result[key] = {g: float(np.median(v)) for g, v in groups.items()}
    return result


def prior_value(name, state, pair, metadata):
    if name == "global_median":
        return state["global"]
    key, field = ("coarse", "assigned_valence") if name == "coarse_valence_median" else ("category", "assigned_category")
    return state[key].get(str(metadata[pair][field]), state["global"])


def audit(*, write=True):
    execution = load(PUBLIC / "execution.json")
    if execution["status"] != "completed_development_diagnosis":
        raise PermissionError("Completed diagnostic required before public numerical audit")
    outputs = (PUBLIC / "independent_numerical_audit.json", PUBLIC / "acquisition_summaries.json")
    if write and any(p.exists() for p in outputs):
        raise PermissionError("Preserve existing audit outputs; no overwrite")
    activation = load(PUBLIC / "activation.json")
    for relative, expected in activation["bindings"].items():
        p = (REPO / relative).resolve()
        if not p.is_relative_to(REPO) or digest(p) != expected:
            raise AssertionError("Frozen dependency changed: " + relative)
    if digest(REPO.parent / "report.tex") != activation["original_report_sha256"]:
        raise AssertionError("Historical manuscript changed")
    programme = load(BASE / "faced_development_protocol_2026-10-11/plan.json")
    reservation = load(BASE / "faced_backup_2026-10-10/participant_reservation.json")
    catalogue = {r["clip"]: r for r in load(BASE / "faced_codebooks_2026-10-10/audit.json")["stimuli"]}
    records = load(BASE / "faced_artifact_control_2026-10-11/source_records.json")["records"] + load(BASE / "faced_artifact_control_2026-10-11/validation_records.json")["records"]
    metadata = {(r["subject"], r["vid"]): {**catalogue[r["vid"]], **r,
                 "family": catalogue[r["vid"]]["source_film_sha256"]} for r in records}
    permitted = {(r["subject"], r["vid"]) for r in programme["input_access"]["future_development_pairs"]}
    if set(metadata) != permitted or len(records) != 1260:
        raise AssertionError("Changed metadata rectangle")
    for p, m in metadata.items():
        expected_role = "source" if p[0] in reservation["development_source"] else "validation"
        if p[0] in reservation["confirmation"] or m["role"] != expected_role:
            raise PermissionError("Protected or misassigned role")
    decisions = {(r["subject"], r["vid"]): r["eeg_available"] for r in load(QUALITY / "signal_audit.json")["trial_decisions"]}
    targets, offsets, spectral_legacy = {}, {}, {}
    for r in rows(QUALITY / "predictions.csv"):
        p = identity(r, metadata)
        y = float(r["target"])
        if not np.isfinite(y) or not 0 <= y <= 7 or p in targets and targets[p] != y:
            raise AssertionError("Invalid/inconsistent canonical original rating")
        targets[p] = y
        k, arm = int(r["rotation"]), r["arm"]
        if arm in ARMS and r["model"] == "context_only" and float(r["alpha"]) == {0: .1, 1: 100., 2: 100.}[k]:
            offsets[k, arm, p] = float(r["prediction"])
        if arm in ARMS and r["model"] == "context_plus_spectral_eeg":
            spectral_legacy[k, arm, float(r["alpha"]), p] = float(r["prediction"])
    if set(targets) != permitted or len(offsets) != 840:
        raise AssertionError("Incomplete canonical target/context-offset projection")
    arms = {(r["rotation"], name): {(q["subject"], q["vid"]) for q in values}
            for r in programme["rotation_manifests"] for name, values in r["arms"].items()}
    source_cv = load(PUBLIC / "source_cv.json")
    folds = {}
    category_missing_cells = 0
    for r in source_cv["rotations"]:
        k = r["rotation"]
        if r["folds"] != activation["source_crossed_folds"][str(k)]:
            raise AssertionError("Runtime source folds differ from frozen activation")
        seen = []
        for f in r["folds"]:
            training = {tuple(p) for p in f["training"]}
            held = {tuple(p) for p in f["held"]}
            hp, ht = {p[0] for p in held}, {metadata[p]["family"] for p in held}
            expected = {p for p in arms[k, "source_fit"] if p[0] not in hp and metadata[p]["family"] not in ht}
            if len(training) != 336 or len(held) != 28 or training != expected or not held <= arms[k, "source_fit"]:
                raise AssertionError("Crossed fold leakage or changed training quadrant")
            train_categories = {metadata[p]["assigned_category"] for p in training}
            held_categories = {metadata[p]["assigned_category"] for p in held}
            category_missing_cells += int(not train_categories & held_categories)
            folds[k, f["person_fold"], f["material_fold"]] = training, held
            seen.extend(held)
        if Counter(seen) != Counter(arms[k, "source_fit"]):
            raise AssertionError("Held source coverage differs")
    manifest = load(PUBLIC / "model_manifest.json")["fits"]
    if len(manifest) != 612 or Counter(r["stage"] for r in manifest) != {"crossed_source_cv": 600, "selected_full_source": 6, "legacy_anchor_reproduction": 6}:
        raise AssertionError("Changed Ridge fit budget/stages")
    for r in manifest:
        k = r["rotation"]
        expected = folds[k, r["person_fold"], r["material_fold"]][0] if r["stage"] == "crossed_source_cv" else arms[k, "source_fit"]
        pairs = [tuple(p) for p in r["source_pairs"]]
        if set(pairs) != expected or len(pairs) != len(expected) or r["source_rows"] != len(expected):
            raise AssertionError("Fitted state source identities differ")
        if hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest() != r["source_identity_sha256"]:
            raise AssertionError("Source fit identity digest differs")
    priors = {}
    constructions = load(PUBLIC / "prior_manifest.json")["constructions"]
    if len(constructions) != 63:
        raise AssertionError("Changed source-prior construction cap")
    for r in constructions:
        k = r["rotation"]
        key = (k, r["person_fold"], r["material_fold"]) if r["stage"] == "crossed_source_cv" else (k, None, None)
        training = folds[key][0] if key[1] is not None else arms[k, "source_fit"]
        if {tuple(p) for p in r["source_pairs"]} != training or key in priors:
            raise AssertionError("Prior construction source identities changed")
        state = source_prior(training, targets, metadata)
        for field in state:
            if field == "global":
                close(r["fitted"][field], state[field], "source median")
            else:
                if r["fitted"][field].keys() != state[field].keys():
                    raise AssertionError("Prior group vocabulary changed")
                for g, v in state[field].items():
                    close(r["fitted"][field][g], v, "source grouped median")
        priors[key] = state
    cv, cv_seen = defaultdict(list), set()
    for r in rows(PUBLIC / "cv_predictions.csv"):
        p = identity(r, metadata)
        k, pf, mf = int(r["rotation"]), int(r["person_fold"]), int(r["material_fold"])
        name, alpha = r["model"], None if not r["alpha"] else float(r["alpha"])
        key = k, name, alpha, p
        if key in cv_seen or p not in folds[k, pf, mf][1] or name not in PRIORITY:
            raise AssertionError("CV duplicate or wrong held identity")
        cv_seen.add(key)
        y, prediction = float(r["target"]), float(r["prediction"])
        close(y, targets[p], "CV target", 0)
        if name in PRIORS:
            close(prediction, prior_value(name, priors[k, pf, mf], p, metadata), "CV source prior")
        elif alpha not in ALPHAS:
            raise AssertionError("Changed Ridge alpha")
        cv[k, name, alpha].append({**r, "vid": p[1], "target": y, "prediction": prediction})
    if len(cv_seen) != 21840:
        raise AssertionError("Incomplete context/prior CV rows")
    selections, source_scores = {}, defaultdict(dict)
    for r in source_cv["selections"]:
        k, name = r["rotation"], r["model"]
        for candidate in r["candidates"]:
            panel = cv[k, name, candidate["alpha"]]
            if {(q["subject"], q["vid"]) for q in panel} != arms[k, "source_fit"]:
                raise AssertionError("Incomplete per-candidate held source rectangle")
            close(candidate["participant_title_mae"], score(panel), "CV MAE")
            close(candidate["trial_mse"], np.mean([(q["target"] - q["prediction"]) ** 2 for q in panel]), "CV MSE")
        choice = min(r["candidates"], key=lambda x: (x["participant_title_mae"], x["alpha"]))
        if choice["alpha"] != r["selected_alpha"]:
            raise AssertionError("Source alpha selection changed")
        selections[k, name] = choice["alpha"]
        source_scores[k][name] = choice["participant_title_mae"]
    for k in range(3):
        for name in PRIORS:
            source_scores[k][name] = score(cv[k, name, None])
        actual = source_cv["strongest_comparators"][str(k)]
        chosen = min(PRIORITY, key=lambda name: (source_scores[k][name], PRIORITY.index(name)))
        if actual["model"] != chosen:
            raise AssertionError("Source-selected strongest comparator changed")
        for name in PRIORITY:
            close(actual["source_cv_mae"][name], source_scores[k][name], "Strong comparator source score")
    decomposition = load(PUBLIC / "target_decomposition.json")
    for role in ("source", "validation"):
        people = sorted({p[0] for p in permitted if metadata[p]["role"] == role})
        films = sorted({metadata[p]["family"] for p in permitted if metadata[p]["role"] == role})
        lookup = {(p[0], metadata[p]["family"]): targets[p] for p in permitted if metadata[p]["role"] == role}
        matrix = np.array([[lookup[s, f] for f in films] for s in people])
        grand, person, title = matrix.mean(), matrix.mean(axis=1), matrix.mean(axis=0)
        residual = matrix - person[:, None] - title[None, :] + grand
        ss = {"person": len(films) * np.square(person - grand).sum(),
              "title": len(people) * np.square(title - grand).sum(), "additive_residual": np.square(residual).sum()}
        total = np.square(matrix - grand).sum()
        close(decomposition[role]["grand_mean"], grand, "Grand mean")
        close(decomposition[role]["total_ss"], total, "Total SS")
        close(sum(ss.values()), total, "SS reconstruction")
        for s, value in zip(people, person):
            close(decomposition[role]["person_means"][s], value, "Participant mean")
        for f, value in zip(films, title):
            close(decomposition[role]["title_means"][f], value, "Film mean")
        for key, value in ss.items():
            close(decomposition[role]["sum_squares"][key], value, "Decomposition SS")
            close(decomposition[role]["fractions"][key], value / total, "Decomposition fraction")
    predictions = defaultdict(list)
    prediction_map = {}
    for r in rows(PUBLIC / "predictions.csv"):
        p = identity(r, metadata)
        k, arm, name = int(r["rotation"]), r["arm"], r["model"]
        key = k, arm, name, p
        if key in prediction_map or p not in arms[k, arm] or name not in PRIORITY + ANCHORS:
            raise AssertionError("Prediction duplicate or wrong validation identity")
        y, value = float(r["target"]), float(r["prediction"])
        close(y, targets[p], "Validation target", 0)
        if not np.isfinite(value) or not 0 <= value <= 7 or boolean(r["eeg_available"]) != decisions[p]:
            raise AssertionError("Prediction scale/availability changed")
        if boolean(r["fallback"]) != (name in ANCHORS and not decisions[p]):
            raise AssertionError("Fixed fallback changed")
        if name in PRIORS:
            close(value, prior_value(name, priors[k, None, None], p, metadata), "Final source prior")
        if name in CONTEXT and float(r["alpha"]) != selections[k, name]:
            raise AssertionError("Final context selection differs")
        if name == "joint_spectral":
            close(value, spectral_legacy[k, arm, float(r["alpha"]), p], "Original spectral anchor")
        if name in ANCHORS and not decisions[p]:
            close(value, offsets[k, arm, p], "Original fallback")
        normalized = {**r, "vid": p[1], "target": y, "prediction": value}
        predictions[k, arm, name].append(normalized)
        prediction_map[key] = normalized
    if len(prediction_map) != 5880:
        raise AssertionError("Incomplete ordinary prediction rectangle")
    raw = defaultdict(dict)
    for r in rows(NEURAL / "checkpoint_predictions.csv"):
        if r["readout"] != "context_residual" or r["arm"] not in ARMS or float(r["learning_rate"]) != .003 or int(r["epoch"]) != 120:
            continue
        p = identity(r, metadata)
        k, arm, seed = int(r["rotation"]), r["arm"], int(r["seed"])
        if p not in arms[k, arm] or seed not in (20261101, 20261102) or seed in raw[k, arm, p]:
            raise AssertionError("Changed selected raw neural panel")
        close(float(r["target"]), targets[p], "Raw neural target", 0)
        raw[k, arm, p][seed] = float(r["prediction_raw"])
    correction = {}
    for key, values in raw.items():
        if set(values) != {20261101, 20261102}:
            raise AssertionError("Missing raw neural seed")
        mean = np.mean(list(values.values()))
        k, arm, p = key
        close(prediction_map[k, arm, "eegnet_residual", p]["prediction"], np.clip(mean, 0, 7), "Raw neural anchor")
        correction[key] = mean - offsets[key]
    donors = {}
    donor_panels = defaultdict(list)
    for r in rows(PUBLIC / "donor_assignments.csv"):
        p = identity(r, metadata)
        d = r["donor_subject"], int(r["donor_vid"])
        k, arm, scheme, repeat = int(r["rotation"]), r["arm"], r["scheme"], int(r["repeat"])
        key = k, arm, scheme, repeat, p
        eligible = boolean(r["eligible"])
        if key in donors or p not in arms[k, arm] or d not in arms[k, arm] or scheme not in SCHEMES or not 0 <= repeat < 32 or int(r["seed"]) != 2026101200 + repeat:
            raise AssertionError("Changed donor role/arm/seed or duplicate")
        group = (lambda q: (q[1], metadata[q]["acquisition_group"])) if scheme == SCHEMES[0] else (lambda q: (q[0], metadata[q]["assigned_valence"]))
        count = sum(decisions[q] and group(q) == group(p) for q in arms[k, arm])
        expected = decisions[p] and count >= 2
        if eligible != expected or eligible and (d == p or not decisions[d] or group(d) != group(p)) or not eligible and d != p:
            raise AssertionError("Derangement/availability/singleton contract violated")
        if int(r["stratum_size"]) != (count if decisions[p] else 0):
            raise AssertionError("Donor stratum size changed")
        donors[key] = d, eligible
        donor_panels[k, arm, scheme, repeat].append((p, d))
    if len(donors) != 53760:
        raise AssertionError("Incomplete donor rectangle")
    for key, mapping in donor_panels.items():
        if Counter(p for p, d in mapping) != Counter(d for p, d in mapping):
            raise AssertionError("Donor assignment is not a permutation")
        k, arm, scheme, repeat = key
        groups = defaultdict(list)
        for p in sorted(arms[k, arm]):
            if decisions[p]:
                group = (p[1], metadata[p]["acquisition_group"]) if scheme == SCHEMES[0] else (p[0], metadata[p]["assigned_valence"])
                groups[group].append(p)
        expected_mapping = {p: p for p in arms[k, arm]}
        rng = np.random.default_rng(2026101200 + repeat)
        for group in sorted(groups, key=lambda value: tuple(str(v) for v in value)):
            members = sorted(groups[group])
            if len(members) == 1:
                continue
            for _ in range(10000):
                permutation = rng.permutation(len(members))
                if np.all(permutation != np.arange(len(members))):
                    break
            else:
                raise AssertionError("Declared deterministic derangement cannot be reproduced")
            expected_mapping.update({p: members[int(permutation[i])] for i, p in enumerate(members)})
        if dict(mapping) != expected_mapping:
            raise AssertionError("Donor mappings do not exactly reproduce declared seeds")
    swapped, swapped_seen = defaultdict(list), set()
    for r in rows(PUBLIC / "swapped_predictions.csv"):
        p = identity(r, metadata)
        k, arm, scheme, anchor, repeat = int(r["rotation"]), r["arm"], r["scheme"], r["anchor"], int(r["repeat"])
        d, eligible = donors[k, arm, scheme, repeat, p]
        key = k, arm, scheme, anchor, repeat, p
        if key in swapped_seen or anchor not in ANCHORS or (r["donor_subject"], int(r["donor_vid"])) != d or boolean(r["eligible"]) != eligible:
            raise AssertionError("Swapped donor/anchor/identity changed")
        swapped_seen.add(key)
        y, value = float(r["target"]), float(r["prediction"])
        close(y, targets[p], "Swap target", 0)
        if boolean(r["eeg_available"]) != decisions[p] or boolean(r["fallback"]) != (not decisions[p]):
            raise AssertionError("Swap availability/fallback changed")
        if not eligible:
            close(value, prediction_map[k, arm, anchor, p]["prediction"], "Unperturbed input")
        if anchor == "eegnet_residual":
            expected = offsets[k, arm, p] + (correction[k, arm, d] if decisions[p] else 0.)
            close(value, np.clip(expected, 0, 7), "Neural donor correction with fixed recipient offset")
        swapped[k, arm, scheme, anchor, repeat].append({**r, "vid": p[1], "target": y, "prediction": value})
    if len(swapped_seen) != 107520:
        raise AssertionError("Incomplete swapped prediction rectangle")
    comparison = load(PUBLIC / "comparison.json")
    acquisition = {"context_panels": [], "pairing_panels": [],
                   "scope": "Fixed acquisition strata; descriptive public-row summaries, no new fitting/arm or independent confirmation."}
    for c in comparison["context_panels"]:
        k, arm = c["rotation"], c["arm"]
        chosen = source_cv["strongest_comparators"][str(k)]["model"]
        if c["source_selected_comparator"] != chosen:
            raise AssertionError("Validation-selected comparator substituted")
        for name in PRIORITY + ANCHORS:
            panel = predictions[k, arm, name]
            check_metric(c[name], panel, "Context/anchor panel")
            groups = sorted({r["acquisition_group"] for r in panel})
            for group in groups:
                acquisition["context_panels"].append({"rotation": k, "arm": arm, "model": name, "acquisition_group": group,
                    **summarize([r for r in panel if r["acquisition_group"] == group])})
        for anchor in ANCHORS:
            close(c["eeg_gain_over_source_selected_comparator"][anchor], score(predictions[k, arm, chosen]) - score(predictions[k, arm, anchor]), "EEG/comparator gain")
    for c in comparison["pairing_panels"]:
        k, arm, scheme, anchor = c["rotation"], c["arm"], c["scheme"], c["anchor"]
        original = predictions[k, arm, anchor]
        check_metric(c["original_full"], original, "Original full panel")
        eligible_pairs = {p for (rk, ra, rs, repeat, p), (_, eligible) in donors.items() if (rk, ra, rs, repeat) == (k, arm, scheme, 0) and eligible}
        if c["eligible_trials"] != len(eligible_pairs) or c["unperturbed_trials"] != len(original) - len(eligible_pairs):
            raise AssertionError("Pairing eligible denominators changed")
        eligible_original = [r for r in original if (r["subject"], r["vid"]) in eligible_pairs]
        if eligible_original:
            check_metric(c["original_eligible"], eligible_original, "Original eligible panel")
        deltas, loss_by_cell = [], defaultdict(list)
        for repeat in range(32):
            panel = swapped[k, arm, scheme, anchor, repeat]
            if {(r["subject"], r["vid"]) for r in panel} != arms[k, arm]:
                raise AssertionError("Changed repeated full rectangle")
            check_metric(c["repeats"][repeat]["full"], panel, "Repeated full panel")
            delta = score(panel) - score(original)
            close(c["repeats"][repeat]["full_delta"], delta, "Repeat delta")
            deltas.append(delta)
            for r in panel:
                p = r["subject"], r["vid"]
                old = prediction_map[k, arm, anchor, p]
                loss_by_cell[p[0], r["family"]].append(abs(r["target"] - r["prediction"]) - abs(old["target"] - old["prediction"]))
        close(c["mean_full_delta"], np.mean(deltas), "Mean repeat losses")
        people, films = sorted({p for p, f in loss_by_cell}), sorted({f for p, f in loss_by_cell})
        matrix = np.array([[np.mean(loss_by_cell[p, f]) for f in films] for p in people])
        rng = np.random.default_rng(425401)
        pd = rng.integers(len(people), size=(4000, len(people)))
        fd = rng.integers(len(films), size=(4000, len(films)))
        interval = np.quantile(matrix[pd[:, :, None], fd[:, None, :]].mean(axis=(1, 2)), [.025, .975])
        for a, b in zip(c["full_panel_crossed_interval"]["descriptive_95_percent_interval"], interval):
            close(a, b, "Crossed mean-recipient-loss interval")
        for group in sorted({r["acquisition_group"] for r in original}):
            subset = [r for r in original if r["acquisition_group"] == group]
            elig = [r for r in subset if (r["subject"], r["vid"]) in eligible_pairs]
            repeat_mae = [score([r for r in swapped[k, arm, scheme, anchor, repeat] if r["acquisition_group"] == group]) for repeat in range(32)]
            acquisition["pairing_panels"].append({"rotation": k, "arm": arm, "scheme": scheme, "anchor": anchor,
                "acquisition_group": group, **summarize(subset), "eligible_trials": len(elig),
                "unperturbed_trials": len(subset) - len(elig), "unavailable_trials": sum(not decisions[r["subject"], r["vid"]] for r in subset),
                "mean_swapped_mae": float(np.mean(repeat_mae)), "mean_loss_delta": float(np.mean(repeat_mae) - score(subset)),
                "repeat_mae": repeat_mae})
    gate = load(PUBLIC / "checkpoint_replay.json")
    if gate["activation_sha256"] != digest(PUBLIC / "activation.json") or gate["plan_sha256"] != digest(PUBLIC / "checkpoint_replay_plan.json") or not gate["passed"] or gate["validation_forward_replay_performed"]:
        raise AssertionError("Changed source fixture replay scope/bindings")
    if len(gate["prediction_checks"]) != 48:
        raise AssertionError("Incomplete source-forward comparison receipt")
    for r in gate["prediction_checks"]:
        close(r["absolute_difference"], abs(r["cached_raw_prediction"] - r["replayed_raw_prediction"]), "Replay absolute difference")
        if r["absolute_difference"] > 1e-5:
            raise AssertionError("Source-forward tolerance exceeded")
    result = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_bindings_checked": len(activation["bindings"]), "ridge_fit_identities_checked": 612,
              "source_prior_constructions_checked": 63, "canonical_targets_checked": 1260,
              "cv_prediction_rows_checked": 21840, "validation_prediction_rows_checked": 5880,
              "donor_rows_checked": 53760, "swapped_prediction_rows_checked": 107520,
              "deterministic_donor_mappings_exactly_reproduced": 53760,
              "source_crossed_cells_with_all_held_categories_absent_in_training": category_missing_cells,
              "neural_swaps_recomputed_from_public_raw_outputs": 53760,
              "source_forward_receipt_checks": 48, "crossed_loss_intervals_recomputed": 24,
              "private_inputs_opened": 0, "original_data_files_opened": 0, "new_fits": 0,
              "confirmation_access": False, "research_question_changed": False,
              "limits": "Public-only numeric replay. Does not independently refit Ridge, read its private features/states, reproduce spectral swap values for switched inputs, or forward-replay validation EEG. Source-forward receipts are checked, not rerun here. Reused development is not independent confirmation.",
              "input_bindings": {p.relative_to(REPO).as_posix(): digest(p) for p in PUBLIC.iterdir() if p.is_file() and p not in outputs}}
    if write:
        for path, value in zip(outputs, (result, acquisition)):
            with path.open("x", encoding="utf-8") as f:
                f.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="Write first audit receipts; refuses overwrite")
    action.add_argument("--check", action="store_true", help="Repeat public numerical checks without writing")
    args = parser.parse_args()
    print(json.dumps(audit(write=args.run)), flush=True)
