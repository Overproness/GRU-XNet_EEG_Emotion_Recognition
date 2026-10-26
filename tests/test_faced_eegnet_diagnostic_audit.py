"""Synthetic fixtures for the independent fixed-grid evidence auditor."""
import copy
import math
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.audit_faced_eegnet_diagnostic import (
    ARMS, Axes, CONTROLS, EPOCHS, RATES, READOUTS, SEEDS,
    audit_results, balanced_mae, chosen_schedule, context_vector, direct_ridge,
    identity_guard, learning_flags, source_offsets,
    verify_cached_projection,
)


def fixture():
    source_people = [f"s{i}" for i in range(10)]
    val_people = ["v0", "v1"]
    metadata = {}; decisions = {}; targets = {}; arms = {}
    categories = ("Amusement", "Anger", "Disgust", "Fear", "Inspiration", "Joy", "Sadness", "Tenderness")
    for i, person in enumerate(source_people+val_people):
        for clip in range(1, 15):
            p = person, clip
            metadata[p] = {"family": f"film{clip}", "acquisition_group": ("legacy_1000", "legacy_250", "modern_1000")[i%3], "category": categories[(clip-1)%8], "coarse": "Positive" if clip%2 else "Negative", "duration": 40.+clip, "trial": ((clip+i)%28)+1}
            decisions[p] = {"eeg_available": p != ("v0", 9)}
            targets[p] = 2.+clip/20+(i%5)*.3
    material_sets = [set(range(1, 9)), set(range(7, 15)), {1, 2, 3, 4, 9, 10, 11, 12}]
    for r, clips in enumerate(material_sets):
        source = {(p, v) for p in source_people for v in clips}
        arms[r] = {"source_fit": source, "source_inference": source, "validation_unseen": {(p, v) for p in val_people for v in set(range(1, 15))-clips}, "validation_familiar": {(p, v) for p in val_people for v in clips}}
    axes = Axes(arms, metadata, decisions, [source_people[2*i:2*i+2] for i in range(5)])
    offsets, independents = source_offsets(axes, targets)
    records = []; checkpoint_rows = []; keyed = {}
    selected_rate = .001; selected_epoch = 60
    for r in range(3):
        for head in READOUTS:
            for rate in RATES if r == 0 else (selected_rate,):
                for seed in SEEDS:
                    key = r, head, rate, seed
                    record = {"rotation": r, "readout": head, "learning_rate": rate, "seed": seed, "status": "completed", "history": [{"epoch": e, "normalized_training_mse": .01} for e in range(1, 121)], "checkpoints": {}}
                    source = arms[r]["source_fit"]
                    record["starting_control_mse"] = math.fsum((targets[p]-offsets[r][head]["fit"][p])**2 for p in source)/len(source)
                    for epoch in EPOCHS:
                        panels = {}
                        for arm in ARMS:
                            panel = {}
                            for p in sorted(arms[r][arm]):
                                if arm.startswith("source"):
                                    raw = targets[p]+.01
                                    if arm == "source_inference": raw += offsets[r][head]["inference"][p]-offsets[r][head]["fit"][p]
                                elif not decisions[p]["eeg_available"]:
                                    raw = offsets[r][head]["inference"][p]
                                else:
                                    error = .05 + (.1 if rate != selected_rate else 0) + (.1 if epoch != selected_epoch else 0)
                                    raw = targets[p]+error+(.001 if seed == SEEDS[0] else -.001)
                                panel[p] = raw
                                checkpoint_rows.append({"rotation": r, "readout": head, "learning_rate": rate, "seed": seed, "epoch": epoch, "arm": arm, "subject": p[0], "vid": p[1], "family": metadata[p]["family"], "acquisition_group": metadata[p]["acquisition_group"], "target": targets[p], "prediction_raw": raw, "eeg_available": decisions[p]["eeg_available"], "fallback": not decisions[p]["eeg_available"]})
                            panels[arm] = panel; keyed[*key, epoch, arm] = panel
                        record["checkpoints"][str(epoch)] = {"training_target_source_mse": math.fsum((targets[p]-panels["source_fit"][p])**2 for p in source)/len(source), "source_inference_mse": math.fsum((targets[p]-panels["source_inference"][p])**2 for p in source)/len(source), "state_dict_sha256": "a"*64}
                    records.append(record)
    ensemble_rows = []; control_rows = []; control_panels = {}; selections = {}
    for head in READOUTS:
        selections[head] = chosen_schedule(keyed, axes, targets, head)[0]
    for r in range(3):
        unseen = arms[r]["validation_unseen"]
        for head in READOUTS:
            for arm in ARMS:
                for p in sorted(arms[r][arm]):
                    raw = math.fsum(keyed[r, head, selected_rate, seed, selected_epoch, arm][p] for seed in SEEDS)/2
                    ensemble_rows.append({"rotation": r, "readout": head, "selected_learning_rate": selected_rate, "selected_epoch": selected_epoch, "arm": arm, "subject": p[0], "vid": p[1], "family": metadata[p]["family"], "acquisition_group": metadata[p]["acquisition_group"], "target": targets[p], "prediction": min(7, max(0, raw)), "eeg_available": decisions[p]["eeg_available"], "fallback": not decisions[p]["eeg_available"]})
        for model in CONTROLS:
            panel = independents[r][model] if model != "context_kernel" else {p: targets[p]+.02 for p in unseen}
            control_panels[r, model] = panel
            for p in sorted(unseen):
                control_rows.append({"rotation": r, "model": model, "subject": p[0], "vid": p[1], "family": metadata[p]["family"], "target": targets[p], "prediction": panel[p]})
    outputs = []
    for r in range(3):
        unseen = arms[r]["validation_unseen"]
        control_errors = {model: balanced_mae(unseen, control_panels[r, model], targets, metadata) for model in CONTROLS}
        strongest = min(CONTROLS, key=lambda m: (control_errors[m], CONTROLS.index(m)))
        for head in READOUTS:
            members = [f for f in records if f["rotation"] == r and f["readout"] == head and f["learning_rate"] == selected_rate]
            seed_errors = [balanced_mae(unseen, keyed[r, head, selected_rate, seed, selected_epoch, "validation_unseen"], targets, metadata) for seed in SEEDS]
            flags = learning_flags(members[0]["starting_control_mse"], [f["checkpoints"]["60"]["training_target_source_mse"] for f in members], [f["checkpoints"]["120"]["training_target_source_mse"] for f in members], seed_errors)
            values = {(row["subject"], row["vid"]): row["prediction"] for row in ensemble_rows if row["rotation"] == r and row["readout"] == head and row["arm"] == "validation_unseen"}
            error = balanced_mae(unseen, values, targets, metadata)
            outputs.append({"rotation": r, "readout": head, "schedule": selections[head], "learning_flags": flags, "strongest_context": strongest, "control_maes": control_errors, "joint_mae": error, "gain": control_errors[strongest]-error})
    execution = {"fits": 20, "selection": selections, "rotations": outputs, "engineering_leads": {head: False for head in READOUTS}, "confirmation_access": False, "research_question_changed": False}
    return axes, checkpoint_rows, ensemble_rows, {"fits": records}, control_rows, execution


class IndependentDiagnosticAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixture()

    def test_complete_synthetic_fixed_grid(self):
        result = audit_results(*self.fixture)
        self.assertTrue(result["passed"])
        self.assertEqual(result["fits_audited"], 20)
        self.assertEqual(result["selection"]["eeg_only"]["epoch"], 60)
        self.assertEqual(result["selection"]["context_residual"]["learning_rate"], .001)
        self.assertFalse(any(result["engineering_leads"].values()))

    def test_protected_identity_rejected_before_poison_target(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        changed = copy.deepcopy(checkpoints)
        changed[-1]["subject"] = "confirmation-person"
        changed[0]["target"] = "this must never be numerically parsed"
        with self.assertRaises(PermissionError):
            audit_results(axes, changed, ensembles, fits, controls, execution)

    def test_all_csvs_guarded_before_any_target(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        poison = copy.deepcopy(checkpoints); poison[0]["target"] = "poison"
        bad = copy.deepcopy(controls); bad[-1]["vid"] = 1000
        with self.assertRaises(PermissionError):
            audit_results(axes, poison, ensembles, fits, bad, execution)

    def test_missing_seed_checkpoint_fails_complete_panel(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        with self.assertRaisesRegex(ValueError, "Incomplete first selection panel"):
            audit_results(axes, checkpoints[1:], ensembles, fits, controls, execution)

    def test_clipping_is_applied_after_averaging(self):
        axes, checkpoints, ensembles, fits, controls, execution = copy.deepcopy(self.fixture)
        for row in checkpoints:
            if row["rotation"] == 1 and row["readout"] == "eeg_only" and row["epoch"] == 60 and row["arm"] == "source_inference" and row["subject"] == "s0" and row["vid"] == 7:
                row["prediction_raw"] = -2. if row["seed"] == SEEDS[0] else 4.
        for row in ensembles:
            if row["rotation"] == 1 and row["readout"] == "eeg_only" and row["arm"] == "source_inference" and row["subject"] == "s0" and row["vid"] == 7:
                row["prediction"] = 2.  # mean of individually clipped outputs; correct value is 1.
        with self.assertRaisesRegex(ValueError, "Post-ensemble clipping"):
            audit_results(axes, checkpoints, ensembles, fits, controls, execution)

    def test_unseen_schedule_transfer_cannot_be_retuned(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        changed = copy.deepcopy(ensembles); changed[-1]["selected_epoch"] = 120
        with self.assertRaisesRegex(ValueError, "Transferred ensemble epoch"):
            audit_results(axes, checkpoints, changed, fits, controls, execution)

    def test_independent_svd_matches_established_ridge(self):
        axes = self.fixture[0]; training = sorted(axes.arms[0]["source_fit"]); evaluation = sorted(axes.arms[0]["validation_unseen"])
        targets = {p: 2.+p[1]/20+(int(p[0][1:])%5)*.3 for p in axes.allowed}
        x = np.stack([context_vector(axes.metadata[p]) for p in training]); xh = np.stack([context_vector(axes.metadata[p]) for p in evaluation]); y = [targets[p] for p in training]
        for alpha in (.1, 100.):
            scaler = StandardScaler(); transformed = scaler.fit_transform(x)
            reference = np.clip(Ridge(alpha=alpha, solver="svd").fit(transformed, y).predict(scaler.transform(xh)), 0, 7)
            obtained = direct_ridge(training, evaluation, targets, axes.metadata, alpha)
            np.testing.assert_allclose([obtained[p] for p in evaluation], reference, atol=1e-12, rtol=0)

    def test_learning_flags_strict_boundaries_and_nonfinite(self):
        self.assertEqual(learning_flags(1., [.8, .8], [.8, .8], [0., .15]), [])
        self.assertIn("insufficient_source_learning", learning_flags(1., [.81, .81], [.81, .81], [0., 0.]))
        self.assertIn("budget_limited_convergence", learning_flags(1., [1., 1.], [.89, .89], [0., 0.]))
        self.assertEqual(learning_flags(1., [1., 1.], [float("nan"), 1.], [0., 0.]), ["invalid_or_nonfinite_learning_metrics"])

    def test_strongest_control_guard_cannot_use_weaker_context(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        altered = copy.deepcopy(execution); altered["rotations"][0]["strongest_context"] = "context_only"
        with self.assertRaisesRegex(ValueError, "Learning/comparator selection"):
            audit_results(axes, checkpoints, ensembles, fits, controls, altered)

    def test_unbounded_training_mse_is_independently_recomputed(self):
        axes, checkpoints, ensembles, fits, controls, execution = self.fixture
        altered = copy.deepcopy(fits); altered["fits"][0]["checkpoints"]["120"]["training_target_source_mse"] = 0.
        with self.assertRaisesRegex(ValueError, "Unbounded training-target MSE"):
            audit_results(axes, checkpoints, ensembles, altered, controls, execution)

    def test_exact_old_control_projection_after_identity_guard(self):
        axes, checkpoints, _, _, controls, _ = self.fixture
        quality = []; representation = []
        for row in controls:
            old = dict(row, arm="validation_unseen", alpha=(.1, 100., 100.)[row["rotation"]] if row["model"] == "context_only" else .1)
            (representation if row["model"] == "context_kernel" else quality).append(old)
        result = verify_cached_projection(axes, quality, representation, controls, checkpoints)
        self.assertTrue(result["exact_targets_and_all_five_control_predictions_reused"])
        bad = copy.deepcopy(representation); bad[0]["prediction"] += .01
        with self.assertRaisesRegex(ValueError, "exactly reused"):
            verify_cached_projection(axes, quality, bad, controls, checkpoints)
        poison = copy.deepcopy(quality); poison[0]["target"] = "must never convert"
        protected = copy.deepcopy(representation); protected[-1]["subject"] = "confirmation-person"
        with self.assertRaises(PermissionError):
            verify_cached_projection(axes, poison, protected, controls, checkpoints)


if __name__ == "__main__":
    unittest.main()
