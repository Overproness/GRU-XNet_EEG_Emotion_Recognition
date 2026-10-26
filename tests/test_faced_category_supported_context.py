"""Synthetic category-support, source-selection and cached-identity tests."""
from contextlib import nullcontext
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gruxnet.faced_classical_contract import CATEGORIES, ALPHAS
from gruxnet.faced_development_diagnosis import COMPARATOR_PRIORITY
from gruxnet.faced_category_supported_context import (
    FIT_CAP, ContextFitLedger, cached_development_inputs, person_folds,
    preserve_partial, select_alpha, strongest_comparator,
)


def fixture():
    people = [f"source-{i:03d}" for i in range(70)]
    clips = list(range(1, 9))
    pairs = [(p, v) for p in people for v in clips]
    metadata = {p: {"role": "source", "assigned_category": CATEGORIES[p[1] - 1], "family": f"film-{p[1]}"} for p in pairs}
    folds = [people[i * 14:(i + 1) * 14] for i in range(5)]
    return pairs, metadata, folds


def test_source_person_folds_preserve_all_categories_and_titles():
    pairs, metadata, folds = fixture()
    result = person_folds(pairs, metadata, folds)
    assert len(result) == 5
    held = []
    for fold in result:
        assert len(fold["training"]) == 448 and len(fold["held"]) == 112
        assert not {p[0] for p in fold["training"]} & {p[0] for p in fold["held"]}
        assert {metadata[p]["assigned_category"] for p in fold["training"]} == set(CATEGORIES)
        assert {p[1] for p in fold["training"]} == set(range(1, 9))
        held.extend(fold["held"])
    assert sorted(held) == sorted(pairs)


def test_person_folds_reject_changed_categories_or_protected_role():
    pairs, metadata, folds = fixture()
    changed = {p: {**m, "assigned_category": "Joy"} for p, m in metadata.items()}
    with pytest.raises(PermissionError, match="eight categories"):
        person_folds(pairs, changed, folds)
    metadata[pairs[0]]["role"] = "confirmation"
    with pytest.raises(PermissionError):
        person_folds(pairs, metadata, folds)


def test_person_folds_reject_duplicate_or_missing_source_person():
    pairs, metadata, folds = fixture()
    folds[0][0] = folds[1][0]
    with pytest.raises(PermissionError):
        person_folds(pairs, metadata, folds)
    with pytest.raises(PermissionError):
        person_folds(pairs[:-1], metadata, folds)


def test_alpha_tie_prefers_small_alpha_even_when_mse_is_worse():
    candidates = [{"alpha": a, "participant_title_mae": 1., "trial_mse": 1000. / a} for a in ALPHAS]
    assert select_alpha(candidates) == .1
    candidates[3]["participant_title_mae"] = .9
    assert select_alpha(candidates) == 100.
    with pytest.raises(PermissionError):
        select_alpha(candidates[:-1])


def test_strongest_comparator_uses_source_scores_then_fixed_priority():
    scores = {name: 1. for name in COMPARATOR_PRIORITY}
    assert strongest_comparator(scores) == COMPARATOR_PRIORITY[0]
    scores["category_median"] = .8
    assert strongest_comparator(scores) == "category_median"
    with pytest.raises(PermissionError):
        strongest_comparator({"category_only": 1.})


class FloatBomb:
    def __float__(self):
        raise AssertionError("Protected cached rating interpreted")


def test_completed_csv_guard_rejects_protected_before_target_float(monkeypatch):
    import gruxnet.faced_category_supported_context as module
    row = {"subject": "protected", "vid": "1", "family": "F", "target": FloatBomb()}
    monkeypatch.setattr(Path, "open", lambda *a, **kw: nullcontext(io.StringIO("")))
    monkeypatch.setattr(module.csv, "DictReader", lambda f: iter([row]))
    with pytest.raises(PermissionError):
        cached_development_inputs({}, {("allowed", 1): {"family": "F"}}, {})


def test_context_fit_cap_stops_before_estimator_fit():
    ledger = object.__new__(ContextFitLedger)
    ledger.records = [None] * FIT_CAP
    with pytest.raises(PermissionError, match="156-fit"):
        ledger.fit(None, None, None, None)


def test_partial_context_attempt_retains_states_and_target_rows(monkeypatch):
    import gruxnet.faced_category_supported_context as module
    observed = {}
    monkeypatch.setattr(module, "save", lambda path, data: observed.update({Path(path).name: data}))
    monkeypatch.setattr(module, "_write_csv", lambda path, data: observed.update({Path(path).name: list(data)}))
    ledger = SimpleNamespace(records=[{"index": 0, "source_pairs": [("source", 1)]}])
    preserve_partial("synthetic", ledger, [{"subject": "source", "target": 1}], [], [{"source_pairs": [("source", 1)]}])
    assert set(observed) == {"partial_model_manifest.json", "partial_cv_predictions.csv", "partial_prior_manifest.json"}
    assert observed["partial_model_manifest.json"]["resumption_authorized"] is False
