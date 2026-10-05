"""Check physical features and the native task's participant and electrode semantics."""
import numpy as np
import pandas as pd

from gruxnet.data import COMMON_CHANNELS
from scripts.native_seediv_diagnostic import make_folds, trial_features, view
from scripts.joint_seediv_diagnostic import balanced_weights


def test_bandpower_tracks_alpha_and_discards_incomplete_tail():
    time = np.arange(1024) / 128
    signal = np.sin(2*np.pi*10*time)[None]
    features, n = trial_features(signal)
    assert n == 2 and np.argmax(features) == 1
    doubled, _ = trial_features(2*signal)
    np.testing.assert_allclose(doubled[1]-features[1], 2*np.log(2), atol=1e-6)
    tailed, count = trial_features(np.concatenate([signal, np.full((1, 200), 1e8)], axis=1))
    assert count == n
    np.testing.assert_array_equal(tailed, features)


def test_folds_hold_out_every_participant_once():
    subjects = [f"SEEDIV:S{i:02d}" for i in range(1,16)]
    folds = make_folds(subjects)
    assert sorted(s for fold in folds for s in fold["test"]) == subjects
    for fold in folds:
        assert [len(fold[p]) for p in ["train", "validation", "test"]] == [9,3,3]
        assert not set(fold["train"]) & set(fold["validation"])
        assert not set(fold["train"]) & set(fold["test"])
        assert not set(fold["validation"]) & set(fold["test"])


def test_common_montage_uses_names_and_binary_view_excludes_neutral():
    channels = list(reversed(COMMON_CHANNELS)) + [f"extra{i}" for i in range(48)]
    x = np.tile(np.arange(248).reshape(1,-1), (4,1))
    table = pd.DataFrame({"original_label": [0,1,2,3], "trial_id": ["neutral", "sad", "fear", "happy"]})
    selected, labels, trials = view(x, table, channels, "common14", "binary")
    assert selected.shape == (3,56)
    np.testing.assert_array_equal(selected[0,:4], [52,53,54,55])
    assert labels.tolist() == [0,0,1]
    assert trials.trial_id.tolist() == ["sad","fear","happy"]
    native, native_labels, _ = view(x, table, channels, "native62", "native4")
    assert native.shape == (4,248) and native_labels.tolist() == [0,1,2,3]


def test_pooling_weights_balance_datasets_and_classes_despite_trial_counts():
    table = pd.DataFrame({"dataset": ["DEAP"]*5 + ["GAMEEMO"]*6,
                          "label": [0,0,0,0,1] + [0,0,1,1,1,1]})
    table["weight"] = balanced_weights(table)
    totals = table.groupby(["dataset","label"]).weight.sum().to_numpy()
    np.testing.assert_allclose(totals, np.repeat(len(table)/4, 4))
    np.testing.assert_allclose(table.weight.sum(), len(table))
