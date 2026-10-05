import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gruxnet.data import aligned_signal, load_provenance, numeric_eeg_keys, sam_rating, sha256, valence_label, verify_deap_labels
from gruxnet.model import CompactGRUXNet, time_frequency
from gruxnet.prepare import preprocess_signal
from gruxnet.provenance import compare_subject
from gruxnet.splits import cap_windows, make_split, sampling_weights, validate_split
from gruxnet.train import aggregate_trials, metrics
from gruxnet.baseline import log_bandpower


def fixture_manifest():
    rows = []
    for dataset in ["DEAP", "GAMEEMO", "SEEDIV"]:
        for subject in range(10):
            sid = f"{dataset}:S{subject:02d}"
            for label in [0, 1]:
                for window in range(3 + label):
                    tid = f"{sid}:T{label}"
                    rows.append(dict(dataset=dataset, subject_id=sid, trial_id=tid, label=label,
                                     sample_id=f"{tid}:W{window}", start_sample=window*512))
    return pd.DataFrame(rows)


def test_seed_trial_order_and_valence():
    keys = numeric_eeg_keys([f"abc_eeg{i}" for i in range(24, 0, -1)] + ["__header__"])
    assert keys[1] == "abc_eeg2" and keys[9] == "abc_eeg10"
    assert [valence_label(i, "SEEDIV") for i in range(4)] == [None, 0, 0, 1]
    assert [valence_label(i, "GAMEEMO") for i in [4, 5, 6]] == [0, None, 1]
    with pytest.raises(ValueError):
        numeric_eeg_keys(keys[:-1])


def test_deap_recovery_detects_inverse_and_rejects_wrong_video_order():
    reference = np.array([[7.71, 7.6, 6.9, 7.83], [8.1, 7.31, 7.28, 8.47]])
    actual = reference.copy()
    actual[0, :2] = 9 - actual[0, :2]
    assert verify_deap_labels(actual, reference) == [1, 1, 0, 0]
    with pytest.raises(ValueError, match="video order"):
        verify_deap_labels(actual, reference[::-1])
    actual[0, 0] = 4.
    with pytest.raises(ValueError, match="Unexplained"):
        verify_deap_labels(actual, reference)


def test_provenance_snapshot_changes_hash_when_source_version_changes(pdf_workspace):
    original = Path(__file__).resolve().parents[1] / "dataset_sources.json"
    snapshot = load_provenance(original)
    assert snapshot["catalog_sha256"] == sha256(original)
    changed = json.loads(original.read_text(encoding="utf-8"))
    changed["datasets"]["DEAP"]["kaggle_version"] = 2
    new_catalog = pdf_workspace / "sources.json"
    new_catalog.write_text(json.dumps(changed), encoding="utf-8")
    updated = load_provenance(new_catalog)
    assert updated["catalog_sha256"] != snapshot["catalog_sha256"]
    assert snapshot["catalog"]["datasets"]["DEAP"]["kaggle_version"] == 1


def test_provenance_rejects_incomplete_or_conflicting_sources(pdf_workspace):
    original = Path(__file__).resolve().parents[1] / "dataset_sources.json"
    invalid = json.loads(original.read_text(encoding="utf-8"))
    invalid["datasets"].pop("SEEDIV")
    path = pdf_workspace / "sources.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_provenance(path)
    invalid = json.loads(original.read_text(encoding="utf-8"))
    invalid["datasets"]["DEAP"]["kaggle_url"] = invalid["bundle"]["kaggle_url"]
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="matching Kaggle URL"):
        load_provenance(path)


def test_reference_comparison_separates_pickle_label_changes_from_eeg_changes():
    ratings = np.array([[7.71, 7.6, 6.9, 7.83], [3.2, 6.5, 4.1, 5.7]])
    signals = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    reference = {"data": signals.copy(), "labels": ratings.copy()}
    local = {"data": signals.copy(), "labels": ratings.copy()}
    local["labels"][0, :2] = 9 - local["labels"][0, :2]
    comparison = compare_subject(local, reference, ratings)
    assert comparison["signal_values_identical"]
    assert comparison["local_signal_sha256"] == comparison["reference_signal_sha256"]
    assert comparison["local_rating_differences_from_spreadsheet"] == [1, 1, 0, 0]
    assert comparison["reference_rating_differences_from_spreadsheet"] == [0, 0, 0, 0]
    local["data"][1, 2, 3] += .125
    comparison = compare_subject(local, reference, ratings)
    assert not comparison["signal_values_identical"]
    assert comparison["maximum_absolute_signal_difference"] == .125
    assert comparison["local_signal_sha256"] != comparison["reference_signal_sha256"]
    reference["data"] = reference["data"][:, :, :-1]
    with pytest.raises(ValueError, match="shapes"):
        compare_subject(local, reference, ratings)


def test_sam_graphics_include_spanning_ellipse_cross_and_underline(pdf_workspace):
    import fitz
    tmp_path = pdf_workspace
    p = tmp_path / "rating.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595.32, height=841.92)
    page.draw_oval(fitz.Rect(328, 485, 367, 635))
    doc.save(p)
    doc.close()
    result = sam_rating(p)
    assert result["valence"] == result["arousal"] == 6
    p = tmp_path / "cross.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595.32, height=841.92)
    page.draw_line(fitz.Point(150, 480), fitz.Point(163, 523))
    page.draw_line(fitz.Point(154, 526), fitz.Point(159, 482))
    page.draw_line(fitz.Point(373, 630), fitz.Point(424, 632))
    doc.save(p)
    doc.close()
    assert sam_rating(p)["valence"] == 2 and sam_rating(p)["arousal"] == 7


def test_named_channel_alignment_not_positional_padding():
    signal = np.array([[11, 12], [21, 22]], dtype=float)
    aligned, mask = aligned_signal(signal, ["F4", "AF3"], ["AF3", "F3", "F4"])
    np.testing.assert_array_equal(aligned, [[21, 22], [0, 0], [11, 12]])
    assert mask.tolist() == [True, False, True]


def test_resampling_preserves_duration_and_physical_frequency():
    t = np.arange(2000) / 200
    signal = np.sin(2*np.pi*10*t)[None]
    resampled, mask = preprocess_signal(signal, 200, ["F3"], ["F3", "F4"])
    assert resampled.shape == (2, 1280) and mask.tolist() == [True, False]
    peak = np.fft.rfftfreq(1280, 1/128)[np.abs(np.fft.rfft(resampled[0])).argmax()]
    assert peak == 10


def test_bandpower_reference_retains_physical_band_and_amplitude():
    t = np.arange(512) / 128
    alpha = np.sin(2*np.pi*10*t)
    features = log_bandpower(np.stack([alpha, 2*alpha])[:, None, :])
    assert features.shape == (2, 4)
    assert features[0, 1] > features[0, 0] + 10
    np.testing.assert_allclose(features[1, 1] - features[0, 1], np.log(4), atol=1e-6)


@pytest.mark.parametrize("protocol,target,subject", [("subject", None, None), ("lodo", "GAMEEMO", None), ("loso", None, "DEAP:S01")])
def test_subject_and_trial_disjoint_protocols(protocol, target, subject):
    frame = fixture_manifest()
    split, _ = make_split(frame, protocol, target=target, test_subject=subject)
    assert (split.groupby("subject_id").split.nunique() == 1).all()
    assert (split.groupby("trial_id").split.nunique() == 1).all()
    if protocol == "lodo":
        assert set(split[split.split == "test"].dataset) == {target}
        assert target not in set(split[split.split != "test"].dataset)
    if protocol == "loso":
        assert split[split.split == "test"].subject_id.unique().tolist() == [subject]
    capped = cap_windows(split, 2)
    assert capped.groupby("trial_id").size().eq(2).all()
    validate_split(capped)
    broken = split.copy()
    broken.loc[0, "split"] = "test" if broken.loc[0, "split"] != "test" else "train"
    with pytest.raises(ValueError, match="Leakage"):
        validate_split(broken)


def test_balanced_sampling_removes_dataset_and_trial_duration_dominance():
    frame = fixture_manifest()
    frame["weight"] = sampling_weights(frame)
    np.testing.assert_allclose(frame.groupby(["dataset", "label"]).weight.sum().values, np.ones(6)/6)
    np.testing.assert_allclose(frame.groupby("trial_id").weight.sum().values, np.ones(60)/60)


def test_missing_channel_values_cannot_affect_predictions():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = CompactGRUXNet(channels=3).eval()
    features = torch.randn(2, 3, 37, 13)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    changed = features.clone()
    changed[~mask] = 1000
    with torch.no_grad():
        torch.testing.assert_close(model(features, mask), model(changed, mask), rtol=0, atol=0)


def test_frequency_identity_is_retained_before_temporal_recurrence():
    model = CompactGRUXNet(channels=14)
    assert model.fusion[0].in_features == 14 * 32 * 4
    assert sum(p.numel() for p in model.parameters()) == 537442
    legacy = CompactGRUXNet(channels=14, frequency_pooling="mean")
    assert legacy.fusion[0].in_features == 14 * 32


def test_stft_uses_actual_time_and_evaluation_is_deterministic():
    x = torch.randn(2, 3, 512)
    mask = torch.ones(2, 3, dtype=torch.bool)
    f1, m1 = time_frequency(x, mask, augment=False)
    f2, m2 = time_frequency(x, mask, augment=False)
    assert f1.shape == (2, 3, 37, 13)
    torch.testing.assert_close(f1, f2, rtol=0, atol=0)
    for _ in range(10):
        f, m = time_frequency(x, mask, augment=True)
        assert m.any(dim=1).all() and torch.isfinite(f).all()
        assert not torch.equal(f, f1)


def test_metrics_aggregate_correlated_windows_into_trials():
    frame = pd.DataFrame([
        dict(dataset="DEAP", subject_id="DEAP:S01", trial_id="a", sample_id="a1", label=0, positive_probability=.1),
        dict(dataset="DEAP", subject_id="DEAP:S01", trial_id="a", sample_id="a2", label=0, positive_probability=.8),
        dict(dataset="DEAP", subject_id="DEAP:S02", trial_id="b", sample_id="b1", label=1, positive_probability=.9),
    ])
    result, trials = metrics(frame, bootstrap=10)
    assert result["window"]["n"] == 3 and result["trial"]["n"] == 2
    assert result["trial"]["accuracy"] == 1 and result["trial"]["confusion_matrix"] == [[1, 0], [0, 1]]
    broken = frame.copy()
    broken.loc[1, "label"] = 1
    with pytest.raises(ValueError):
        aggregate_trials(broken)
