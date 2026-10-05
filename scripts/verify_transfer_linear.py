"""Refit the anchored linear controls and compare every candidate and prediction."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.data import sha256, write_json
from gruxnet.transfer_controls import linear_controls, load_inputs


def compare_nested(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            compare_nested(a, e)
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    else:
        assert actual == expected


def verify(output):
    config = json.loads((output / "config.json").read_text())
    root = Path(__file__).resolve().parents[1]
    for name, expected in config["source_sha256"].items():
        assert sha256(root / "gruxnet" / name) == expected, name
    x, target, z, sources, binding = load_inputs(Path(config["pack"]), Path(config["common_cache"]))
    assert binding == config["input_binding"]
    folds = json.loads((output / "folds.json").read_text())
    replay = output / "linear_replay"
    replay.mkdir(exist_ok=False)
    actual = linear_controls(x, target, z, sources, folds, replay)
    expected = json.loads((output / "linear_comparison.json").read_text())
    compare_nested(actual, expected)
    saved = pd.read_csv(output / "linear_trial_predictions.csv")
    recreated = pd.read_csv(replay / "linear_trial_predictions.csv")
    pd.testing.assert_frame_equal(saved.drop(columns="positive_probability"),
                                  recreated.drop(columns="positive_probability"))
    np.testing.assert_allclose(saved.positive_probability, recreated.positive_probability,
                               rtol=1e-8, atol=1e-10)
    candidates = sum(len(f["validation_candidates"]) for r in expected.values() for f in r["folds"])
    selected = sum(len(r["folds"]) for r in expected.values())
    assert candidates == 80 and selected == 20 and len(saved) == 3240
    result = {"passed": True, "validation_candidate_fits_reproduced": candidates,
              "selected_models_reproduced": selected, "test_probabilities_reproduced": len(saved),
              "input_binding_checked": True, "all_candidate_and_aggregate_metrics_reproduced": True,
              "predictions_sha256": sha256(output / "linear_trial_predictions.csv"),
              "comparison_sha256": sha256(output / "linear_comparison.json"),
              "verifier_sha256": sha256(Path(__file__)),
              "scope": "Refit all declared linear validation candidates from retained derived trial features. This does not regenerate original waveform features."}
    write_json(output / "linear_verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
