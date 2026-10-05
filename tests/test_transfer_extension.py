import numpy as np
import pandas as pd
import pytest

from gruxnet import transfer_controls as core
from gruxnet.transfer_extension import participant_folds, target_mapping


@pytest.mark.parametrize("count", [15, 28, 32])
def test_complete_participant_rotation(count):
    subjects = [f"S{index:02d}" for index in range(count)]
    folds = participant_folds(subjects)
    tested = []
    for fold in folds:
        parts = [set(fold[key]) for key in ["train", "validation", "test"]]
        assert set.union(*parts) == set(subjects)
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        tested.extend(fold["test"])
    assert sorted(tested) == subjects
    assert max(len(f["test"]) for f in folds) - min(len(f["test"]) for f in folds) <= 1
    assert participant_folds(list(reversed(subjects))) == folds


def test_target_slot_and_balanced_exposures_restore_after_failure():
    old_conditions, old_index = dict(core.CONDITIONS), dict(core.DATASET_INDEX)
    table = pd.DataFrame([{"dataset": name, "label": label} for name in ["DEAP", "GAMEEMO", "SEEDIV"]
                          for label in [0, 1] for _ in range(4)])
    with pytest.raises(RuntimeError):
        with target_mapping("DEAP"):
            assert core.DATASET_INDEX["DEAP"] == 0
            assert core.CONDITIONS["joint"][0] == "DEAP"
            single = core.balanced_batches(table, ["DEAP"], 2, 42, 0)
            joint = core.balanced_batches(table, core.CONDITIONS["joint"], 6, 42, 0)
            np.testing.assert_array_equal(single[:, :30].ravel(), joint[:, :10].ravel())
            np.testing.assert_array_equal(single[:, 30:].ravel(), joint[:, 10:20].ravel())
            raise RuntimeError("restore mapping even if a run fails")
    assert core.CONDITIONS == old_conditions and core.DATASET_INDEX == old_index
