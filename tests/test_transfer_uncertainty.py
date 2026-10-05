import numpy as np

from scripts.analyze_transfer_targets import comparison, pooled_ba


def test_pooled_counts_preserve_class_balance_with_single_class_participants():
    counts = np.array([[2, 0, 0, 0], [0, 0, 3, 1]])
    assert np.isnan(pooled_ba(counts)).all()
    assert pooled_ba(counts.sum(0)) == .625
    paired = comparison(counts[None], counts[None], np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
    assert paired["class_deficient_draws_excluded"] == 2
    assert paired["valid_draws"] == 2
    assert paired["mean_BA_difference"] == 0
    assert paired["paired_subject_percentile_95"] == [0., 0.]
