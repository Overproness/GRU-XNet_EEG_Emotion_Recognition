import numpy as np
import pandas as pd
from gruxnet.material_controls import annotate,assignment,masks,correctness,coverage
from gruxnet.session_controls import paired_batches
from gruxnet.temporal_controls import make_folds
from test_session_controls import cohort


def test_equal_training_people_and_labels_and_identical_unseen_validation_test():
    table=annotate(cohort()); fold=make_folds(table.subject_id.tolist())[0]
    for session in (1,2,3):
        for rotation in (0,1,2):
            a=masks(table,fold,session,rotation,"exposed"); b=masks(table,fold,session,rotation,"unexposed")
            for part in ("validation","test"): np.testing.assert_array_equal(a[part],b[part])
            assert len(a["train"])==len(b["train"])==108
            for idx in (a,b):
                training=table.iloc[idx["train"]]
                assert training.groupby(["subject_id","original_label"]).size().eq(3).all()
                assert set(training.session)=={session}
                assert not set(training.subject_id)&set(fold["test"])
                assert not set(table.iloc[idx["validation"]].material_key)&set(training.material_key)
            assert set(table.iloc[a["test"]].material_key).issubset(set(table.iloc[a["train"]].material_key))
            assert not set(table.iloc[b["test"]].material_key)&set(table.iloc[b["train"]].material_key)
            assert len(set(table.iloc[a["train"]].material_key)&set(table.iloc[b["train"]].material_key))==4


def test_all_rotations_cover_every_trial_once_and_pair_training_draws():
    table=annotate(cohort()); folds=make_folds(table.subject_id.tolist()); rows=[]
    for session in (1,2,3):
        for rotation in (0,1,2):
            for fold in folds:
                signatures=[]
                for arm in ("exposed","unexposed"):
                    idx=masks(table,fold,session,rotation,arm)
                    _,sig=paired_batches(table,idx["train"],42+1000*fold["fold"]+10000*rotation,updates=8)
                    signatures.append(sig)
                    part=table.iloc[idx["test"]].copy(); part["model"],part["arm"],part["seed"]="test",arm,42; rows.append(part)
                assert signatures[0]==signatures[1]
    combined=pd.concat(rows,ignore_index=True); coverage(combined)
    for _,part in combined.groupby("arm"): assert set(part.trial_id)==set(table.trial_id)


def test_rank_assignment_depends_on_material_not_participant_or_input_row_order():
    table=cohort(); first=annotate(table); second=annotate(table.sample(frac=1,random_state=3))
    a=first.set_index("trial_id").material_rank.sort_index(); b=second.set_index("trial_id").material_rank.sort_index()
    pd.testing.assert_series_equal(a,b)
    assert first.groupby("material_key").material_rank.nunique().eq(1).all()
    for _,part in first.groupby(["session","original_label"]): assert set(part.material_rank)==set(range(6))


def test_binary_bootstrap_prediction_matches_defined_probability_floor_and_ties():
    rows=pd.DataFrame({"original_label":[3,3,1],"p_0":[1.,.2,.2],"p_1":[1e-20,.4,.6],"p_2":[2e-20,.4,.2]})
    # A nearly-neutral distribution uses the defined mass floor; exact binary ties choose negative.
    np.testing.assert_array_equal(correctness(rows,"binary"),[False,False,True])
