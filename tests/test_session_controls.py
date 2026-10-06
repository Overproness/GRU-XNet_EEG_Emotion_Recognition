import numpy as np
import pandas as pd
from gruxnet.session_controls import paired_batches,source_masks,protocol_views
from gruxnet.temporal_controls import make_folds,COARSE
from gruxnet.data import SEED_LABELS


def cohort():
    # Public tests must not require the author's private raw-data caches.
    rows=[]
    for subject in range(1,16):
        for session in (1,2,3):
            for trial,label in enumerate(SEED_LABELS[session],1):
                rows.append({"subject_id":f"SEEDIV:S{subject:02}",
                             "trial_id":f"SEEDIV:S{subject:02}:R{session}:T{trial:02}",
                             "session":session,"original_label":label})
    return pd.DataFrame(rows)


def test_other_sessions_never_enter_fitting_or_selection():
    table=cohort(); fold=make_folds(table.subject_id.tolist())[0]
    for source in (1,2,3):
        masks=source_masks(table,fold,source)
        assert len(masks["train"])==216 and len(masks["validation"])==72
        for part in ("train","validation"):
            assert set(table.iloc[masks[part]].session)=={source}
            assert not set(table.iloc[masks[part]].subject_id)&set(fold["test"])
        for target,indices in masks["test"].items():
            assert len(indices)==72
            assert set(table.iloc[indices].session)=={target}
            assert set(table.iloc[indices].subject_id)==set(fold["test"])


def test_participant_emotion_and_rank_exposure_match_across_sources():
    table=cohort(); fold=make_folds(table.subject_id.tolist())[0]
    signatures=[]
    for source in (1,2,3):
        masks=source_masks(table,fold,source)
        batches,signature=paired_batches(table,masks["train"],42,updates=12)
        signatures.append(signature)
        assert set(batches.ravel()).issubset(set(masks["train"]))
        for batch in batches:
            assert np.bincount(COARSE[table.iloc[batch].original_label.to_numpy()],minlength=3).tolist()==[20,20,20]
    assert len(set(signatures))==1


def test_each_transfer_direction_tests_each_trial_once():
    table=cohort(); rows=[]
    for source in (1,2,3):
        part=table[["trial_id","subject_id","session"]].copy()
        part["source_session"],part["test_session"],part["seed"]=source,part.session,42
        rows.append(part)
    views=protocol_views(pd.concat(rows,ignore_index=True))
    assert set(views)=={"same_session","unseen_cycle1","unseen_cycle2"}
    for rows in views.values():
        assert len(rows)==1080 and not rows.trial_id.duplicated().any()
    for key in ("unseen_cycle1","unseen_cycle2"):
        assert (views[key].source_session!=views[key].test_session).all()
