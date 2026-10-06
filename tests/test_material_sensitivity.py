import numpy as np
from scripts.analyze_session_material_sensitivity import crossed_weights,protocol_scores
from test_session_controls import cohort


def test_crossed_bootstrap_preserves_emotion_strata_and_material_uncertainty():
    table=cohort()
    subjects,materials,sw,mw=crossed_weights(table,n=200,seed=9)
    assert (sw.sum(1)==15).all() and (mw.sum(1)==72).all()
    for _,rows in materials.groupby(["session","original_label"]):
        assert (mw[:,rows.index].sum(1)==6).all()
    # Identical performance for every person still has uncertainty over clips.
    table["p_0"],table["p_1"],table["p_2"]=1.,0.,0.
    correct=table.trial_id.str.endswith("01") & table.original_label.eq(3)
    table.loc[correct,["p_0","p_1","p_2"]]=[0.,0.,1.]
    point,draws=protocol_scores(table,subjects,materials,sw,mw,"coarse3")
    assert 0<point<1 and draws.std()>0
