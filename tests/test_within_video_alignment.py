import numpy as np
import pandas as pd
from scripts.within_video_alignment import pairs,scores,aggregate


def cohort():
    return pd.DataFrame({'subject_id':['a','b','c'],'trial_id':['a1','b1','c2'],
        'source_session':[1]*3,'material_rotation':[0]*3,'fold':[0]*3,
        'material_key':['v1','v1','v2'],'label':[0,1,1],'original_label':[3.,7.,7.],
        'p_0':[.9,.1,.2],'p_1':[.1,.9,.8]})


def test_exchange_is_within_video_other_subject_and_scores_not_probabilities():
    table=cohort(); r,d,w,eligible=pairs(table)
    np.testing.assert_array_equal(r,[0,1]); np.testing.assert_array_equal(d,[1,0]); np.testing.assert_array_equal(eligible,[0,1])
    target,included,value,classes=scores(table,'binary','DEAP',r,d)
    np.testing.assert_array_equal(value['aligned']['BA'],[1.,1.]); np.testing.assert_array_equal(value['exchanged']['BA'],[0.,0.])
    np.testing.assert_allclose(value['aligned']['logloss'],-np.log(.9)); np.testing.assert_allclose(value['exchanged']['logloss'],-np.log(.1))


def test_dyadic_weights_use_both_people_and_flag_nonestimable_draws():
    table=cohort(); r,d,w,eligible=pairs(table); target,included,value,classes=scores(table,'binary','DEAP',r,d)
    point,draws,valid=aggregate(value['aligned']['BA'],target,included,classes,w,r,d,np.zeros(2,dtype=int),
                              np.array([[2.,1.,0.],[2.,0.,0.]]),np.ones((2,1)))
    assert point==1.; np.testing.assert_array_equal(valid,[True,False]); assert draws[0]==1.


def test_context_only_exchange_is_invariant_with_individual_labels():
    table=cohort(); table['p_0']=.3; table['p_1']=.7
    r,d,w,eligible=pairs(table); target,included,value,classes=scores(table,'binary','DEAP',r,d)
    for stat in ('BA','logloss'): np.testing.assert_array_equal(value['aligned'][stat],value['exchanged'][stat])
