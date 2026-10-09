"""The boundary adapter permits shared validation people but rejects leakage."""
import textwrap
import pandas as pd
import pytest
from scripts.analyze_cbramod_readout_boundary_adapter import effective_source, ORIGINAL


def guard(source, roles):
    begin = source.index('            for i, a in enumerate(ROLES):')
    end = source.index("            if set(roles['train'].material_key)", begin)
    code = textwrap.dedent(source[begin:end])
    exec(compile(code, '<declared-role-boundary>', 'exec'), {
        'ROLES': ('train', 'validation_unseen', 'validation_familiar'), 'roles': roles})


def panels():
    return {role: pd.DataFrame({'trial_id': [trial], 'subject_id': [subject]})
        for role, trial, subject in (('train', 'T', 'training_person'),
            ('validation_unseen', 'U', 'heldout_person'), ('validation_familiar', 'F', 'heldout_person'))}


def test_shared_validation_people_are_allowed_without_altering_trials():
    roles = panels()
    with pytest.raises(ValueError, match='role leakage'):
        guard(ORIGINAL.read_text(encoding='utf-8'), roles)
    guard(effective_source(), roles)


@pytest.mark.parametrize('field,value', (('subject_id', 'training_person'), ('trial_id', 'T'), ('trial_id', 'F')))
def test_training_people_and_all_pairwise_trial_leaks_still_fail(field, value):
    roles = panels(); roles['validation_unseen'].loc[0, field] = value
    with pytest.raises(ValueError, match='role leakage'):
        guard(effective_source(), roles)
