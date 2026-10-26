"""Protect documented identity semantics and metadata-only extraction boundaries."""
import importlib.util
from pathlib import Path
import zipfile
import pytest

script = Path(__file__).resolve().parents[1] / 'scripts/audit_faced_codebooks.py'
spec = importlib.util.spec_from_file_location('faced_codebooks', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def miniature_source(path, *, formula=False, external=False, hidden=False):
    state = ' state="hidden"' if hidden else ''
    mode = ' TargetMode="External"' if external else ''
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"' + state + '/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"' + mode + '/></Relationships>')
        value = '<f>1+1</f><v>2</v>' if formula else '<v>2</v>'
        z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1">' + value + '</c></row></sheetData></worksheet>')


def test_formula_cached_result_is_not_treated_as_documented_input(tmp_path):
    path = tmp_path / 'Stimuli_info.xlsx'
    miniature_source(path, formula=True)
    with pytest.raises(ValueError, match='Formula'):
        module.read_cells(path)


def test_external_relationship_is_not_followed(tmp_path):
    path = tmp_path / 'Task_event.xlsx'
    miniature_source(path, external=True)
    with pytest.raises(ValueError, match='External'):
        module.read_cells(path)


def test_hidden_sheet_requires_separate_review(tmp_path):
    path = tmp_path / 'DataStructureOfBehaviouralData.xlsx'
    miniature_source(path, hidden=True)
    with pytest.raises(ValueError, match='inventory'):
        module.read_cells(path)


def test_participant_file_is_outside_parser_scope(tmp_path):
    path = tmp_path / 'participant_ratings.xlsx'
    miniature_source(path)
    with pytest.raises(ValueError, match='three small'):
        module.read_cells(path)


def test_presentation_order_is_not_catalogue_identity():
    rows = [{'trial': i, 'vid': 29 - i} for i in range(1, 29)]
    mapping = module.validate_identity_projection(rows)
    assert mapping[28] == 1
    assert mapping[1] == 28


def test_duplicate_video_identity_cannot_join_twice():
    rows = [{'trial': i, 'vid': 1} for i in range(1, 29)]
    with pytest.raises(ValueError, match='Duplicate'):
        module.validate_identity_projection(rows)


def test_score_payload_is_rejected_without_inspecting_it():
    class ForbiddenScore:
        def __iter__(self):
            raise AssertionError('A score payload was read')
        def __float__(self):
            raise AssertionError('A score payload was read')
    rows = [{'trial': i, 'vid': i, 'score': ForbiddenScore()} for i in range(1, 29)]
    with pytest.raises(ValueError, match='identity projection'):
        module.validate_identity_projection(rows)


def test_boolean_identity_cannot_stand_in_for_video_one():
    rows = [{'trial': i, 'vid': i} for i in range(1, 29)]
    rows[0]['vid'] = True
    with pytest.raises(ValueError, match='exact integers'):
        module.validate_identity_projection(rows)


def test_arousal_valence_swap_is_detected_before_using_scores():
    cells = {'A2': 'score', 'A4': 'trial', 'A5': 'vid', 'A6': 'Accuracy', 'A7': 'ResponseTime', 'C2': '0-7',
             'C3': '"joy","tenderness","inspiration","amusement","anger","disgust","fear","sadness","valence","arousal","familiarity","liking"'}
    with pytest.raises(ValueError, match='item order'):
        module.decode_behaviour(cells)
