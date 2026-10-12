"""Check and seal the approved exploratory paper; access public artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import fitz

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / 'paper/exploratory_evaluation_2026-10-12'
ARCHIVE = REPO / 'docs/paper_archive/2026-10-12-before-approved-exploratory-revision'
DOCS = [('main', 'main', 'manuscript', 'faced_exploratory_evaluation'),
        ('supplement', 'supplement', 'supplementary_methods', 'faced_exploratory_supplement')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def bindings(values):
    for name, expected in values.items():
        assert sha(REPO / name) == expected, name

def check_evidence():
    activation = read(PAPER / 'evidence_activation.json')
    receipt = read(PAPER / 'evidence_receipt.json')
    audit = read(PAPER / 'independent_numeric_audit.json')
    assert activation['author_framing_approved'] and not activation['confirmation_access']
    assert activation['new_fits'] == 0 and not activation['original_data_access']
    bindings(activation['input_bindings'])
    assert receipt['input_bindings'] == activation['input_bindings']
    bindings(receipt['output_bindings'])
    assert receipt['activation_sha256'] == sha(PAPER / 'evidence_activation.json')
    assert audit['passed'] and not audit['protected_access']
    assert audit['evidence_sha256'] == sha(PAPER / 'evidence.json')
    assert audit['producer_receipt_sha256'] == sha(PAPER / 'evidence_receipt.json')
    assert audit['paired_units_joined'] == 840 and audit['panels'] == 6
    assert audit['crossed_draws_independently_replayed'] == 24000
    assert audit['max_interval_endpoint_discrepancy'] < 1e-12
    assert all(r['interval_includes_zero'] for r in audit['results'])
    layout = read(PAPER / 'figures/figure_layout_receipt.json')
    bindings(layout['input_bindings'])
    bindings(layout['output_bindings'])
    assert layout['display_point_count'] == 12 and layout['minimum_text_font_pt'] >= 8
    evidence = read(PAPER / 'evidence.json')
    panels = {(p['anchor'], p['scheme'], p['rotation'], p['arm']): p for p in evidence['pairing_panels']}
    for point in layout['display_points']:
        p = panels[tuple(point[k] for k in ('anchor', 'scheme', 'rotation', 'arm'))]
        assert point['mean_full_delta'] == p['mean_full_delta']
        assert point['descriptive_95_percent_interval'] == p['full_panel_crossed_interval']['descriptive_95_percent_interval']
    for folder in ('current_progress_2026-10-12', 'submission_note_2026-10-12'):
        old = read(REPO / 'paper' / folder / 'build_receipt.json')
        assert old['passed']
        bindings(old['bindings'])
    for item in read(ARCHIVE / 'manifest.json')['files']:
        assert sha(ARCHIVE / item['archive']) == item['sha256']
        assert sha(Path(item['source'])) == item['sha256']
    assert sha(REPO.parent / 'report.tex') == sha(ARCHIVE / 'original/report.tex')
    return {'matched_units': 840, 'independent_draws': 24000,
            'input_bindings': len(activation['input_bindings']),
            'output_bindings': len(receipt['output_bindings']),
            'archived_files_unchanged': len(read(ARCHIVE / 'manifest.json')['files']),
            'publication_figure_points_unchanged': 12}

def tex_words(source):
    source = re.sub(r'\\(?:cite|ref|label)\{[^}]*\}', '', source)
    source = re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?', '', source)
    return len(re.findall(r"\b\w+(?:[-.'/]\w+)*\b", source))

def check_documents():
    source = (PAPER / 'manuscript.tex').read_text(encoding='utf-8')
    supplement = (PAPER / 'supplementary_methods.tex').read_text(encoding='utf-8')
    assert source.index('Muhammad Muntazar') < source.index('Muhammad Wasif Shakeel')
    abstract = source.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    abstract_words = tex_words(abstract)
    assert 150 <= abstract_words <= 250
    assert all(x in abstract for x in ('Purpose.', 'Methods.', 'Results.', 'Conclusion.'))
    displays = len(re.findall(r'\\input\{tables/', source + supplement)) + (source + supplement).count(r'\begin{figure}')
    assert displays == 8
    docs = {}
    for label, folder, stem, final in DOCS:
        pdf = PAPER / 'output/pdf' / (final + '.pdf')
        log = (PAPER / 'build' / folder / (stem + '.log')).read_text(errors='replace')
        assert not re.search(r'Overfull|Citation.*undefined|Reference.*undefined', log), label
        document = fitz.open(pdf)
        text = '\n'.join(page.get_text() for page in document)
        assert not any(s in text for s in ('[?]', '??', 'TODO', 'TBD', '\ufffd')), label
        assert 'Muhammad Muntazar' in text and 'Muhammad Wasif Shakeel' in text
        outside = []
        for n, page in enumerate(document, 1):
            for block in page.get_text('blocks'):
                if block[0] < -.1 or block[1] < -.1 or block[2] > page.rect.width + .1 or block[3] > page.rect.height + .1:
                    outside.append(n)
        assert not outside, (label, outside)
        docs[label] = {'pages': len(document), 'bytes': pdf.stat().st_size,
                       'whitespace_word_count': len(text.split()),
                       'conservative_alphanumeric_word_count': len(re.findall(r'\b\w+\b', text)),
                       'overfull_boxes': 0, 'undefined_references': 0,
                       'outside_page_text_blocks': 0, 'sha256': sha(pdf)}
        assert sha(pdf) == sha(PAPER / 'build' / folder / (stem + '.pdf'))
        assert label != 'main' or docs[label]['conservative_alphanumeric_word_count'] <= 3000
    body = source.split(r'\section{Introduction}', 1)[1].split(r'\section*{Declarations}', 1)[0]
    body = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', body, flags=re.S)
    body = re.sub(r'\\input\{[^}]*\}', '', body)
    return {'pdfs': docs, 'abstract_words': abstract_words,
            'body_words_without_float_text': tex_words(body),
            'main_display_items': 3, 'supplement_display_items': 5,
            'total_display_items': displays,
            'word_count_method': 'Main PDF inclusive count includes title, authors, abstract, body, tables, figure text, declarations, references and page numbers; conservative regex splits all punctuation.'}

def prepare():
    assert not (PAPER / 'build_receipt.json').exists(), 'Preserve sealed paper; create a new revision'
    audit = check_evidence()
    output = PAPER / 'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    renders = PAPER / 'tmp/pdfs/final-review-v4'
    renders.mkdir(parents=True, exist_ok=True)
    for label, folder, stem, final in DOCS:
        shutil.copy2(PAPER / 'build' / folder / (stem + '.pdf'), output / (final + '.pdf'))
        doc = fitz.open(output / (final + '.pdf'))
        (renders / (label + '.txt')).write_text('\n'.join(p.get_text() for p in doc), encoding='utf-8')
        for number, page in enumerate(doc, 1):
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(renders / f'{label}-{number}.png')
    result = {'passed': True, 'evidence_checks': audit, **check_documents(),
              'visual_review_pending': True, 'render_directory': renders.relative_to(REPO).as_posix()}
    (PAPER / 'tmp/preflight.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))

def seal():
    path = PAPER / 'build_receipt.json'
    assert not path.exists(), 'Preserve sealed paper; create a new revision'
    visual = read(PAPER / 'visual_review.json')
    result = {'passed': True, 'evidence_checks': check_evidence(), **check_documents()}
    assert visual['passed']
    for label in ('main', 'supplement'):
        assert visual[label]['pages_inspected'] == list(range(1, result['pdfs'][label]['pages'] + 1))
        assert visual[label]['pdf_sha256'] == result['pdfs'][label]['sha256']
    result.update({'visual_review': visual, 'new_fits': 0,
                   'new_cached_matched_sensitivity_panels': 6,
                   'original_or_private_array_access': False,
                   'protected_outcomes_opened': False,
                   'author_framing_approved': True,
                   'independent_confirmation_claimed': False,
                   'actual_submission': False, 'author_declarations_complete': False,
                   'preparation_target': 'Journal of Medical and Biological Engineering, Original Article, subscription route',
                   'compiler': 'Tectonic 0.17.0, untrusted mode',
                   'fontconfig_notice': 'Default configuration notice emitted on Windows; final embedded-font rendering inspected'})
    paths = [p for p in PAPER.rglob('*') if p.is_file() and not {'build', 'tmp'}.intersection(p.relative_to(PAPER).parts)]
    paths += [Path(__file__), ARCHIVE / 'manifest.json']
    result['bindings'] = {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)}
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('bindings', 'visual_review')}))

def verify():
    saved = read(PAPER / 'build_receipt.json')
    bindings(saved['bindings'])
    assert saved['evidence_checks'] == check_evidence()
    for k, v in check_documents().items():
        assert saved[k] == v, k
    print(json.dumps({'passed': True, 'sealed_bindings_checked': len(saved['bindings']),
                      'protected_access': False, 'actual_submission': False}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else seal if args.seal else verify)()
