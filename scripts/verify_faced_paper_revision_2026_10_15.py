"""Build, inspect and seal the 15 October paper using public artifacts only.

This verifies packaging and selected numerical transcriptions, not model fitting
or independent scientific confirmation. No dataset/private path is opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import fitz

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / 'paper/exploratory_evaluation_2026-10-15'
OLD = REPO / 'paper/exploratory_evaluation_2026-10-12'
NN = REPO / 'results/development/faced_stronger_baseline_continuation_2026-10-15'
COVERAGE = REPO / 'results/development/faced_coverage_missing_2026-10-13'
NN_AUDIT = REPO / 'results/development/faced_baseline_independent_audit_2026-10-15/completion_v2.json'
ARCHIVE = REPO / 'docs/paper_archive/2026-10-15-before-paper-update'
DOCUMENTS = [('main', 'manuscript', 'faced_exploratory_evaluation'),
             ('supplement', 'supplementary_methods', 'faced_exploratory_supplement')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def check_bindings(base, values):
    for name, expected in values.items():
        path = base / name
        assert path.is_relative_to(base), name
        assert sha(path) == expected, name


def evidence():
    old = read(OLD / 'build_receipt.json')
    assert old['passed']
    check_bindings(REPO, old['bindings'])
    preservation = read(PAPER / 'predecessor_preservation.json')
    assert sha(OLD / 'build_receipt.json') == preservation['old_build_receipt_sha256']
    assert sha(ARCHIVE / 'manifest.json') == preservation['archive_manifest_sha256']
    archived = read(ARCHIVE / 'manifest.json')['files']
    for entry in archived:
        assert sha(REPO / entry['source']) == entry['sha256'], entry['source']
        assert sha(ARCHIVE / entry['archive']) == entry['sha256'], entry['archive']
    # Check unchanged public experiment outputs against their frozen independent audits.
    neural_audit = read(NN_AUDIT)
    assert neural_audit['passed'] and neural_audit['prediction_rows'] == 6260
    assert not neural_audit['protected_or_validation_numeric_access']
    check_bindings(NN, neural_audit['public_result_hashes'])
    spectral_audit = read(COVERAGE / 'independent_audit.json')
    assert spectral_audit['passed'] and not spectral_audit['confirmation_access']
    check_bindings(COVERAGE, spectral_audit['bindings'])
    # Earlier display tables/figure are copied exactly; only their placement changes.
    copied = ['primary_mae.tex', 'all_original_conditions.tex', 'category_gain_intervals.tex',
              'matched_context_sensitivity.tex', 'all_pairing_intervals.tex',
              'target_decomposition.tex', 'source_selection.tex']
    for name in copied:
        assert sha(PAPER / 'tables' / name) == sha(OLD / 'tables' / name), name
    assert sha(PAPER / 'figures/pairing_publication.pdf') == sha(OLD / 'figures/pairing_publication.pdf')
    # Verify every numerical row in the newly added two-panel table at displayed precision.
    source = read(COVERAGE / 'comparison.json')
    nn = read(NN / 'comparison.json')
    table = (PAPER / 'tables/source_followup.tex').read_text(encoding='utf-8')
    rows = [('Category median', source['category_median_metrics']['participant_title_mae'])]
    labels = {'context_only': 'Matched context', 'late_joint': 'Late EEG + context',
              'distributed_joint': 'Distributed EEG + context'}
    for item in source['candidate_metrics']:
        if item['alpha'] == 100:
            rows.append((labels[item['model']] + r', $\alpha=100$', item['participant_title_mae']))
    for label, value in rows:
        assert label + r' & \multicolumn{3}{r}{' + f'{value:.4f}' + r'}\\' in table, label
    for i, item in enumerate(nn['metrics']):
        label = '360 (fixed primary)' if item['epoch'] == 360 else str(item['epoch'])
        row = f"{label} & {nn['source_mse_by_checkpoint'][0][i]:.4f} & {nn['source_mse_by_checkpoint'][1][i]:.4f} & {item['ensemble_held_mae']:.4f}"
        assert row + r'\\' in table, row
    for label, key in [('Category median', 'category_held_mae'),
                       (r'Matched context, $\alpha=100$', 'context_held_mae')]:
        assert label + r' & \multicolumn{3}{r}{' + f"{nn['metrics'][-1][key]:.4f}" + r'}\\' in table
    assert nn['primary_epoch'] == 360 and nn['learning_flags'] == ['budget_limited_convergence']
    assert not nn['protected_access'] and not nn['validation_access']
    assert source['category_median_metrics']['prediction_rows'] == 840
    assert not source['protected_access'] and not source['validation_access']
    return {'predecessor_sealed_bindings': len(old['bindings']),
            'archive_pairs_unchanged': len(archived), 'neural_public_hashes': 7,
            'spectral_public_hashes': len(spectral_audit['bindings']),
            'old_tables_unchanged': len(copied), 'new_table_numeric_rows_checked': 9,
            'models_refitted': 0, 'private_or_original_data_access': False,
            'reserved_access': False, 'scientific_confirmation': False}


def word_count(text):
    return len(re.findall(r'\b\w+\b', text))


def document_checks():
    main = (PAPER / 'manuscript.tex').read_text(encoding='utf-8')
    supplement = (PAPER / 'supplementary_methods.tex').read_text(encoding='utf-8')
    assert main.index('Muhammad Muntazar') < main.index('Muhammad Wasif Shakeel')
    abstract = main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    assert 150 <= word_count(abstract) <= 250
    assert all(x in abstract for x in ('Purpose.', 'Methods.', 'Results.', 'Conclusion.'))
    displays = len(re.findall(r'\\input\{tables/', main + supplement)) + (main + supplement).count(r'\begin{figure}')
    assert displays == 8, displays
    bibliography = (PAPER / 'references.bib').read_text(encoding='utf-8')
    cited = set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}', main + supplement)
                for k in group.split(','))
    keys = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    assert cited <= keys, cited - keys
    details = {}
    for label, stem, output in DOCUMENTS:
        path = PAPER / 'output/pdf' / (output + '.pdf')
        assert sha(path) == sha(PAPER / 'build' / label / (stem + '.pdf'))
        log = (PAPER / 'build' / label / (stem + '.log')).read_text(errors='replace')
        assert not re.search(r'Overfull|Citation.*undefined|Reference.*undefined', log), label
        doc = fitz.open(path)
        text = '\n'.join(page.get_text() for page in doc)
        assert not any(token in text for token in ('[?]', '??', 'TODO', 'TBD', '\ufffd')), label
        assert all(name in text for name in ('Muhammad Muntazar', 'Muhammad Wasif Shakeel'))
        for number, page in enumerate(doc, 1):
            for block in page.get_text('blocks'):
                assert block[0] >= -.1 and block[1] >= -.1 and block[2] <= page.rect.width+.1 and block[3] <= page.rect.height+.1, (label, number)
        details[label] = {'pages': len(doc), 'sha256': sha(path), 'bytes': path.stat().st_size,
                          'inclusive_alphanumeric_words': word_count(text),
                          'overfull_boxes': 0, 'undefined_citations_or_references': 0,
                          'outside_page_text_blocks': 0}
        assert label != 'main' or word_count(text) <= 3000, word_count(text)
    return {'pdfs': details, 'abstract_alphanumeric_words': word_count(abstract),
            'combined_display_items': displays, 'cited_bibliographic_keys': sorted(cited)}


def build():
    assert not (PAPER / 'build_receipt.json').exists(), 'Preserve sealed revision'
    evidence()
    compiler = REPO / '.publication_deps/tectonic/tectonic.exe'
    env = dict(os.environ)
    env['TECTONIC_CACHE_DIR'] = env.get('FACED_TECTONIC_CACHE_DIR',
                                     str(Path(tempfile.gettempdir()) / 'gru-xnet-tex-cache'))
    assert Path(env['TECTONIC_CACHE_DIR']).is_dir(), 'Set FACED_TECTONIC_CACHE_DIR to populated cache'
    output = PAPER / 'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    renders = PAPER / 'tmp/pdfs/review'
    renders.mkdir(parents=True, exist_ok=True)
    for label, stem, final in DOCUMENTS:
        folder = PAPER / 'build' / label
        folder.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(compiler), '--untrusted', '--keep-logs', '--keep-intermediates',
                        '--outdir', str(folder), str(PAPER / (stem + '.tex'))],
                       cwd=REPO, env=env, check=True)
        shutil.copy2(folder / (stem + '.pdf'), output / (final + '.pdf'))
        doc = fitz.open(output / (final + '.pdf'))
        (renders / (label + '.txt')).write_text('\n'.join(page.get_text() for page in doc), encoding='utf-8')
        for number, page in enumerate(doc, 1):
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(renders / f'{label}-{number}.png')
    result = {'evidence': evidence(), **document_checks(), 'visual_review_pending': True}
    write(PAPER / 'tmp/preflight.json', result)
    print(json.dumps(result))


def seal():
    assert not (PAPER / 'build_receipt.json').exists(), 'Preserve sealed revision'
    result = {'passed': True, 'date': '2026-10-15', 'evidence': evidence(), **document_checks()}
    visual = read(PAPER / 'visual_review.json')
    assert visual['passed']
    for label in ('main', 'supplement'):
        assert visual[label]['pdf_sha256'] == result['pdfs'][label]['sha256']
        assert visual[label]['pages_inspected'] == list(range(1, result['pdfs'][label]['pages'] + 1))
    numeric_review = read(PAPER / 'independent_numeric_review.json')
    assert numeric_review['checked_items_passed']
    check_bindings(REPO, numeric_review['snapshot_bindings'])
    result.update({'visual_review_sha256': sha(PAPER / 'visual_review.json'),
                   'new_fits_this_revision': 0, 'new_original_data_access': False,
                   'reserved_outcomes_opened': False, 'research_question_changed': False,
                   'actual_submission': False, 'author_declarations_complete': False,
                   'compiler': 'Tectonic 0.17.0, untrusted mode',
                   'renderer': 'PyMuPDF, 1.5 scale',
                   'fontconfig_notice': 'Windows default-config notice; embedded fonts visually checked'})
    paths = [p for p in PAPER.rglob('*') if p.is_file()
             and not {'build', 'tmp'}.intersection(p.relative_to(PAPER).parts)]
    paths += [Path(__file__), REPO / 'scripts/prepare_faced_paper_revision_2026_10_15.py',
              ARCHIVE / 'manifest.json', NN_AUDIT, COVERAGE / 'independent_audit.json',
              REPO / 'docs/publication/FACED_EEGNet_Evidence_Recheck_2026-10-15.md',
              REPO / 'docs/publication/FACED_Manuscript_Upgrade_Review_2026-10-15.md']
    result['bindings'] = {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)}
    write(PAPER / 'build_receipt.json', result)
    print(json.dumps({'passed': True, 'bindings': len(result['bindings']), 'pdfs': result['pdfs']}))


def verify():
    receipt = read(PAPER / 'build_receipt.json')
    check_bindings(REPO, receipt['bindings'])
    assert receipt['evidence'] == evidence()
    for key, value in document_checks().items():
        assert receipt[key] == value, key
    print(json.dumps({'passed': True, 'bindings': len(receipt['bindings']), 'reserved_access': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--build', action='store_true')
    actions.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    (build if args.build else seal if args.seal else verify)()
