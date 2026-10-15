"""Preserve the sealed predecessor before writing the public-evidence-only revision."""
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / 'paper/exploratory_evaluation_2026-10-12'
NEW = REPO / 'paper/exploratory_evaluation_2026-10-15'
ARCHIVE = REPO / 'docs/paper_archive/2026-10-15-before-paper-update'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not NEW.exists() and not ARCHIVE.exists(), 'Preserve existing revision/archive'
    receipt = json.loads((OLD / 'build_receipt.json').read_text(encoding='utf-8'))
    for name, expected in receipt['bindings'].items():
        assert sha(REPO / name) == expected, name
    paths = sorted(p for p in OLD.rglob('*') if p.is_file()
                   and not {'build', 'tmp'}.intersection(p.relative_to(OLD).parts))
    entries = []
    for source in paths:
        relative = source.relative_to(REPO)
        destination = ARCHIVE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
        entries.append({'source': relative.as_posix(), 'archive': relative.as_posix(),
                        'sha256': sha(source), 'bytes': source.stat().st_size})
    manifest = {'date': '2026-10-15', 'predecessor_commit': 'b6c14b0e2',
                'old_sealed_bindings_checked': len(receipt['bindings']), 'files': entries}
    (ARCHIVE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (ARCHIVE / 'README.md').write_text(
        '# Paper preserved before the 15 October update\n\n'
        'The author requested an evidence update after the completed EEGNet diagnostic. '
        'This archive preserves the unchanged 12 October paper, supplement, PDFs, '
        'evidence and reviews. All 31 predecessor bindings passed before copying; '
        'each archive copy was SHA-256 verified. Build/render intermediates are excluded.\n\n'
        'The revision is `paper/exploratory_evaluation_2026-10-15/`. '
        'The approved exploratory question is retained. No data access, model fitting, '
        'reserved-outcome access or submission occurs in this step.\n', encoding='utf-8')
    NEW.mkdir()
    for name in ('manuscript.tex', 'supplementary_methods.tex', 'references.bib'):
        shutil.copy2(OLD / name, NEW / name)
    shutil.copytree(OLD / 'tables', NEW / 'tables')
    (NEW / 'figures').mkdir()
    shutil.copy2(OLD / 'figures/pairing_publication.pdf', NEW / 'figures/pairing_publication.pdf')
    (NEW / 'predecessor_preservation.json').write_text(json.dumps({
        'archive_manifest': (ARCHIVE / 'manifest.json').relative_to(REPO).as_posix(),
        'archive_manifest_sha256': sha(ARCHIVE / 'manifest.json'),
        'old_build_receipt_sha256': sha(OLD / 'build_receipt.json'),
        'new_question': False, 'new_fits': 0, 'reserved_access': False}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'archived_files': len(entries), 'old_bindings_checked': len(receipt['bindings'])}))


if __name__ == '__main__':
    main()
