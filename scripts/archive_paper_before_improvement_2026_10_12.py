"""Preserve the approved paper and bound build tools before improvement work."""
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / 'paper/exploratory_evaluation_2026-10-12'
ARCHIVE = REPO / 'docs/paper_archive/2026-10-12-before-improvement'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    receipt = json.loads((PAPER / 'build_receipt.json').read_text(encoding='utf-8'))
    assert receipt['passed']
    for name, expected in receipt['bindings'].items():
        assert sha(REPO / name) == expected, name
    if ARCHIVE.exists():
        raise FileExistsError('Preserve the existing archive; verify rather than overwrite')
    files = sorted(p for p in PAPER.rglob('*') if p.is_file() and not {'build', 'tmp'}.intersection(p.relative_to(PAPER).parts))
    files += [REPO / n for n in receipt['bindings'] if n.startswith('scripts/')]
    manifest = []
    for source in files:
        relative = source.relative_to(REPO)
        target = ARCHIVE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source)
        manifest.append({'source': relative.as_posix(), 'archive': relative.as_posix(),
                         'sha256': sha(source), 'bytes': source.stat().st_size})
    (ARCHIVE / 'manifest.json').write_text(json.dumps({'date':'2026-10-12',
        'paper_revision':'395cb374f', 'reason':'User requested preserving current papers before attempting improvements',
        'files':manifest}, indent=2)+'\n', encoding='utf-8')
    (ARCHIVE / 'README.md').write_text('# Approved papers before improvement work\n\n'
        '12 October 2026. This write-once archive preserves the approved six-page main paper, '
        'five-page supplement, editable sources, figures, evidence and review/build receipts '
        'from Git revision `395cb374f`, with their bound build tools. Every copy was SHA-256 '
        'checked after all 31 original receipt bindings passed. Temporary builds/renders are excluded.\n\n'
        'The active approved paper package and the earlier paper archives remain unchanged. '
        'New improvement protocols and findings will use separate files. The user authorized '
        'improving the work, but any distinct research-question adoption still requires showing '
        'the findings and obtaining the author\'s decision. No protected access or fitting occurs in this archive step.\n',
        encoding='utf-8')
    print(json.dumps({'archived_files':len(manifest), 'archive':str(ARCHIVE), 'original_bindings_verified':31}))

if __name__ == '__main__':
    main()
