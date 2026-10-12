"""Copy and hash-check paper predecessors before the approved framing revision."""
from pathlib import Path
import hashlib
import json
import shutil

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / 'docs/paper_archive/2026-10-12-before-approved-exploratory-revision'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if DEST.exists():
        raise FileExistsError('Archive is write-once; verify the existing manifest instead')
    for folder in ('current_progress_2026-10-12', 'submission_note_2026-10-12'):
        receipt = json.loads((REPO / 'paper' / folder / 'build_receipt.json').read_text())
        assert receipt['passed']
        for name, expected in receipt['bindings'].items():
            assert digest(REPO / name) == expected, name
    copies = []
    for folder in ('current_progress_2026-10-12', 'submission_note_2026-10-12'):
        base = REPO / 'paper' / folder
        for src in sorted(base.rglob('*')):
            if not src.is_file() or {'build', 'tmp'} & set(src.relative_to(base).parts):
                continue
            copies.append((src, Path('paper') / folder / src.relative_to(base)))
    for src in (REPO.parent / 'report.tex', REPO.parent / 'DL-Report.pdf'):
        if src.is_file():
            copies.append((src, Path('original') / src.name))
    manifest = []
    for src, relative in copies:
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        assert digest(target) == digest(src)
        manifest.append({'source': str(src), 'archive': relative.as_posix(),
                         'bytes': src.stat().st_size, 'sha256': digest(src)})
    (DEST / 'manifest.json').write_text(json.dumps({'purpose': 'Author-approved exploratory framing revision; preserve historical versions', 'files': manifest}, indent=2)+'\n', encoding='utf-8')
    (DEST / 'README.md').write_text('# Paper versions before the approved exploratory revision\n\nCreated 12 October 2026 before writing the revised paper. Both previous manuscript packages, their compiled PDFs, and available original GRU-XNet source/PDF were copied without changes and verified by SHA-256. The two earlier sealed build receipts were verified first. Build intermediates and temporary page renders are excluded.\n\nThe new approved paper is `paper/exploratory_evaluation_2026-10-12/`. The original GRU-XNet paper and earlier exploratory drafts remain historical artifacts; their claims are not silently rewritten. Five original referenced figures were already unavailable, as recorded in earlier archives.\n', encoding='utf-8')
    print(json.dumps({'archived_files': len(manifest), 'path': str(DEST)}))

if __name__ == '__main__':
    main()
