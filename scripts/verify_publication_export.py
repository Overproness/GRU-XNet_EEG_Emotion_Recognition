"""Independently check the bounded export manifest and preserved manuscript bytes."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import re

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
PAPER_SHA="06262fb4070b2faa88b16bead411a9282ecac8920f838bcd4a92d8bacf4fd8f0"
PDF_SHA="921f07f271bc4dadbabaf62ede384c543681b9fe0aec02c490cf7d76ef9de101"


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def verify(check_sources=True):
    manifest=json.loads((REPO/"results/development/export_manifest.json").read_text())
    seen=set()
    for record in manifest["files"]:
        target=(REPO/record["repository_export"]).resolve()
        if not target.is_relative_to(REPO) or target in seen:
            raise ValueError("Unexpected or duplicate export target")
        seen.add(target)
        if not target.is_file() or target.stat().st_size>2_000_000:
            raise ValueError(f"Missing or oversized artifact: {record['repository_export']}")
        if checksum(target)!=record["export_sha256"]:
            raise ValueError(f"Changed exported bytes: {record['repository_export']}")
        if check_sources:
            source=(WORKSPACE/record["workspace_source"]).resolve()
            if not source.is_relative_to(WORKSPACE) or checksum(source)!=record["source_sha256"]:
                raise ValueError(f"Changed local source: {record['workspace_source']}")
    archive=REPO/"docs/paper_archive/2026-10-05-pre-exploration"
    if checksum(archive/"report.tex")!=PAPER_SHA or checksum(archive/"DL-Report.pdf")!=PDF_SHA:
        raise ValueError("Historical manuscript preservation changed")
    if check_sources and checksum(WORKSPACE/"report.tex")!=PAPER_SHA:
        raise ValueError("Working manuscript changed without a fresh archive")
    archive_record=json.loads((archive/"manifest.json").read_text())
    if len(archive_record["missing_referenced_assets"])!=5:
        raise ValueError("Unexpected historical source asset record")
    links=0
    documents=[p for p in seen if p.suffix==".md"]+[REPO/"README.md",REPO/"PUBLICATION.md",archive/"README.md"]
    for document in documents:
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)",document.read_text(encoding="utf-8")):
            if re.match(r"(?:https?://|mailto:|#)",raw): continue
            destination=(document.parent/raw.split("#",1)[0].strip("<>")).resolve()
            if not destination.exists():
                raise ValueError(f"Broken local link in {document.name}: {raw}")
            links+=1
    result={"passed":True,"exported_files":len(seen),"source_hashes_checked":check_sources,
            "local_links_checked":links,"historical_source_sha256":PAPER_SHA,
            "historical_pdf_sha256":PDF_SHA,"known_missing_source_figures":5,
            "scope":"Export/source byte integrity, local links and historical archive; no numerical experiment validation or manuscript compilation"}
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--export-only",action="store_true",help="For a public checkout without original local sources")
    args=parser.parse_args()
    verify(check_sources=not args.export_only)
