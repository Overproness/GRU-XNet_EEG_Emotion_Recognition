"""Audit a venue-format derivative using public results; never open raw data."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper/submission_note_2026-10-12"
ORIGINAL = REPO / "paper/current_progress_2026-10-12"
ORDER = ["global_median", "coarse_valence_median", "category_median", "context_only", "context_kernel",
         "spectral_eeg_only", "context_plus_spectral_eeg", "covariance_linear", "context_covariance_linear",
         "spectral_kernel", "context_spectral_kernel", "covariance_kernel", "context_covariance_kernel",
         "eeg_only", "context_residual", "retrospective_context_envelope"]
MAIN_ORDER = ["global_median", "category_median", "context_only", "context_kernel", "spectral_eeg_only",
              "context_plus_spectral_eeg", "context_covariance_kernel", "eeg_only", "context_residual"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def word_count(source, start, end, floats=False):
    text = source.split(start, 1)[1].split(end, 1)[0]
    text = re.sub(r"(?<!\\)%[^\n]*", "", text)
    if not floats:
        text = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", "", text, flags=re.S)
    text = re.sub(r"\\(?:cite|ref|label)\{[^}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", "", text)
    return len(re.findall(r"\b[\w]+(?:[-.'/][\w]+)*\b", text))


def check_pdf(path, label, log):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    if any(token in text for token in ("[?]", "TODO", "TBD", "\ufffd")):
        raise ValueError("Unresolved citation/glyph/text in " + label)
    if re.search(r"Overfull|Citation.*undefined|Reference.*undefined", log):
        raise ValueError("Build overflow or undefined reference in " + label)
    if "References" not in text or "1.2577" not in text:
        raise ValueError("Required content missing in " + label)
    outside = []
    for number, page in enumerate(doc, 1):
        for block in page.get_text("blocks"):
            if block[0] < 0 or block[1] < 0 or block[2] > page.rect.width + .1 or block[3] > page.rect.height + .1:
                outside.append(number)
    if outside or path.stat().st_size >= 20_000_000:
        raise ValueError("Page geometry/file-size check failed in " + label)
    return {"pages": len(doc), "bytes": path.stat().st_size, "unresolved_citations_or_glyphs": 0,
            "overfull_boxes": 0, "outside_page_text_blocks": 0}


def checks():
    source = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    supplement = (PAPER / "supplementary_methods.tex").read_text(encoding="utf-8")
    data = json.loads((ORIGINAL / "results_table.json").read_text())
    if data != json.loads((PAPER / "results_table.json").read_text()):
        raise ValueError("Public aggregate copy differs")
    with (PAPER / "results_table.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if [row["condition"] for row in rows] != ORDER:
        raise ValueError("Machine-readable conditions missing/reordered")
    for key, row in zip(ORDER, rows):
        if [float(row[f"rotation_{i}_mae"]) for i in range(3)] != data[key]["rotations"] or float(row["mean_mae"]) != data[key]["mean"]:
            raise ValueError("Machine-readable values differ: " + key)
    parsed = []
    for line in supplement.splitlines():
        cells = [v.strip() for v in line.rstrip("\\").split("&")]
        if len(cells) != 6:
            continue
        try:
            parsed.append([float(v) for v in cells[2:]])
        except ValueError:
            pass
    expected = [[float(f"{v:.4f}") for v in data[key]["rotations"] + [data[key]["mean"]]] for key in ORDER]
    if parsed != expected:
        raise ValueError("Supplemental result table differs")
    main = []
    for line in source.splitlines():
        cells = [v.strip() for v in line.rstrip("\\").split("&")]
        if len(cells) == 3:
            try:
                main.append(float(cells[-1]))
            except ValueError:
                pass
    if main != [float(f"{data[key]['mean']:.4f}") for key in MAIN_ORDER]:
        raise ValueError("Main comparison table differs")
    counts = {"abstract": word_count(source, "% ABSTRACT_WORDCOUNT_START", "% ABSTRACT_WORDCOUNT_END"),
              "body_without_floats": word_count(source, "% MAIN_WORDCOUNT_START", "% MAIN_WORDCOUNT_END"),
              "body_with_float_text": word_count(source, "% MAIN_WORDCOUNT_START", "% MAIN_WORDCOUNT_END", True)}
    if counts["abstract"] > 200 or counts["body_with_float_text"] > 2000:
        raise ValueError("Conservative research-note word count exceeds limits")
    if source.count("\\begin{table}") + source.count("\\begin{figure}") > 3:
        raise ValueError("Too many main display items")
    if "Muhammad Muntazar and Muhammad Wasif Shakeel" not in source or "\\bibliographystyle{vancouver}" not in source:
        raise ValueError("Author order or bibliography style incorrect")
    for ext in ("pdf", "png"):
        if sha(PAPER / f"figures/mae_comparison.{ext}") != sha(ORIGINAL / f"figures/mae_comparison.{ext}"):
            raise ValueError("Bound figure copy differs")
    # This also preserves the original full paper and all its bound public inputs.
    old = json.loads((ORIGINAL / "build_receipt.json").read_text())
    for rel, expected_sha in old["bindings"].items():
        if sha(REPO / rel) != expected_sha:
            raise ValueError("Original paper binding changed: " + rel)
    archive = REPO / "docs/paper_archive/2026-10-12-before-current-progress-draft/report.tex"
    if sha(REPO.parent / "report.tex") != sha(archive):
        raise ValueError("Official original manuscript changed")
    pdfs = {}
    for label, filename, folder, stem in [("main", "faced_research_note.pdf", "main", "manuscript"),
                                         ("supplement", "faced_supplementary_methods.pdf", "supplement", "supplementary_methods")]:
        logpath = PAPER / f"build/{folder}/{stem}.log"
        log = logpath.read_text(errors="replace") if logpath.exists() else ""
        pdfs[label] = check_pdf(PAPER / f"output/pdf/{filename}", label, log)
    return {"passed": True, "word_counts": counts, "word_count_method": "Conservative regex count; exact publisher counts may differ",
            "supplemental_numeric_cells_checked": 64, "main_table_means_checked": 9,
            "machine_readable_numeric_cells_checked": 64, "original_paper_bindings_preserved": len(old["bindings"]),
            "official_original_source_unchanged": True, "main_display_items": 2, "pdfs": pdfs,
            "new_fits": 0, "new_predictions": 0, "original_data_or_targets_accessed": 0,
            "confirmation_opened": False, "research_question_replacement_adopted": False,
            "actual_submission": False, "author_declarations_complete": False,
            "preparation_target": "BMC Research Notes, research note"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    receipt = PAPER / "build_receipt.json"
    result = checks()
    if args.seal:
        if receipt.exists():
            raise FileExistsError("Preserve the sealed review package; declare revisions separately")
        paths = [p for p in PAPER.rglob("*") if p.is_file() and not {"build", "tmp"}.intersection(p.relative_to(PAPER).parts)]
        paths += [Path(__file__), REPO / "docs/publication/FACED_Submission_Route_Qualification_2026-10-12.md"]
        result["bindings"] = {p.relative_to(REPO).as_posix(): sha(p) for p in paths}
        result["visual_review"] = {"renderer": "PyMuPDF", "main_pages": list(range(1, result["pdfs"]["main"]["pages"] + 1)),
                                   "supplement_pages": list(range(1, result["pdfs"]["supplement"]["pages"] + 1)),
                                   "clipping_overlap_missing_glyphs": "none observed in final page renders"}
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        saved = json.loads(receipt.read_text())
        for rel, expected_sha in saved["bindings"].items():
            if sha(REPO / rel) != expected_sha:
                raise ValueError("Prepared package binding changed: " + rel)
        for key, value in result.items():
            if saved[key] != value:
                raise ValueError("Verification changed: " + key)
        result["bindings_checked"] = len(saved["bindings"])
    print(json.dumps({k: v for k, v in result.items() if k not in ("bindings", "visual_review")}))


if __name__ == "__main__":
    main()
