"""Source metadata, genuine valence labels, and electrode alignment.

DEAP order: TorchEEG's dataset constants (linked in PUBLICATION.md).
SEED-IV order and session labels: the supplied Channel Order.xlsx / ReadMe.txt.
GAMEEMO labels: graphical participant SAM selections, never game IDs.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

DATASETS = ("DEAP", "GAMEEMO", "SEEDIV")
DEAP_CHANNELS = "FP1 AF3 F3 F7 FC5 FC1 C3 T7 CP5 CP1 P3 P7 PO3 O1 OZ PZ FP2 AF4 FZ F4 F8 FC6 FC2 CZ C4 T8 CP6 CP2 P4 P8 PO4 O2".split()
COMMON_CHANNELS = "AF3 AF4 F3 F4 F7 F8 FC5 FC6 O1 O2 P7 P8 T7 T8".split()
SEED_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}
POLICY = {
    "DEAP": "valence < 5: negative; > 5: positive; exactly 5: exclude",
    "GAMEEMO": "participant SAM valence 1..4: negative; 6..9: positive; 5: exclude",
    "SEEDIV": "sad(1)/fear(2): negative; happy(3): positive; neutral(0): exclude",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_provenance(path: Path) -> dict:
    """Bind a declared source catalog to an audit; this does not download/verify sources."""
    raw = path.read_bytes()
    catalog = json.loads(raw)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("Expected source catalog schema_version 1")
    datasets = catalog.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError("Source catalog must identify DEAP, GAMEEMO, and SEEDIV exactly once")
    for source in [catalog.get("bundle"), *datasets.values()]:
        if not isinstance(source, dict):
            raise ValueError("Source catalog requires a bundle and dataset source records")
        ref = source.get("kaggle_ref", "")
        version = source.get("kaggle_version")
        if (not isinstance(ref, str) or len(ref.split("/")) != 2 or not all(ref.split("/"))
                or source.get("kaggle_url") != f"https://www.kaggle.com/datasets/{ref}"
                or type(version) is not int or version < 1):
            raise ValueError("Each source must have a matching Kaggle URL/reference and positive integer version")
    return {"catalog_sha256": hashlib.sha256(raw).hexdigest(), "catalog": catalog,
            "binding_scope": "Declared source catalog snapshot; external integrity evidence is recorded separately"}


def roots(root: Path) -> dict[str, Path]:
    root = root.resolve()
    if (root / "Emotion Recognition EEG Datasets").is_dir():
        root = root / "Emotion Recognition EEG Datasets"
    found = {"DEAP": root / "deap", "GAMEEMO": root / "GAMEEMO", "SEEDIV": root / "sead-4"}
    for name, p in found.items():
        if not p.is_dir():
            raise FileNotFoundError(f"Missing {name} source directory: {p}")
    return found


def seed_channels(root: Path) -> list[str]:
    names = pd.read_excel(root / "Channel Order.xlsx", header=None).iloc[:, 0].astype(str)
    channels = [s.strip().upper() for s in names]
    if len(channels) != 62 or len(set(channels)) != 62:
        raise ValueError("Invalid SEED-IV channel order")
    text = (root / "ReadMe.txt").read_text(encoding="utf-8-sig")
    for session, expected in SEED_LABELS.items():
        match = re.search(rf"session{session}_label\s*=\s*\[([^\]]+)\]", text)
        if not match or [int(x) for x in re.findall(r"\d+", match[1])] != expected:
            raise ValueError(f"SEED-IV session {session} labels disagree with supplied ReadMe.txt")
    return channels


def valence_label(value: float, dataset: str) -> int | None:
    if dataset == "SEEDIV":
        if value not in (0, 1, 2, 3):
            raise ValueError(f"Unknown SEED-IV label: {value}")
        return None if value == 0 else int(value == 3)
    if not np.isfinite(value) or not 1 <= value <= 9:
        raise ValueError(f"Invalid {dataset} valence: {value}")
    return None if value == 5 else int(value > 5)


def deap_reference(root: Path):
    # Optional local dependency directory supports the user's read-only Conda env.
    try:
        import xlrd  # noqa: F401
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".publication_deps"))
    candidates = [p for p in [root / "metadata_xls/participant_ratings.xls", root / "Metadata/participant_ratings.xls"] if p.is_file()]
    if not candidates:
        raise FileNotFoundError("DEAP participant_ratings.xls is required to verify labels in this download")
    tables = [pd.read_excel(p).sort_values(["Participant_id", "Experiment_id"]).reset_index(drop=True) for p in candidates]
    if any(not tables[0].equals(t) for t in tables[1:]):
        raise ValueError("The two DEAP participant-rating spreadsheets disagree")
    table = tables[0]
    if len(table) != 1280 or table.duplicated(["Participant_id", "Experiment_id"]).any():
        raise ValueError("Invalid DEAP participant/experiment identifiers")
    references = {}
    for subject in range(1, 33):
        rows = table[table.Participant_id == subject].sort_values("Experiment_id")
        if rows.Experiment_id.tolist() != list(range(1, 41)):
            raise ValueError(f"DEAP subject {subject}: missing experiment IDs")
        references[subject] = rows[["Valence", "Arousal", "Dominance", "Liking"]].to_numpy(dtype=float)
    return references, {"file": str(candidates[0].resolve()), "sha256": sha256(candidates[0]),
                        "join": "Participant_id + Experiment_id (video order, not presentation Trial)"}


def verify_deap_labels(actual: np.ndarray, reference: np.ndarray) -> list[int]:
    if actual.shape != reference.shape or not np.allclose(actual[:, 2:], reference[:, 2:], atol=1e-8):
        raise ValueError("DEAP dominance/liking do not match the metadata video order")
    matches = np.isclose(actual, reference, atol=1e-8)
    inverse = np.isclose(actual, 9 - reference, atol=1e-8)
    if np.any(~matches & ~inverse):
        raise ValueError("Unexplained DEAP rating modification; check source data")
    return (~matches).sum(axis=0).tolist()


def numeric_eeg_keys(keys) -> list[str]:
    pairs = sorted((int(m[1]), k) for k in keys if (m := re.search(r"_eeg(\d+)$", k)))
    if [p[0] for p in pairs] != list(range(1, 25)):
        raise ValueError("Expected exactly the numeric EEG trial keys 1..24")
    return [p[1] for p in pairs]


def sam_rating(pdf: Path) -> dict:
    """Read strokes/circles/underlines against the nine fixed SAM positions.

    Includes long ellipses spanning both rows and crosses made of two strokes.
    Rejects conflicting marks instead of silently substituting a game label.
    The output retains geometry and PDF digest for visual review.
    """
    import fitz
    anchors = np.linspace(110, 492, 9)
    with fitz.open(pdf) as doc:
        if len(doc) != 1 or abs(doc[0].rect.width - 595.32) > 2:
            raise ValueError(f"Unsupported SAM page layout: {pdf}")
        marks = {"valence": [], "arousal": []}
        for drawing in doc[0].get_drawings():
            r = drawing["rect"]
            if drawing["type"] != "s" or r.y0 < 440:
                continue
            x = (r.x0 + r.x1) / 2
            for name, low, high in [("valence", 440, 550), ("arousal", 550, 700)]:
                center = (r.y0 + r.y1) / 2
                # A single ellipse can surround one choice in both SAM rows.
                belongs = low <= center < high or (r.y0 < 530 and r.y1 > 575)
                if belongs:
                    nearest = int(np.argmin(np.abs(anchors - x)))
                    distance = float(abs(anchors[nearest] - x))
                    if distance > 26:
                        raise ValueError(f"Ambiguous {name} mark: {pdf} (x={x:.1f})")
                    marks[name].append({"rating": nearest + 1, "rect": list(r), "distance": distance})
        result = {}
        for name, selections in marks.items():
            values = {m["rating"] for m in selections}
            if len(values) != 1:
                raise ValueError(f"Expected one {name} choice in {pdf}; got {sorted(values)}")
            result[name] = values.pop()
        result.update(pdf_sha256=sha256(pdf), marks=marks, method="SAM graphical selection v1")
        return result


def aligned_signal(signal: np.ndarray, source: list[str], target: list[str]):
    source = [s.upper() for s in source]
    if len(source) != signal.shape[0] or len(set(source)) != len(source):
        raise ValueError("Signal/channel-order mismatch")
    out = np.zeros((len(target), signal.shape[1]), dtype=np.float32)
    mask = np.zeros(len(target), dtype=bool)
    for i, name in enumerate(target):
        if name in source:
            out[i] = signal[source.index(name)]
            mask[i] = True
    if not mask.any() or not np.isfinite(out).all():
        raise ValueError("Missing/non-finite EEG channels")
    return out, mask


def audit(root: Path, output: Path, provenance: Path | None = None) -> dict:
    declared_sources = load_provenance(provenance) if provenance is not None else None
    paths = roots(root)
    seed = seed_channels(paths["SEEDIV"])
    ratings = []
    counts = {}
    deap_labels, reference_info = deap_reference(paths["DEAP"])
    changed = np.zeros(4, dtype=int)
    for name in DATASETS:
        labels, subjects, excluded, n_trials = [], set(), 0, 0
        if name == "DEAP":
            files = sorted((paths[name] / "data_preprocessed_python").glob("s*.dat"))
            if len(files) != 32:
                raise ValueError(f"Expected 32 DEAP subject files; found {len(files)}")
            for f in files:
                with f.open("rb") as stream:
                    data = pickle.load(stream, encoding="latin1")
                if data["data"].shape != (40, 40, 8064) or data["labels"].shape != (40, 4):
                    raise ValueError(f"Unexpected DEAP arrays in {f}")
                subjects.add(f.stem)
                reference = deap_labels[int(f.stem[1:])]
                changed += verify_deap_labels(data["labels"], reference)
                labels.extend(valence_label(float(v), name) for v in reference[:, 0])
                del data
        elif name == "GAMEEMO":
            for folder in sorted(paths[name].glob("(S*)")):
                subject = folder.name[1:-1]
                subjects.add(subject)
                for game in range(1, 5):
                    pdf = folder / "SAM Ratings" / f"G{game}.pdf"
                    rating = sam_rating(pdf)
                    csv = folder / "Preprocessed EEG Data" / ".csv format" / f"{subject}G{game}AllChannels.csv"
                    if not csv.is_file():
                        raise FileNotFoundError(csv)
                    labels.append(valence_label(rating["valence"], name))
                    ratings.append({"subject_id": f"GAMEEMO:{subject}", "trial_id": f"GAMEEMO:{subject}:G{game}",
                                    "pdf": str(pdf.resolve()), **rating})
            if len(subjects) != 28:
                raise ValueError("Expected 28 GAMEEMO participants")
        else:
            for session in range(1, 4):
                files = list((paths[name] / "eeg_raw_data" / str(session)).glob("*.mat"))
                ids = [int(f.stem.split("_")[0]) for f in files]
                if sorted(ids) != list(range(1, 16)):
                    raise ValueError(f"SEED-IV session {session}: expected all 15 subjects")
                for f in files:
                    metadata = {k: shape for k, shape, _ in whosmat(f)}
                    for k in numeric_eeg_keys(metadata):
                        if metadata[k][0] != 62:
                            raise ValueError(f"SEED-IV channel mismatch: {f}/{k}")
                    subjects.add(f.stem.split("_")[0])
                    labels.extend(valence_label(v, name) for v in SEED_LABELS[session])
        n_trials = len(labels)
        excluded = labels.count(None)
        counts[name] = {"subjects": len(subjects), "source_trials": n_trials, "excluded_neutral_trials": excluded,
                        "eligible_trials": n_trials - excluded, "trial_class_counts": dict(Counter(v for v in labels if v is not None))}
        print(f"Audited {name}: {counts[name]}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "gameemo_sam_ratings.json", ratings)
    pd.DataFrame([{k: r[k] for k in ("subject_id", "trial_id", "valence", "arousal", "pdf_sha256")} for r in ratings]).to_csv(output / "gameemo_sam_ratings.csv", index=False)
    result = {"label_policy": POLICY, "datasets": counts, "source_roots": {k: str(v) for k, v in paths.items()},
              "common_channels": COMMON_CHANNELS, "seed_channels": seed,
              "ignored": ["pre-existing augmented_datasets", "duplicate sead-4/seed_iv tree"],
              "deap_label_recovery": {**reference_info, "changed_counts_valence_arousal_dominance_liking": changed.tolist(),
                                      "observed_modification": "Changed ratings match 9 - spreadsheet original; use spreadsheet ratings"},
              "sam_review": "Geometry extracted; inspect PDFs/contact sheet before manuscript use."}
    if declared_sources is not None:
        result["source_provenance"] = declared_sources
    write_json(output / "dataset_audit.json", result)
    return result


def source_trials(root: Path, ratings: list[dict]):
    """Stream at most one participant/session file in RAM."""
    paths = roots(root)
    seed = seed_channels(paths["SEEDIV"])
    deap_labels, reference_info = deap_reference(paths["DEAP"])
    for f in sorted((paths["DEAP"] / "data_preprocessed_python").glob("s*.dat")):
        with f.open("rb") as stream:
            values = pickle.load(stream, encoding="latin1")
        checksum = sha256(f)
        subject = f"DEAP:{f.stem.upper()}"
        reference = deap_labels[int(f.stem[1:])]
        verify_deap_labels(values["labels"], reference)
        for trial, (signal, label) in enumerate(zip(values["data"], reference), start=1):
            yield {"dataset": "DEAP", "subject_id": subject, "session": "1", "trial_id": f"{subject}:T{trial:02d}",
                   "source": str(f.resolve()), "source_sha256": checksum, "source_key": str(trial), "original_label": float(label[0]),
                   "pickle_valence": float(values["labels"][trial - 1, 0]), "rating_metadata_sha256": reference_info["sha256"],
                   "label": valence_label(float(label[0]), "DEAP"), "baseline_removed_samples": 384}, signal[:32, 384:], DEAP_CHANNELS, 128
        del values
    rating_map = {r["trial_id"]: r for r in ratings}
    for folder in sorted(paths["GAMEEMO"].glob("(S*)")):
        subject = folder.name[1:-1]
        for game in range(1, 5):
            trial_id = f"GAMEEMO:{subject}:G{game}"
            rating = rating_map[trial_id]
            pdf = folder / "SAM Ratings" / f"G{game}.pdf"
            if sha256(pdf) != rating["pdf_sha256"]:
                raise ValueError(f"SAM PDF changed since audit: {pdf}")
            f = folder / "Preprocessed EEG Data" / ".csv format" / f"{subject}G{game}AllChannels.csv"
            signal = pd.read_csv(f, usecols=COMMON_CHANNELS)[COMMON_CHANNELS].to_numpy(dtype=np.float32).T
            yield {"dataset": "GAMEEMO", "subject_id": f"GAMEEMO:{subject}", "session": str(game), "trial_id": trial_id,
                   "source": str(f.resolve()), "source_sha256": sha256(f), "source_key": f"G{game}", "original_label": rating["valence"],
                   "label": valence_label(rating["valence"], "GAMEEMO"), "baseline_removed_samples": 0,
                   "rating_pdf_sha256": rating["pdf_sha256"]}, signal, COMMON_CHANNELS, 128
    for session in range(1, 4):
        files = sorted((paths["SEEDIV"] / "eeg_raw_data" / str(session)).glob("*.mat"), key=lambda f: int(f.stem.split("_")[0]))
        for f in files:
            values = loadmat(f)
            checksum = sha256(f)
            subject = f"SEEDIV:S{int(f.stem.split('_')[0]):02d}"
            for key in numeric_eeg_keys(values):
                trial = int(re.search(r"_eeg(\d+)$", key)[1])
                label = SEED_LABELS[session][trial - 1]
                yield {"dataset": "SEEDIV", "subject_id": subject, "session": str(session), "trial_id": f"{subject}:R{session}:T{trial:02d}",
                       "source": str(f.resolve()), "source_sha256": checksum, "source_key": key, "original_label": label,
                       "label": valence_label(label, "SEEDIV"), "baseline_removed_samples": 0}, values[key], seed, 200
            del values
