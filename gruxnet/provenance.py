"""Compare supplied reference DEAP files without altering either release or caches."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import pickle
from zipfile import ZipFile

import numpy as np

from .data import deap_reference, roots, sha256, write_json


def signal_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    checksum = hashlib.sha256(json.dumps({"shape": array.shape, "dtype": array.dtype.str}).encode())
    checksum.update(memoryview(contiguous).cast("B"))
    return checksum.hexdigest()


def compare_subject(local: dict, reference: dict, ratings: np.ndarray) -> dict:
    left, right = local["data"], reference["data"]
    if left.shape != right.shape or left.shape[0] != len(ratings):
        raise ValueError("Reference and local EEG shapes or trial counts differ")
    if local["labels"].shape != ratings.shape or reference["labels"].shape != ratings.shape:
        raise ValueError("Reference and local rating dimensions differ")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Non-finite EEG in the supplied comparison files")
    equal = np.array_equal(left, right)
    difference = 0.0 if equal else max(float(np.max(np.abs(a - b))) for a, b in zip(left, right))
    return {"signal_values_identical": bool(equal), "signal_shape": list(left.shape),
            "local_signal_dtype": left.dtype.str, "reference_signal_dtype": right.dtype.str,
            "local_signal_sha256": signal_digest(left), "reference_signal_sha256": signal_digest(right),
            "maximum_absolute_signal_difference": difference,
            "local_rating_differences_from_spreadsheet": (~np.isclose(local["labels"], ratings, rtol=0, atol=1e-8)).sum(axis=0).tolist(),
            "reference_rating_differences_from_spreadsheet": (~np.isclose(reference["labels"], ratings, rtol=0, atol=1e-8)).sum(axis=0).tolist()}


def compare_deap(official: Path, data_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Comparison already exists: {output}. Choose a fresh output file.")
    local_root = roots(data_root)["DEAP"]
    ratings, metadata = deap_reference(local_root)
    official = official.resolve()
    archive = ZipFile(official) if official.is_file() else None
    directory = official / "data_preprocessed_python" if (official / "data_preprocessed_python").is_dir() else official
    subjects = []
    try:
        for subject in range(1, 33):
            name = f"s{subject:02d}.dat"
            if archive is None:
                reference_file = directory / name
                raw = reference_file.read_bytes()
                reference_location = str(reference_file)
            else:
                members = [m for m in archive.infolist() if not m.is_dir() and PurePosixPath(m.filename).name == name]
                if len(members) != 1:
                    raise ValueError(f"Expected exactly one {name} in the supplied ZIP; found {len(members)}")
                raw = archive.read(members[0])
                reference_location = members[0].filename
            reference_file_sha256 = hashlib.sha256(raw).hexdigest()
            reference = pickle.load(io.BytesIO(raw), encoding="latin1")
            del raw
            local_file = local_root / "data_preprocessed_python" / name
            with local_file.open("rb") as stream:
                local = pickle.load(stream, encoding="latin1")
            for label, data in [("local", local), ("reference", reference)]:
                if data["data"].shape != (40, 40, 8064) or data["labels"].shape != (40, 4):
                    raise ValueError(f"Unexpected {label} DEAP subject dimensions: {name}")
            record = {"subject": name, "local_file": str(local_file), "reference_file": reference_location,
                      "local_file_sha256": sha256(local_file), "reference_file_sha256": reference_file_sha256,
                      **compare_subject(local, reference, ratings[subject])}
            subjects.append(record)
            print(f"Compared {name}: signal identical={record['signal_values_identical']}; "
                  f"reference rating differences={record['reference_rating_differences_from_spreadsheet']}", flush=True)
            del reference, local
    finally:
        if archive is not None:
            archive.close()
    result = {"reference_input": str(official), "metadata": metadata, "subjects": subjects,
              "all_32_subject_signals_identical": all(s["signal_values_identical"] for s in subjects),
              "all_reference_ratings_match_spreadsheet": all(not any(s["reference_rating_differences_from_spreadsheet"]) for s in subjects),
              "scope": "All 40 trials, 40 channels, and 8064 samples per subject, including baseline and peripheral channels",
              "signal_hash_format": "SHA-256 of JSON shape/dtype followed by contiguous array bytes; numeric equality is also checked independently",
              "origin_limit": "The reference path is user-supplied. This comparator verifies contents; first-party download provenance must be established separately."}
    write_json(output, result)
    return result
