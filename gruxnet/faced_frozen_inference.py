"""Frozen FACED inference and target-free prediction locks.

This qualification backend accepts source or manufactured inputs only. It has
no recording, target, fitting or confirmation reader. Paths and byte hashes
come from the caller's separately frozen declaration; a hash authenticates
bytes, not the scientific provenance of the declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
import uuid
import zipfile

import numpy as np

from .faced_classical_contract import ACQUISITIONS, CATEGORIES
from .faced_coverage_missing import ENGINEERED_MISSING_REASONS
from .faced_proposal_preprocessing import COMMON_CHANNELS
from .faced_temporal_coverage import DEVELOPMENT_CLIPS, REGIONS

METHODS = ("category_median", "context_only", "late_joint", "distributed_joint")
CONTEXT_COLUMNS = tuple("category:" + c for c in CATEGORIES) + tuple(
    "acquisition:" + a for a in ACQUISITIONS
) + ("catalogue_duration_seconds", "trial")
ALPHA = 100.0
POLICY = "shared_three_region_qc_unchanged_thresholds_v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_METADATA_KEYS = frozenset(("subject", "vid", "role", "family", "assigned_category",
                            "acquisition_group", "catalogue_duration_seconds", "trial"))
_PREDICTION_KEYS = frozenset(("subject", "vid", "role", "family", "assigned_category",
    "acquisition_group", "model", "alpha", "prediction", "shared_eeg_available",
    "common_category_fallback", "routing", "qc_sha256", "input_sha256", "state_sha256"))


def canonical_bytes(value):
    """Canonical UTF-8 JSON with an intentional final newline; never NaN."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Require a lowercase SHA-256 byte hash")
    return value


def _source_pairs_sha256(pairs):
    """Existing producer identity hash: compact JSON, without final newline."""
    pairs = tuple(tuple(p) for p in pairs)
    if (len(pairs) != 980 or any(len(p) != 2 or not isinstance(p[0], str) or not p[0]
            or type(p[1]) is not int for p in pairs) or len(set(pairs)) != 980):
        raise PermissionError("Require 980 exact distinct source training identities")
    people = {p[0] for p in pairs}
    if len(people) != 70 or set(pairs) != {(p, v) for p in people for v in DEVELOPMENT_CLIPS}:
        raise PermissionError("Source training identities differ from the declared rectangle")
    ordered = sorted(pairs)
    if list(pairs) != ordered:
        raise PermissionError("Frozen source identity order must be canonical")
    return sha256_bytes(json.dumps(ordered, separators=(",", ":")).encode())


def _number(value, label, lower=None, upper=None):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Require a numeric " + label)
    number = float(value)
    if not math.isfinite(number) or (lower is not None and number < lower) or (
            upper is not None and number > upper):
        raise ValueError("Invalid finite " + label)
    return number


def _integer(value, label, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("Require an exact integer " + label)
    return value


def _array(values, shape, label):
    array = np.asarray(values)
    if array.dtype != np.dtype("float64") or array.shape != shape or not np.isfinite(array).all():
        raise ValueError("Require finite float64 " + label + " with shape " + str(shape))
    # The backing bytes are immutable, including against setflags(write=True).
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float64).reshape(shape)


def array_sha256(array):
    values = np.ascontiguousarray(array)
    return sha256_bytes(str(values.dtype).encode() + str(values.shape).encode() + values.tobytes())


def _object_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON field")
        result[name] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_object_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def _bound_bytes(path, expected_sha256, cap):
    _hash(expected_sha256)
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Require an existing non-symlink bound file")
    with path.open("rb") as stream:
        payload = stream.read(cap + 1)
    if len(payload) > cap or sha256_bytes(payload) != expected_sha256:
        raise ValueError("Bound file size or byte hash differs")
    return payload


@dataclass(frozen=True)
class FrozenRidge:
    model: str
    coefficient: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    variance: np.ndarray
    eeg_imputation_mean: np.ndarray | None
    state_sha256: str
    source_identity_sha256: str
    alpha: float = ALPHA
    solver: str = "svd"
    fit_intercept: bool = True

    def __post_init__(self):
        if self.model not in METHODS[1:] or self.alpha != ALPHA or self.solver != "svd" or self.fit_intercept is not True:
            raise ValueError("Only the frozen alpha-100 SVD/intercept recipe is permitted")
        _hash(self.state_sha256); _hash(self.source_identity_sha256)
        dimension = 13 if self.model == "context_only" else 163
        for field, shape in (("coefficient", (dimension,)), ("intercept", ()),
                             ("mean", (dimension,)), ("scale", (dimension,)), ("variance", (dimension,))):
            object.__setattr__(self, field, _array(getattr(self, field), shape, field))
        if np.any(self.scale <= 0) or np.any(self.variance < 0):
            raise ValueError("Scaler scale must be positive and variance nonnegative")
        expected_scale = np.sqrt(self.variance)
        # StandardScaler uses this numerical constant-feature bound for 980
        # training rows; a tiny nonzero variance can legitimately have scale 1.
        eps = np.finfo(np.float64).eps
        constant = self.variance <= 980 * eps * self.variance + (980 * self.mean * eps) ** 2
        expected_scale[constant] = 1.0
        if not np.array_equal(self.scale, expected_scale):
            raise ValueError("Scaler moments are inconsistent")
        if self.model == "context_only":
            if self.eeg_imputation_mean is not None:
                raise ValueError("Context state cannot contain an EEG imputation mean")
        else:
            object.__setattr__(self, "eeg_imputation_mean", _array(self.eeg_imputation_mean, (150,), "EEG mean"))

    def predict(self, features):
        """Evaluate frozen state algebra; this method does not impute or fit."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not len(x) or not np.isfinite(x).all():
            raise ValueError("Invalid finite frozen-model input geometry")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            values = ((x - self.mean) / self.scale) @ self.coefficient + self.intercept
        if not np.isfinite(values).all():
            raise ValueError("Frozen state produced nonfinite predictions")
        return np.clip(values, 0.0, 7.0)


def load_frozen_state(path, expected_sha256, record, source_identity_sha256):
    """Read one authenticated exact-schema numeric NPZ without pickle loading."""
    _hash(source_identity_sha256)
    model = record.get("model")
    dimension = 13 if model == "context_only" else 163
    if (model not in METHODS[1:] or record.get("stage") != "full_source_state_qualification"
            or record.get("alpha") != ALPHA or record.get("training_rows") != 980
            or record.get("feature_dimension") != dimension
            or record.get("source_identity_sha256") != source_identity_sha256
            or record.get("private_state_sha256") != expected_sha256):
        raise PermissionError("State binding is not the frozen full-source alpha-100 recipe")
    if _source_pairs_sha256(record.get("training_pairs", ())) != source_identity_sha256:
        raise PermissionError("State training identities do not match the frozen source identity hash")
    expected = {"coefficient", "intercept", "mean", "scale", "variance", "alpha"}
    if model != "context_only":
        expected.add("eeg_imputation_mean")
    payload = _bound_bytes(path, expected_sha256, 65536)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if (len(members) != len(expected) or {m.filename for m in members} != {k + ".npy" for k in expected}
                or sum(m.file_size for m in members) > 65536 or any(m.flag_bits & 1 for m in members)):
            raise ValueError("Unexpected, duplicate, encrypted or oversized state archive members")
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        if len(saved.files) != len(expected) or set(saved.files) != expected:
            raise ValueError("Unexpected state keys")
        values = {name: saved[name] for name in expected}
    alpha = _array(values.pop("alpha"), (), "alpha")
    if float(alpha) != ALPHA:
        raise ValueError("State alpha differs from declared recipe")
    return FrozenRidge(model=model, state_sha256=expected_sha256,
                       source_identity_sha256=source_identity_sha256,
                       eeg_imputation_mean=values.pop("eeg_imputation_mean", None), **values)


@dataclass(frozen=True)
class FrozenBundle:
    states: object
    category_prior: object
    prior_sha256: str
    source_identity_sha256: str

    def __post_init__(self):
        _hash(self.prior_sha256); _hash(self.source_identity_sha256)
        if set(self.states) != set(METHODS[1:]) or set(self.category_prior) != set(CATEGORIES):
            raise ValueError("Require all three frozen states and eight supported category priors")
        states = dict(self.states)
        for name, state in states.items():
            if not isinstance(state, FrozenRidge) or state.model != name or state.source_identity_sha256 != self.source_identity_sha256:
                raise ValueError("States disagree on model or source identities")
        context = states["context_only"]
        for model in METHODS[2:]:
            for field in ("mean", "variance", "scale"):
                if getattr(context, field).tobytes() != getattr(states[model], field)[:13].tobytes():
                    raise ValueError("Matched context scaler prefixes differ")
        priors = {c: _number(self.category_prior[c], "category prior", 0, 7) for c in CATEGORIES}
        object.__setattr__(self, "states", MappingProxyType(states))
        object.__setattr__(self, "category_prior", MappingProxyType(priors))

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_bytes({"states": {m: s.state_sha256 for m, s in self.states.items()},
            "prior": self.prior_sha256, "source_identity": self.source_identity_sha256,
            "alpha": ALPHA, "solver": "svd", "fit_intercept": True,
            "context_columns": CONTEXT_COLUMNS, "shared_missing_policy": POLICY}))


def load_frozen_bundle(state_bindings, prior_path, prior_sha256, source_identity_sha256):
    """Load already-fitted states. Each binding has path, sha256 and fit record."""
    if set(state_bindings) != set(METHODS[1:]):
        raise PermissionError("Exactly three fitted state bindings are required")
    states = {}
    for model in METHODS[1:]:
        binding = state_bindings[model]
        if set(binding) != {"path", "sha256", "record"} or binding["record"].get("model") != model:
            raise ValueError("Invalid model state binding")
        states[model] = load_frozen_state(binding["path"], binding["sha256"], binding["record"], source_identity_sha256)
    prior = _json(_bound_bytes(prior_path, prior_sha256, 65536))
    if (not isinstance(prior, dict) or set(prior) != {"global", "coarse", "category", "video"}
            or not isinstance(prior["category"], dict) or set(prior["category"]) != set(CATEGORIES)
            or not isinstance(prior["coarse"], dict) or set(prior["coarse"]) != {"negative", "positive"}
            or not isinstance(prior["video"], dict) or set(prior["video"]) != {str(v) for v in DEVELOPMENT_CLIPS}):
        raise ValueError("Unexpected full-source prior schema or support")
    _number(prior["global"], "global prior", 0, 7)
    for key in ("coarse", "category", "video"):
        for value in prior[key].values():
            _number(value, key + " prior", 0, 7)
    return FrozenBundle(states, prior["category"], prior_sha256, source_identity_sha256)


def qualify_metadata(metadata_rows, mode, source_identity_sha256=None):
    """Guard the complete declared roster before any EEG callback is invoked."""
    if mode not in ("source", "manufactured"):
        raise PermissionError("Confirmation/validation access is not activated by this backend")
    rows = tuple(dict(row) for row in metadata_rows)
    if not rows or len(rows) > 980:
        raise PermissionError("Require a bounded nonempty qualification roster")
    identities, titles, people = set(), {}, set()
    # Global identity/role/schema barrier precedes any numeric conversion.
    for row in rows:
        if set(row) != _METADATA_KEYS:
            raise PermissionError("Metadata must contain only the exact target-free schema")
        if (not isinstance(row["subject"], str) or not row["subject"] or row["subject"].strip() != row["subject"]
                or type(row["vid"]) is not int or not 1 <= row["vid"] <= 28
                or row["role"] != mode):
            raise PermissionError("Malformed, validation or reserved identity")
        pair = row["subject"], row["vid"]
        if pair in identities:
            raise PermissionError("Duplicate declared trial")
        identities.add(pair); people.add(pair[0])
        if not isinstance(row["family"], str) or not row["family"] or row["assigned_category"] not in CATEGORIES or row["acquisition_group"] not in ACQUISITIONS:
            raise PermissionError("Malformed catalogue family/category/acquisition")
        definition = row["family"], row["assigned_category"]
        if pair[1] in titles and titles[pair[1]] != definition:
            raise PermissionError("Catalogue title definition varies across people")
        titles[pair[1]] = definition
    if mode == "source" and (len(rows) != 980 or len(people) != 70
            or identities != {(p, v) for p in people for v in DEVELOPMENT_CLIPS}):
        raise PermissionError("Source qualification requires exactly 70 by 14 source trials")
    if mode == "source" and len({r["family"] for r in rows}) != 14:
        raise PermissionError("Source development title-family scope differs")
    if mode == "source" and (source_identity_sha256 is None or
            _source_pairs_sha256(sorted(identities)) != _hash(source_identity_sha256)):
        raise PermissionError("Exact source participant/trial identities differ from frozen fitted states")
    for row in rows:
        _number(row["catalogue_duration_seconds"], "catalogue duration", 0.000001)
        _integer(row["trial"], "trial ordinal", 1, 28)
    durations, acquisitions, ordinals = {}, {}, set()
    for row in rows:
        vid, person = row["vid"], row["subject"]
        duration = float(row["catalogue_duration_seconds"])
        if vid in durations and durations[vid] != duration:
            raise PermissionError("Catalogue duration varies across people")
        if person in acquisitions and acquisitions[person] != row["acquisition_group"]:
            raise PermissionError("Participant acquisition group changes between trials")
        if (person, row["trial"]) in ordinals:
            raise PermissionError("Duplicate participant trial ordinal")
        durations[vid] = duration; acquisitions[person] = row["acquisition_group"]
        ordinals.add((person, row["trial"]))
    if mode == "source":
        counts = {c: sum(definition[1] == c for definition in titles.values()) for c in CATEGORIES}
        if counts != {c: (1 if c in ("Fear", "Inspiration") else 2) for c in CATEGORIES}:
            raise PermissionError("Source category/title support differs")
    return rows


def _context(row):
    return np.asarray([float(row["assigned_category"] == c) for c in CATEGORIES]
        + [float(row["acquisition_group"] == a) for a in ACQUISITIONS]
        + [float(row["catalogue_duration_seconds"]), float(row["trial"])], dtype=np.float64)


def qualify_quality(metadata_rows, quality_rows):
    """Authenticate exact three-region QC, preserving engineered failure flags."""
    metadata = {(r["subject"], r["vid"]): r for r in metadata_rows}
    quality = {}
    for raw in quality_rows:
        row = dict(raw)
        required = {"subject", "vid", "role", "region", "technical_passed", "review_flags"}
        if set(row) not in (required, required | {"eeg_available"}):
            raise PermissionError("QC must have the exact target-free schema")
        if not isinstance(row["subject"], str) or type(row["vid"]) is not int:
            raise PermissionError("Malformed QC identity")
        pair = row["subject"], row["vid"]
        if pair not in metadata or row["role"] != metadata[pair]["role"] or row["region"] not in REGIONS:
            raise PermissionError("Undeclared QC identity, role or region")
        key = (*pair, row["region"])
        if key in quality:
            raise PermissionError("Duplicate regional QC record")
        if row["technical_passed"] is not True or not isinstance(row["review_flags"], list):
            raise PermissionError("Technical/numerical QC failures cannot become missing EEG")
        for flag in row["review_flags"]:
            if not isinstance(flag, dict) or set(flag) != {"channel", "reason"}:
                raise ValueError("Require literal channel/reason QC flags")
            if not isinstance(flag["reason"], str) or flag["reason"] not in ENGINEERED_MISSING_REASONS:
                raise PermissionError("Unknown QC reason cannot become missing EEG")
            channels = {"all"} if flag["reason"] == "repair_capacity_exceeded" else set(COMMON_CHANNELS)
            if not isinstance(flag["channel"], str) or flag["channel"] not in channels:
                raise PermissionError("QC flag channel differs from unchanged policy")
        if "eeg_available" in row and (type(row["eeg_available"]) is not bool or row["eeg_available"] != (not row["review_flags"])):
            raise PermissionError("Regional availability disagrees with QC flags")
        quality[key] = row
    if set(quality) != {(*p, region) for p in metadata for region in REGIONS}:
        raise PermissionError("Every declared trial needs exactly early, middle and late QC")
    available = {p: all(not quality[(*p, region)]["review_flags"] for region in REGIONS) for p in metadata}
    hashes = {p: sha256_bytes(canonical_bytes([quality[(*p, region)] for region in REGIONS])) for p in metadata}
    return available, hashes


def infer_four_methods(bundle, metadata_rows, quality_rows, feature_loader, *, mode):
    """Produce every target-free method row in declared trial/method order.

    Callback signature is ``feature_loader((subject, vid), region)`` and is
    never invoked for any region of an unavailable trial. The complete roster
    and all QC are checked before the first callback. No imputation happens at
    prediction time: matched context and both EEG models share the category
    fallback; practical category always uses that prior.
    """
    if not isinstance(bundle, FrozenBundle):
        raise TypeError("Require an authenticated FrozenBundle")
    quality_rows = tuple(quality_rows)
    rows = qualify_metadata(metadata_rows, mode, bundle.source_identity_sha256)
    available, qc_hashes = qualify_quality(rows, quality_rows)
    features = {}
    for row in rows:
        pair = row["subject"], row["vid"]
        if not available[pair]:
            continue
        regional = [_array(feature_loader(pair, region), (150,), "regional EEG") for region in REGIONS]
        context = _context(row)
        distributed = np.stack(regional).mean(axis=0)
        if not np.isfinite(distributed).all():
            raise ValueError("Nonfinite distributed feature is fatal")
        features[pair] = {"context_only": context,
            "late_joint": np.concatenate((context, regional[2])),
            "distributed_joint": np.concatenate((context, distributed))}
    # All available feature callbacks have passed before producing any rows.
    predictions = {m: {} for m in METHODS[1:]}
    available_pairs = [(r["subject"], r["vid"]) for r in rows if available[(r["subject"], r["vid"])]]
    if available_pairs:
        for method in METHODS[1:]:
            values = bundle.states[method].predict(np.stack([features[p][method] for p in available_pairs]))
            predictions[method] = dict(zip(available_pairs, values))
    result = []
    for row in rows:
        pair = row["subject"], row["vid"]
        prior = bundle.category_prior[row["assigned_category"]]
        for method in METHODS:
            fallback = method != "category_median" and not available[pair]
            value = prior if method == "category_median" or fallback else predictions[method][pair]
            result.append({k: row[k] for k in ("subject", "vid", "role", "family", "assigned_category", "acquisition_group")} | {
                "model": method, "alpha": None if method == "category_median" else ALPHA,
                "prediction": float(value), "shared_eeg_available": available[pair],
                "common_category_fallback": fallback,
                "routing": "category_prior" if method == "category_median" else ("common_category_fallback" if fallback else "frozen_ridge"),
                "qc_sha256": qc_hashes[pair],
                "input_sha256": None if method == "category_median" or fallback else array_sha256(features[pair][method]),
                "state_sha256": bundle.prior_sha256 if method == "category_median" else bundle.states[method].state_sha256})
    validate_prediction_rows(result, rows, mode=mode, bundle=bundle, quality_rows=quality_rows)
    return result


def validate_prediction_rows(prediction_rows, metadata_rows, *, mode, bundle, quality_rows):
    """Require exact identities, four methods, order, denominators and routing."""
    metadata = qualify_metadata(metadata_rows, mode, bundle.source_identity_sha256)
    availability, qc_hashes = qualify_quality(metadata, quality_rows)
    rows = list(prediction_rows)
    if len(rows) != 4 * len(metadata):
        raise PermissionError("All four predictions for every declared trial are required")
    for index, declared in enumerate(metadata):
        group = rows[4 * index:4 * index + 4]
        pair = declared["subject"], declared["vid"]
        for method, row in zip(METHODS, group):
            if not isinstance(row, dict) or set(row) != _PREDICTION_KEYS:
                raise PermissionError("Prediction lock must use the exact target-free schema")
            for key in ("subject", "vid", "role", "family", "assigned_category", "acquisition_group"):
                if type(row[key]) is not type(declared[key]) or row[key] != declared[key]:
                    raise PermissionError("Prediction identity, role, family or order differs")
            if row["model"] != method or row["alpha"] != (None if method == "category_median" else ALPHA):
                raise PermissionError("Prediction method order or frozen alpha differs")
            _number(row["prediction"], "clipped prediction", 0, 7)
            _hash(row["qc_sha256"]); _hash(row["state_sha256"])
            if type(row["shared_eeg_available"]) is not bool or type(row["common_category_fallback"]) is not bool:
                raise ValueError("Require Boolean shared availability/fallback")
            if row["shared_eeg_available"] != availability[pair] or row["qc_sha256"] != qc_hashes[pair]:
                raise PermissionError("Prediction QC binding differs from declared regional records")
            fallback = method != "category_median" and not row["shared_eeg_available"]
            routing = "category_prior" if method == "category_median" else ("common_category_fallback" if fallback else "frozen_ridge")
            if row["common_category_fallback"] != fallback or row["routing"] != routing:
                raise PermissionError("Prediction routing disagrees with shared QC")
            expected_state = bundle.prior_sha256 if method == "category_median" else bundle.states[method].state_sha256
            if row["state_sha256"] != expected_state:
                raise PermissionError("Prediction state binding differs")
            if method == "category_median" or fallback:
                if row["input_sha256"] is not None or row["prediction"] != bundle.category_prior[declared["assigned_category"]]:
                    raise PermissionError("Category route must equal the frozen prior without EEG input")
            else:
                _hash(row["input_sha256"])
        if len({r["shared_eeg_available"] for r in group}) != 1 or len({r["qc_sha256"] for r in group}) != 1:
            raise PermissionError("Methods disagree on shared availability or QC provenance")
    return rows, metadata


def _write_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def export_prediction_lock(output_directory, prediction_rows, metadata_rows, quality_rows, *, mode, bundle):
    """Publish a complete byte-bound lock; never replace or resume an output.

    A permanent exclusive reservation blocks concurrent producers and partial
    attempts. A complete staging directory is renamed only after all files are
    flushed. Readers accept a lock only after verifying every byte binding.
    Failed staging/reservation evidence is preserved for investigation.
    """
    quality_rows = tuple(quality_rows)
    rows, metadata = validate_prediction_rows(prediction_rows, metadata_rows, mode=mode, bundle=bundle, quality_rows=quality_rows)
    output = Path(output_directory)
    if not output.name or output.exists() or output.is_symlink():
        raise FileExistsError("Preserve an existing or partial prediction lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.parent / ("." + output.name + ".reservation")
    _write_new(reservation, canonical_bytes({"output_name": output.name, "bundle_sha256": bundle.fingerprint}))
    staging = output.parent / ("." + output.name + ".staging-" + uuid.uuid4().hex)
    staging.mkdir()
    payload = canonical_bytes(rows)
    manifest = {"schema": "FACED_TARGET_FREE_FOUR_METHOD_LOCK/v1", "mode": mode,
        "bundle_sha256": bundle.fingerprint, "roster_sha256": sha256_bytes(canonical_bytes(list(metadata))),
        "prediction_sha256": sha256_bytes(payload), "trials": len(metadata), "predictions": len(rows),
        "method_order": list(METHODS), "context_columns": list(CONTEXT_COLUMNS),
        "shared_missing_policy": POLICY, "available_trials": sum(rows[i]["shared_eeg_available"] for i in range(0, len(rows), 4)),
        "fallback_trials": sum(not rows[i]["shared_eeg_available"] for i in range(0, len(rows), 4)),
        "targets_present": False, "confirmation_activated": False}
    manifest_payload = canonical_bytes(manifest)
    lock = {"manifest_sha256": sha256_bytes(manifest_payload), "prediction_sha256": sha256_bytes(payload)}
    _write_new(staging / "predictions.json", payload)
    _write_new(staging / "manifest.json", manifest_payload)
    _write_new(staging / "LOCK.complete.json", canonical_bytes(lock))
    if output.exists():
        raise FileExistsError("Output appeared during lock publication; preserve staging")
    os.rename(staging, output)
    return verify_prediction_lock(output, metadata, quality_rows, mode=mode, bundle=bundle)


def verify_prediction_lock(output_directory, metadata_rows, quality_rows, *, mode, bundle):
    """Recheck complete directory, canonical bytes, bindings and exact roster."""
    output = Path(output_directory)
    if not output.is_dir() or output.is_symlink() or {p.name for p in output.iterdir()} != {
            "predictions.json", "manifest.json", "LOCK.complete.json"}:
        raise PermissionError("Require an exact completed prediction lock directory")
    payloads = {}
    for name in ("predictions.json", "manifest.json", "LOCK.complete.json"):
        path = output / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
            raise PermissionError("Unexpected lock file type or size")
        payloads[name] = path.read_bytes()
    objects = {name: _json(data) for name, data in payloads.items()}
    if any(canonical_bytes(objects[name]) != data for name, data in payloads.items()):
        raise ValueError("Lock JSON is not canonical")
    lock = objects["LOCK.complete.json"]
    if set(lock) != {"manifest_sha256", "prediction_sha256"} or lock != {
            "manifest_sha256": sha256_bytes(payloads["manifest.json"]),
            "prediction_sha256": sha256_bytes(payloads["predictions.json"])}:
        raise ValueError("Prediction lock byte hash mismatch")
    rows, metadata = validate_prediction_rows(objects["predictions.json"], metadata_rows, mode=mode, bundle=bundle, quality_rows=quality_rows)
    expected = {"schema": "FACED_TARGET_FREE_FOUR_METHOD_LOCK/v1", "mode": mode,
        "bundle_sha256": bundle.fingerprint, "roster_sha256": sha256_bytes(canonical_bytes(list(metadata))),
        "prediction_sha256": sha256_bytes(payloads["predictions.json"]), "trials": len(metadata), "predictions": len(rows),
        "method_order": list(METHODS), "context_columns": list(CONTEXT_COLUMNS),
        "shared_missing_policy": POLICY, "available_trials": sum(rows[i]["shared_eeg_available"] for i in range(0, len(rows), 4)),
        "fallback_trials": sum(not rows[i]["shared_eeg_available"] for i in range(0, len(rows), 4)),
        "targets_present": False, "confirmation_activated": False}
    if canonical_bytes(objects["manifest.json"]) != canonical_bytes(expected):
        raise PermissionError("Lock manifest bindings or denominators differ")
    return {"passed": True, "mode": mode, "trials": len(metadata), "predictions": len(rows),
            "manifest_sha256": lock["manifest_sha256"], "prediction_sha256": lock["prediction_sha256"],
            "bundle_sha256": bundle.fingerprint, "targets_present": False, "confirmation_activated": False}
