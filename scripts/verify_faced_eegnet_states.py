"""Post-completion, CPU-only FACED EEGNet state-dictionary authentication.

Reads public execution/fit/selection receipts and the sixty named private states.
No waveform, rating, target/prediction CSV, model construction or training I/O.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import unittest

import torch

REPO = Path(__file__).resolve().parents[1]
PACKET_NAME = "faced_eegnet_diagnostic_v2_2026-10-11"
READOUTS = ("eeg_only", "context_residual")
RATES = (.0003, .001, .003)
SEEDS = (20261101, 20261102)
EPOCHS = (30, 60, 120)
PARAMETER_COUNT = 1841
NORM_TOLERANCE = 1e-6
STATE_SHAPES = {
    "temporal.weight": (8, 1, 1, 64),
    "spatial.weight": (16, 1, 30, 1),
    "depthwise.weight": (16, 1, 1, 16),
    "pointwise.weight": (16, 16, 1, 1),
    "head.weight": (1, 256),
    "head.bias": (1,),
}
for _name, _channels in (("bn1", 8), ("bn2", 16), ("bn3", 16)):
    for _field in ("weight", "bias", "running_mean", "running_var"):
        STATE_SHAPES[f"{_name}.{_field}"] = (_channels,)
    STATE_SHAPES[f"{_name}.num_batches_tracked"] = ()
PARAMETER_KEYS = tuple(k for k in STATE_SHAPES if k.endswith((".weight", ".bias")))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bounded_path(directory, name):
    directory = Path(directory).resolve(strict=True)
    if not isinstance(name, str) or not name or name in (".", "..") or any(c in name for c in "/\\:"):
        raise PermissionError("Require a private/public packet basename")
    candidate = (directory / name).resolve()
    if candidate.parent != directory:
        raise PermissionError("Resolved file leaves its named packet")
    return candidate


def state_geometry(state, epoch):
    if epoch not in EPOCHS or not isinstance(state, (dict, OrderedDict)) or set(state) != set(STATE_SHAPES):
        raise ValueError("Changed frozen scalar-adapter state keys or epoch")
    for key, shape in STATE_SHAPES.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided or tuple(value.shape) != shape:
            raise ValueError("Invalid CPU state tensor geometry: " + key)
        expected_dtype = torch.int64 if key.endswith("num_batches_tracked") else torch.float32
        if value.dtype != expected_dtype or not torch.isfinite(value).all().item():
            raise ValueError("Invalid state dtype/finiteness: " + key)
        if key.endswith("running_var") and torch.any(value < 0).item():
            raise ValueError("Negative BN running variance")
        if key.endswith("num_batches_tracked") and value.item() != 70 * epoch:
            raise ValueError("BN counter differs from 560-source-trial/batch-eight schedule")
    count = sum(state[k].numel() for k in PARAMETER_KEYS)
    if count != PARAMETER_COUNT:
        raise ValueError("Scalar EEGNet parameter count changed")
    head = float(state["head.weight"].square().sum(dim=1).sqrt().max())
    spatial = float(state["spatial.weight"].square().sum(dim=2).sqrt().max())
    if head > .25 + NORM_TOLERANCE or spatial > 1 + NORM_TOLERANCE:
        raise ValueError("Post-update head/spatial constraints exceeded")
    return {"state_tensors": len(state), "trainable_parameters": count,
            "bn_batches_tracked": 70 * epoch, "head_max_l2": head,
            "spatial_max_electrode_l2": spatial, "all_tensors_finite": True,
            "all_tensors_cpu": True, "float_tensor_dtype": "float32"}


def authenticate_state(directory, name, expected_sha256, epoch):
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Malformed checkpoint SHA-256")
    path = bounded_path(directory, name)
    if not path.is_file() or path.suffix != ".pt" or sha(path) != expected_sha256:
        raise ValueError("Private checkpoint SHA-256 differs from public fit receipt")
    # No unsafe-pickle fallback, constructor, RNG changes or accelerator allocation.
    with path.open("rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if sha(path) != expected_sha256:
        raise ValueError("Checkpoint changed during CPU authentication")
    return {"private_state_file": name, "sha256": expected_sha256,
            "hash_matches": True, "weights_only_cpu_load": True,
            **state_geometry(state, epoch)}


def completed_records(execution, fit_packet, selection_packet):
    if execution.get("status") != "completed" or execution.get("failure") is not None:
        raise PermissionError("State audit requires a completed diagnostic")
    if (execution.get("new_neural_fits"), execution.get("completed_neural_fits"), execution.get("fits")) != (20, 20, 20):
        raise PermissionError("Require all twenty completed neural fits")
    if execution.get("confirmation_access") is not False or execution.get("research_question_changed") is not False:
        raise PermissionError("Protected/paper contract changed")
    selection = selection_packet.get("selection")
    if not isinstance(selection, dict) or set(selection) != set(READOUTS) or execution.get("selection") != selection:
        raise ValueError("Missing or inconsistent two-readout schedule selection")
    for schedule in selection.values():
        if schedule.get("learning_rate") not in RATES or schedule.get("epoch") not in EPOCHS:
            raise ValueError("Undeclared selected schedule")
    expected = {(0, head, rate, seed) for head in READOUTS for rate in RATES for seed in SEEDS}
    expected |= {(rotation, head, selection[head]["learning_rate"], seed)
                 for rotation in (1, 2) for head in READOUTS for seed in SEEDS}
    records = fit_packet.get("fits")
    if not isinstance(records, list) or len(records) != 20:
        raise ValueError("Incomplete checkpoint fit ledger")
    keyed = {}
    for record in records:
        key = record["rotation"], record["readout"], record["learning_rate"], record["seed"]
        if key in keyed or key not in expected or record.get("status") != "completed":
            raise ValueError("Duplicate, changed or incomplete checkpoint fit identity")
        if set(record.get("checkpoints", {})) != {str(e) for e in EPOCHS}:
            raise ValueError("Three complete checkpoints required per fit")
        history = record.get("history", [])
        if len(history) != 120 or [h.get("epoch") for h in history] != list(range(1, 121)):
            raise ValueError("Incomplete fixed training history")
        keyed[key] = record
    if set(keyed) != expected:
        raise ValueError("Changed twenty-fit diagnostic grid")
    return keyed


def audit_states(repo=REPO):
    repo = Path(repo).resolve(strict=True)
    public = (repo / "results/development" / PACKET_NAME).resolve(strict=True)
    if not public.is_relative_to(repo):
        raise PermissionError("Public packet leaves repository")
    execution_path = bounded_path(public, "execution.json")
    execution_sha = sha(execution_path)
    execution = load_json(execution_path)
    # The completion gate precedes even opening the private packet directory.
    if execution.get("status") != "completed":
        raise PermissionError("No private-state access before completed execution")
    fit_path = bounded_path(public, "fit_records.json")
    selection_path = bounded_path(public, "selection.json")
    fit_sha = sha(fit_path); selection_sha = sha(selection_path)
    records = completed_records(execution, load_json(fit_path), load_json(selection_path))
    private = (repo.parent / "publication_runs" / PACKET_NAME).resolve(strict=True)
    private_parent = (repo.parent / "publication_runs").resolve(strict=True)
    if private.parent != private_parent or not private.is_relative_to(repo.parent):
        raise PermissionError("Private checkpoint packet leaves named workspace directory")
    activation_path = bounded_path(public, "activation.json")
    activation_sha = sha(activation_path)
    if execution.get("activation_sha256") != activation_sha:
        raise ValueError("Execution activation hash mismatch")
    activation = load_json(activation_path)
    for rel in ("gruxnet/eegnet_regression.py", "gruxnet/faced_eegnet_runner.py"):
        path = (repo / rel).resolve(strict=True)
        if not path.is_relative_to(repo) or activation["code_bindings"].get(rel) != sha(path):
            raise ValueError("Frozen scalar-adapter/engine source changed")
    names = set(); checks = []
    for (rotation, head, rate, seed), record in sorted(records.items()):
        for epoch in EPOCHS:
            checkpoint = record["checkpoints"][str(epoch)]
            expected_name = f"r{rotation}_{head}_lr{rate}_s{seed}_e{epoch}.pt"
            name = checkpoint.get("private_state_file")
            if name != expected_name or name in names:
                raise PermissionError("Changed/duplicate canonical checkpoint basename")
            names.add(name)
            check = authenticate_state(private, name, checkpoint["state_dict_sha256"], epoch)
            checks.append({"rotation": rotation, "readout": head, "learning_rate": rate,
                           "seed": seed, "epoch": epoch, **check})
    if len(checks) != 60:
        raise ValueError("Exactly sixty authenticated checkpoints required")
    if any(sha(path) != expected for path, expected in
           ((execution_path, execution_sha), (fit_path, fit_sha),
            (selection_path, selection_sha), (activation_path, activation_sha))):
        raise ValueError("Public checkpoint evidence changed during authentication")
    return {"scope": "Post-completion CPU state-dictionary authentication only",
            "execution_sha256": execution_sha, "fit_records_sha256": fit_sha,
            "selection_sha256": selection_sha, "activation_sha256": activation_sha,
            "completed_neural_fits": 20, "checkpoints_authenticated": 60,
            "expected_trainable_parameters": PARAMETER_COUNT, "norm_tolerance": NORM_TOLERANCE,
            "maximum_head_l2": max(c["head_max_l2"] for c in checks),
            "maximum_spatial_electrode_l2": max(c["spatial_max_electrode_l2"] for c in checks),
            "states": checks, "checkpoint_weights_loaded_on_cpu_only": True,
            "model_constructed": False, "random_number_generator_used": False,
            "raw_signal_or_rating_files_opened": False, "target_or_prediction_csvs_opened": False,
            "training_or_forward_passes_executed": False,
            "confirmation_access": False, "research_question_changed": False}


def synthetic_state(epoch=30):
    state = {key: torch.zeros(shape, dtype=torch.int64 if key.endswith("num_batches_tracked") else torch.float32)
             for key, shape in STATE_SHAPES.items()}
    for key in state:
        if key.endswith("num_batches_tracked"):
            state[key].fill_(70 * epoch)
        elif key.endswith("running_var"):
            state[key].fill_(1)
    return state


@contextmanager
def synthetic_directory():
    # Ordinary workspace mkdir avoids this Windows environment's mkdtemp ACL issue.
    directory = REPO.parent / f"_egstate_test_{os.getpid()}"
    directory.mkdir()
    try:
        yield directory
    finally:
        if directory.resolve().parent != REPO.parent.resolve():
            raise PermissionError("Synthetic cleanup leaves intended workspace")
        path = directory / "synthetic.pt"
        if path.exists():
            path.unlink()
        directory.rmdir()  # Empty directory only; no recursive deletion.


class SyntheticStateTests(unittest.TestCase):
    """Fabricated state tensors; no original checkpoint or data access."""

    def test_valid_frozen_geometry(self):
        result = state_geometry(synthetic_state(), 30)
        self.assertEqual((result["state_tensors"], result["trainable_parameters"]), (21, 1841))

    def test_dtype_geometry_and_nonfinite_failures(self):
        for value in (torch.zeros(2, 256), torch.zeros(1, 256, dtype=torch.float64), torch.full((1, 256), float("nan"))):
            state = synthetic_state(); state["head.weight"] = value
            with self.assertRaises(ValueError):
                state_geometry(state, 30)

    def test_bn_counter_variance_and_constraints(self):
        changes = (("bn2.num_batches_tracked", torch.tensor(1)), ("bn1.running_var", -torch.ones(8)),
                   ("head.weight", torch.ones(1, 256)), ("spatial.weight", torch.ones(16, 1, 30, 1)))
        for key, value in changes:
            state = synthetic_state(); state[key] = value
            with self.assertRaises(ValueError):
                state_geometry(state, 30)

    def test_safe_serialization_and_bad_sha(self):
        with synthetic_directory() as directory:
            path = Path(directory) / "synthetic.pt"
            with path.open("xb") as stream:
                torch.save(synthetic_state(), stream)
            receipt = authenticate_state(directory, path.name, sha(path), 30)
            self.assertTrue(receipt["weights_only_cpu_load"])
            for bad in ("0" * 64, "malformed"):
                with self.assertRaises(ValueError):
                    authenticate_state(directory, path.name, bad, 30)

    def test_basename_path_boundary(self):
        with synthetic_directory() as directory:
            for name in ("../outside.pt", "..\\outside.pt", "C:outside.pt", ".", "/outside.pt"):
                with self.assertRaises(PermissionError):
                    bounded_path(directory, name)

    def test_incomplete_execution_precedes_private_access(self):
        with self.assertRaises(PermissionError):
            completed_records({"status": "running"}, {}, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticStateTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    result = audit_states(args.repo)
    destination = bounded_path(args.repo / "results/development" / PACKET_NAME, "checkpoint_state_audit.json")
    if args.check:
        if load_json(destination) != result:
            raise ValueError("Checkpoint state audit receipt changed")
    else:
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checkpoints_authenticated": result["checkpoints_authenticated"],
                      "maximum_head_l2": result["maximum_head_l2"],
                      "maximum_spatial_electrode_l2": result["maximum_spatial_electrode_l2"]}))


if __name__ == "__main__":
    main()
