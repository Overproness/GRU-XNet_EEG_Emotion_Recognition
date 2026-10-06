"""Regenerate all DEAP prefix features from source recordings without cache reuse."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gruxnet.data import COMMON_CHANNELS, sha256, source_trials, write_json
from gruxnet.prepare import preprocess_signal
from gruxnet.temporal_controls import sequence_features
from gruxnet.grouped_material_controls import load_deap

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("data", "common-cache", "cache", "output"): parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args(); x, table, info = load_deap(a.cache)
    lineage = {r["trial_id"]: r for r in json.loads((a.common_cache/"lineage.json").read_text()) if r["dataset"] == "DEAP"}
    sources = {r["source"]: r["source_sha256"] for r in lineage.values()}; positions = dict(zip(table.trial_id, table.index))
    iterator = source_trials(a.data, []); retained = 0; excluded = 0; seen = set()
    for number in range(1280):
        metadata, signal, channels, fs = next(iterator)
        if metadata["dataset"] != "DEAP" or metadata["source_sha256"] != sources[metadata["source"]]: raise ValueError("Changed source population/hash")
        if metadata["label"] is None:
            if metadata["original_label"] != 5 or metadata["trial_id"] in positions: raise ValueError("Wrong excluded rating")
            excluded += 1; continue
        i = positions[metadata["trial_id"]]; seen.add(metadata["trial_id"]); retained += 1
        row = table.iloc[i]
        if metadata["label"] != row.label or abs(metadata["original_label"]-row.original_label) > 1e-12: raise ValueError("Changed corrected label")
        filtered, mask = preprocess_signal(signal, fs, channels, COMMON_CHANNELS)
        if not mask.all() or filtered.shape != (14, 7680): raise ValueError("Incorrect source duration/montage")
        windows = filtered.reshape(14, 15, 512).transpose(1, 0, 2)
        cached = np.load(a.common_cache/lineage[metadata["trial_id"]]["cache_file"], allow_pickle=False)
        np.testing.assert_array_equal(windows, cached)
        np.testing.assert_array_equal(sequence_features(windows[:10]), x[i])
        if (number+1) % 160 == 0: print(f"Regenerated DEAP source trials {number+1}/1280", flush=True)
    if retained != 1264 or excluded != 16 or seen != set(table.trial_id): raise ValueError("Incomplete raw replay")
    record = {"passed": True, "source_trials": 1280, "retained_trials": retained, "excluded_midpoints": excluded,
              "full_waveforms_exactly_reproduced": retained, "prefix_sequences_exactly_reproduced": retained,
              "maximum_array_difference": 0, "cache_fingerprint": info["fingerprint"], "sequences_sha256": sha256(a.cache/"sequences.npy"),
              "source_hashes": {name: sha256(Path(__file__).resolve().parents[1]/name) for name in ("gruxnet/data.py", "gruxnet/prepare.py", "gruxnet/temporal_controls.py", "gruxnet/grouped_material_controls.py", "scripts/verify_deap_temporal_input.py")},
              "scope": "Reread original mirror files and corrected spreadsheet loader; regenerate all filtered waveforms and first40s Welch tokens exactly. No data/label changes, first-party signal authentication or model fitting."}
    write_json(a.output, record); print(json.dumps(record, indent=2))
