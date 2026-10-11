"""Qualify pinned DENS inventory and annotation headers without reading data rows.

Only full public metadata/codebook JSON, README/CHANGES, Git-annex pointer text,
and the first line of TSV files are returned to the application. HTTP/TLS may
buffer additional response bytes; this script cannot measure those bytes and
does not claim that no annotation bytes were transported. No EEG objects are
requested, no annotation data rows are decoded or retained, and no fitting runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


REPO = Path(__file__).resolve().parents[1]
PACKET = REPO / "results/development/dens_metadata_qualification_2026-10-12"
DESTINATION = PACKET / "inventory.json"
COMMIT = "5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c"
EXPECTED_TREE = "1552b175dffd076bb75c50c994cd154f021bfe0a"
RAW = f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds003751/{COMMIT}/"
API = "https://api.github.com/repos/OpenNeuroDatasets/ds003751/"
USER_AGENT = "GRU-XNet-DENS-public-metadata-qualification"
ROOT_METADATA = {"README", "CHANGES", "dataset_description.json", "participants.json"}
ANNEX = re.compile(r"MD5E-s(?P<size>[0-9]+)--(?P<md5>[0-9a-f]{32})(?:\.(?P<suffix>set|fdt))?$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def get(url: str, *, header_only: bool = False, bound: int = 1_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    started = datetime.now(timezone.utc).isoformat()
    with urllib.request.urlopen(request, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        declared = int(declared) if declared is not None else None
        common = {
            "url": url,
            "checked_utc": started,
            "final_url": response.geturl(),
            "status": response.status,
            "content_length": declared,
            "content_range": response.headers.get("Content-Range"),
            "content_encoding": response.headers.get("Content-Encoding"),
        }
        if header_only:
            # A logical readline returns only the header. Its internal HTTP/TLS
            # buffers can fetch other bytes; neither those bytes nor rows are
            # decoded, saved, hashed, printed, or used in any computation.
            body = response.readline(bound + 1)
            if len(body) > bound or not body.endswith(b"\n"):
                raise ValueError("TSV header missing or exceeds bound: " + url)
            common.update({
                "read_scope": "first_line_only",
                "application_body_bytes_returned": len(body),
                "application_data_rows_returned": 0,
                "response_body_bytes_transported": None,
                "transport_measurement": "Not observable here; HTTP/TLS buffered transport may include data rows up to the declared response length. Zero outcome transport is not claimed.",
                "transport_body_upper_bound_if_content_length_valid": declared,
                "response_closed_immediately_after_header": True,
            })
        else:
            body = response.read(bound + 1)
            if len(body) > bound:
                raise ValueError("Allowed metadata exceeds bound: " + url)
            common.update({"read_scope": "complete_allowed_metadata", "application_body_bytes_returned": len(body), "complete_body_sha256": sha256(body)})
        return body, common


def fetch_entry(entry: dict):
    path = entry["path"]
    header_only = path.endswith(".tsv")
    pointer = entry["mode"] == "120000"
    if pointer:
        if not path.endswith((".set", ".fdt")) and path != "LICENSE":
            raise ValueError("Unexpected symlink requiring a new access declaration: " + path)
        bound = 4096
    elif header_only:
        bound = 16384
    elif path in ROOT_METADATA or path.endswith("_eeg.json"):
        bound = 16384
    else:
        raise PermissionError("Entry is outside metadata access allowlist: " + path)
    body, access = get(RAW + path, header_only=header_only, bound=bound)
    result = {"path": path, "git_blob_sha1": entry["sha"], "git_blob_bytes": entry["size"], "git_mode": entry["mode"], "access": access}
    if header_only:
        text = body.decode("utf-8-sig").rstrip("\r\n")
        if "\n" in text or "\r" in text:
            raise ValueError("Unexpected multiline TSV header")
        result.update({"header": text, "columns": text.split("\t"), "header_bytes_sha256": sha256(body), "complete_blob_authenticated": False, "identity_limit": "Pinned URL and tree blob inventory; header-only reads cannot verify the complete annotation blob hash."})
    else:
        if len(body) != entry["size"] or blob_sha(body) != entry["sha"]:
            raise ValueError("Pinned metadata Git-blob authentication failed: " + path)
        result.update({"complete_blob_authenticated": True, "sha256": sha256(body)})
        if pointer:
            target = body.decode("utf-8")
            key = target.split("/")[-1]
            match = ANNEX.fullmatch(key)
            if not match or target.split("/")[-2] != key:
                raise ValueError("Unexpected annex pointer: " + path)
            result.update({"pointer_target": target, "annex_key": key, "annex_expected_bytes": int(match["size"]), "annex_md5": match["md5"], "object_body_requested": False})
        elif path.endswith(".json"):
            result["metadata"] = json.loads(body)
        else:
            result["metadata_text"] = body.decode("utf-8")
    return result


def aggregate(receipt: dict):
    entries = receipt["tree_entries"]
    paths = {entry["path"] for entry in entries}
    subjects = sorted({path.split("/")[0] for path in paths if path.startswith("sub-")})
    eeg = {path.split("/")[0] for path in paths if path.endswith("_eeg.set")}
    behavior = {path.split("/")[0] for path in paths if path.endswith("_beh.tsv")}
    events = {path.split("/")[0] for path in paths if path.endswith("_events.tsv")}
    headers = receipt["annotation_headers"]
    families = {"behavior": "_beh.tsv", "events": "_events.tsv", "channels": "_channels.tsv", "participants": "participants.tsv"}
    header_groups = {}
    for kind, suffix in families.items():
        grouped = {}
        for item in headers:
            if item["path"].endswith(suffix):
                grouped.setdefault(item["header"], []).append(item["path"])
        header_groups[kind] = [{"columns": key.split("\t"), "file_count": len(value), "paths": sorted(value)} for key, value in sorted(grouped.items())]
    return {
        "subject_keys": len(subjects), "subjects": subjects,
        "subjects_with_eeg_set": len(eeg), "subjects_with_behavior_tsv": len(behavior),
        "subjects_with_events_tsv": len(events),
        "potential_eeg_behavior_joins": len(eeg & behavior),
        "potential_eeg_behavior_event_joins": len(eeg & behavior & events),
        "potential_join_subjects": sorted(eeg & behavior & events),
        "subjects_missing_behavior": sorted(eeg - behavior),
        "subjects_missing_events": sorted(eeg - events),
        "usable_trial_count": None, "trial_join_authenticated": False,
        "annotation_header_variants": header_groups,
        "annotation_codebook_json_paths": sorted(path for path in paths if path.endswith(("_beh.json", "_events.json"))),
        "stimuli_paths": sorted(path for path in paths if path.startswith("stimuli/")),
        "code_paths": sorted(path for path in paths if path.startswith("code/")),
        "tsv_first_lines_read": len(headers),
        "logical_annotation_header_bytes": sum(item["access"]["application_body_bytes_returned"] for item in headers),
        "annotation_data_rows_returned_to_application": 0,
        "annotation_response_transport_bytes": None,
        "annotation_response_transport_byte_upper_bound": sum(item["access"]["content_length"] or 0 for item in headers),
        "zero_rating_or_event_body_transport_claimed": False,
        "waveform_object_body_requests": 0,
        "set_metadata_object_body_requests": 0,
    }


def build():
    commit_body, commit_access = get(API + "git/commits/" + COMMIT)
    commit = json.loads(commit_body)
    if commit["sha"] != COMMIT or commit["tree"]["sha"] != EXPECTED_TREE:
        raise ValueError("Pinned release commit/tree changed")
    tree_body, tree_access = get(API + "git/trees/" + EXPECTED_TREE + "?recursive=1")
    tree = json.loads(tree_body)
    if tree["sha"] != EXPECTED_TREE or tree["truncated"]:
        raise ValueError("Incomplete or wrong tree")
    entries = sorted(({key: entry[key] for key in ("path", "mode", "type", "sha", "size") if key in entry} for entry in tree["tree"]), key=lambda x: x["path"])
    tasks = [entry for entry in entries if entry["type"] == "blob" and (entry["path"] in ROOT_METADATA or entry["path"].endswith("_eeg.json") or entry["path"].endswith(".tsv") or entry["mode"] == "120000")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(fetch_entry, tasks))
    result = {
        "schema": "dens_inventory_header_qualification_v1", "date": "2026-10-12",
        "registry_url": "https://openneuro.org/datasets/ds003751/versions/1.0.6",
        "snapshot": "1.0.6", "commit": COMMIT, "tree_sha1": EXPECTED_TREE,
        "commit_metadata": {"author": commit["author"], "committer": commit["committer"], "message": commit["message"]},
        "git_api_access": [commit_access, tree_access],
        "tree_entries": entries,
        "metadata": [item for item in fetched if "metadata" in item or "metadata_text" in item],
        "annotation_headers": [item for item in fetched if "header" in item],
        "annex_pointers": [item for item in fetched if "annex_key" in item],
        "scope": "Public inventory, full metadata/codebooks/pointers, and annotation header lines only; no EEG objects or annotation data rows returned to the application. Buffered HTTP/TLS annotation transport beyond the first line cannot be measured here.",
        "unresolved": [
            "Header names alone do not prove rating semantics, observed scale, trial completeness or joins.",
            "No behavioral rating or event data rows are returned to the application; usable trial count remains unverified.",
            "Snapshot annex pointers authenticate expected EEG object length/MD5; no object bodies are downloaded by this script.",
            "Current S3 object URLs are not snapshot-specific; compare downloaded allowed metadata object bytes to pinned annex digest before use.",
            "Full annotation blob identity is not authenticated by header-only reads.",
            "README ODbL/DBCL and dataset_description CC0 statements require reconciliation.",
        ],
        "new_fits": 0, "new_predictions": 0,
        "original_rating_values_decoded": 0, "original_event_data_rows_decoded": 0,
        "confirmation_outcomes_opened": 0, "research_question_changed": False,
        "script_sha256": sha256(Path(__file__).read_bytes()),
    }
    result["summary"] = aggregate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Offline integrity/aggregate replay; performs no HTTP access")
    args = parser.parse_args()
    if args.check:
        result = json.loads(DESTINATION.read_bytes())
        if result["script_sha256"] != sha256(Path(__file__).read_bytes()):
            raise ValueError("Qualification script changed")
        if result["commit"] != COMMIT or result["tree_sha1"] != EXPECTED_TREE:
            raise ValueError("Inventory release changed")
        if aggregate(result) != result["summary"]:
            raise ValueError("Inventory aggregates changed")
        for key in ("original_rating_values_decoded", "original_event_data_rows_decoded", "new_fits", "new_predictions", "confirmation_outcomes_opened", "research_question_changed"):
            if result[key] not in (0, False):
                raise PermissionError("Access/experiment boundary changed")
    else:
        if DESTINATION.exists():
            raise FileExistsError("Preserve the existing qualification receipt")
        result = build()
        PACKET.mkdir(parents=True, exist_ok=True)
        with DESTINATION.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "offline_check": args.check, "summary": result["summary"], "new_fits": 0, "rating_values_decoded": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
