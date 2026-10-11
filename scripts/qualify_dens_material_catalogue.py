"""Read DENS author-catalogue metadata and documentation; never fetch data bodies.

The only non-documentation file fetched is the content-addressed 64-byte
FileNameFormat.txt. Movie, EEG, behavior, and rating downloads are unsupported.
Use --verify to replay the saved catalogue offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "results/development/dens_metadata_qualification_2026-10-12/material_catalogue.json"
SOURCE_REVIEW = REPO / "docs/publication/DENS_Methods_Source_Review_2026-10-12.md"
EXPECTED_IDS = (10, 26, 40, 51, 54, 67, 92, 98, 109, 113, 152, 198, 199, 201, 210, 214)
FORMAT_SHA = "b7af6c73184d6a2afb9426caad16a9d368974db4a053db074a94d740ee25825f"
URLS = {
    "project": "https://api.osf.io/v2/nodes/m8dwz/",
    "film_component": "https://api.osf.io/v2/nodes/tgcj8/",
    "catalogue": "https://api.osf.io/v2/nodes/tgcj8/files/osfstorage/?page%5Bsize%5D=100",
    "filename_format": "https://osf.io/download/342tn/",
    "eeglab_events": "https://eeglab.org/tutorials/04_Import/Importing_Event_Epoch_Info.html",
    "eeglab_structures": "https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def permitted_url(url: str) -> bool:
    if url in set(URLS.values()):
        return True
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        return False
    if parsed.netloc == "files.osf.io":
        return parsed.path.rstrip("/") == "/v1/resources/tgcj8/providers/osfstorage/628609c92817743b79f7da72"
    return parsed.netloc == "storage.googleapis.com" and parsed.path == "/cos-osf-prod-files-us-east1/" + FORMAT_SHA


class ExactURLRedirects(urllib.request.HTTPRedirectHandler):
    """Reject redirects outside the finite documentation allowlist."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not permitted_url(newurl):
            raise PermissionError("Unexpected documentation redirect; no body fetched")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(key: str, *, retain_body: bool) -> dict:
    url = URLS[key]
    maximum = 64 if key == "filename_format" else 2 * 1024 * 1024
    opener = urllib.request.build_opener(ExactURLRedirects())
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GRU-XNet-public-metadata-qualification/1"})
            with opener.open(request, timeout=30) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > maximum:
                    raise ValueError("Documentation response exceeds declared access bound")
                body = response.read(maximum + 1)
                if len(body) > maximum:
                    raise ValueError("Documentation response exceeds access bound")
                if not permitted_url(response.geturl()):
                    raise PermissionError("Unexpected final URL")
                record = {
                    "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": response.status, "body_bytes": len(body),
                    "body_sha256": sha(body), "content_type": response.headers.get("Content-Type"),
                    "retained_body": retain_body,
                }
                if retain_body:
                    record["body_utf8"] = body.decode("utf-8")
                if key == "filename_format" and (len(body) != 64 or sha(body) != FORMAT_SHA):
                    raise ValueError("Known filename codebook identity changed")
                return record
        except (PermissionError, ValueError, UnicodeDecodeError):
            raise
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(1)
    raise RuntimeError("Documentation retrieval failed for " + key) from error


def normalize_catalogue(source_documents: dict) -> tuple[list[dict], dict]:
    catalogue = json.loads(source_documents["catalogue"]["body_utf8"])
    if catalogue.get("links", {}).get("next"):
        raise ValueError("Unexpected pagination; access remains bounded to one catalogue page")
    entries = catalogue["data"]
    if len(entries) != 70 or catalogue["links"]["meta"]["total"] != 70:
        raise ValueError("Author catalogue inventory changed; qualify separately")
    films, format_record = [], None
    for entry in entries:
        attr = entry["attributes"]
        name = attr["name"]
        if attr["kind"] != "file":
            raise ValueError("Unexpected directory entry")
        if name == "FileNameFormat.txt":
            format_record = {
                "name": name, "osf_file_id": entry["id"], "guid": attr["guid"],
                "byte_size": attr["size"], "current_version": attr["current_version"],
                "inventory_sha256": attr["extra"]["hashes"]["sha256"],
                "documentation_url": "https://osf.io/" + attr["guid"] + "/",
                "text": source_documents["filename_format"]["body_utf8"],
            }
            continue
        match = re.fullmatch(r"(\d+)-(.+)\.(mp4|webm)", name)
        if not match:
            raise ValueError("Unexpected non-movie catalogue entry")
        digest = attr["extra"]["hashes"]["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Missing inventory movie hash")
        films.append({
            "catalogue_stimulus_id": int(match.group(1)), "filename": name,
            "catalogue_name": match.group(2), "extension": match.group(3),
            "osf_file_id": entry["id"], "guid": attr["guid"],
            "byte_size": attr["size"], "current_version": attr["current_version"],
            "inventory_sha256": digest, "inventory_md5": attr["extra"]["hashes"].get("md5"),
            "created_utc": attr["date_created"], "modified_utc": attr["date_modified"],
            "duration_seconds": None, "duration_metadata_present": False,
            "movie_body_downloaded": False, "hash_verified_against_movie_bytes": False,
        })
    if len(films) != 69 or len({film["catalogue_stimulus_id"] for film in films}) != 69:
        raise ValueError("Unexpected film count or duplicate catalogue ID")
    if not format_record or format_record["inventory_sha256"] != FORMAT_SHA or format_record["byte_size"] != 64:
        raise ValueError("Filename codebook metadata changed")
    return sorted(films, key=lambda row: row["catalogue_stimulus_id"]), format_record


def build() -> dict:
    documents = {key: fetch(key, retain_body=key not in {"eeglab_events", "eeglab_structures"}) for key in URLS}
    project = json.loads(documents["project"]["body_utf8"])["data"]
    component = json.loads(documents["film_component"]["body_utf8"])["data"]
    if project["id"] != "m8dwz" or component["id"] != "tgcj8":
        raise ValueError("Author project identity changed")
    if component["relationships"]["parent"]["data"]["id"] != project["id"]:
        raise ValueError("Film component is not part of the paper-linked project")
    films, format_record = normalize_catalogue(documents)
    by_id = {row["catalogue_stimulus_id"]: row for row in films}
    if not all(identifier in by_id for identifier in EXPECTED_IDS):
        raise ValueError("An expected published emotional material is absent")
    return {
        "date": "2026-10-12", "scope": "Public author catalogue metadata and software documentation only",
        "script_sha256": sha(Path(__file__).read_bytes()),
        "manual_source_review_sha256": sha(SOURCE_REVIEW.read_bytes()),
        "original_release": {"dataset": "ds003751", "snapshot": "1.0.6", "commit": "5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c"},
        "author_catalogue": {
            "project_id": project["id"], "project_title": project["attributes"]["title"],
            "component_id": component["id"], "component_title": component["attributes"]["title"],
            "catalogue_entries": 70, "film_entries": 69, "filename_document_entries": 1,
            "catalogue_is_the_wider_validation_collection": True,
            "catalogue_is_not_69_authenticated_eeg_recorded_materials": True,
        },
        "source_documents": documents,
        "filename_document": format_record,
        "films": films,
        "published_protocol": {
            "verification_basis": "Manual review of final primary publication; no programmatic protocol extraction",
            "source_review_file": str(SOURCE_REVIEW.relative_to(REPO)).replace("\\", "/"),
            "doi": "10.1109/ACCESS.2023.3266804",
            "title": "Emotion Recognition Using Temporally Localized Emotional Events in EEG With Naturalistic Context: DENS# Dataset",
            "authors": ["Mohammad Asif", "Sudhakar Mishra", "Majithia Tejas Vinodbhai", "Uma Shanker Tiwary"],
            "journal": "IEEE Access", "volume": 11, "pages": "39913–39925",
            "publication_date": "2023-04-13", "current_version_date": "2023-04-26",
            "url": "https://doi.org/10.1109/ACCESS.2023.3266804",
            "accessible_primary_copy": "https://www.researchgate.net/publication/370007995_Emotion_Recognition_using_Temporally_Localized_Emotional_Events_in_EEG_with_Naturalistic_Context_DENS_Dataset",
            "emotional_candidate_count": 16, "emotional_candidate_catalogue_ids": list(EXPECTED_IDS),
            "main_emotional_trials": 9, "main_non_emotional_trials": 2, "main_total_trials": 11,
            "nominal_clip_duration_seconds": 60, "nominal_main_baseline_seconds": 80,
            "documented_whole_clip_valence_scale": [1, 9],
            "catalogue_matches_by_id": {str(i): by_id[i]["filename"] for i in EXPECTED_IDS},
            "neutral_material_exact_identity_authenticated": False,
            "actual_shown_movie_bytes_authenticated": False,
        },
        "eeglab_format_conventions": {
            "basis": "Official EEGLAB documentation; not authentication of DENS TSV export",
            "event_latency_internal_unit": "samples", "event_duration_internal_unit": "samples",
            "continuous_latency_seconds_formula": "(latency - 1) / srate",
            "continuous_duration_seconds_formula": "duration / srate",
            "first_sample_corresponds_to_time_zero": True,
            "boundary_events_may_use_fractional_samples_and_mark_removed_or_concatenated_data": True,
            "actual_dens_export_clock_units_authenticated": False,
        },
        "access": {
            "api_metadata_body_bytes_read": sum(documents[key]["body_bytes"] for key in ("project", "film_component", "catalogue")),
            "software_documentation_body_bytes_read": sum(documents[key]["body_bytes"] for key in ("eeglab_events", "eeglab_structures")),
            "filename_document_body_bytes_read": 64, "movie_body_bytes_read": 0,
            "waveform_body_bytes_read": 0, "behavior_body_bytes_read": 0, "rating_values_read": 0,
            "external_messages_sent": 0, "new_fits": 0, "new_predictions": 0,
            "confirmation_access": False, "research_question_changed": False,
        },
        "qualification_limit": "Metadata hashes are publisher inventory assertions; no movie, waveform, individual rating, trial endpoint, behavior join, source-title independence, or predictive benefit authenticated",
    }


def verify(packet: dict) -> dict:
    if packet["script_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("Catalogue-producing script changed")
    if packet["manual_source_review_sha256"] != sha(SOURCE_REVIEW.read_bytes()):
        raise ValueError("Manual primary-source review changed")
    documents = packet["source_documents"]
    for key, source in documents.items():
        if key not in URLS or source["url"] != URLS[key]:
            raise ValueError("Unexpected source URL")
        if source["retained_body"]:
            body = source["body_utf8"].encode("utf-8")
            if len(body) != source["body_bytes"] or sha(body) != source["body_sha256"]:
                raise ValueError("Retained metadata source identity changed")
    films, format_record = normalize_catalogue(documents)
    if films != packet["films"] or format_record != packet["filename_document"]:
        raise ValueError("Catalogue normalization does not replay")
    if packet["published_protocol"]["emotional_candidate_catalogue_ids"] != list(EXPECTED_IDS):
        raise ValueError("Manual published material set changed")
    if packet["access"]["filename_document_body_bytes_read"] != 64:
        raise ValueError("Filename-document bound changed")
    for key in ("movie_body_bytes_read", "waveform_body_bytes_read", "behavior_body_bytes_read", "rating_values_read", "external_messages_sent", "new_fits", "new_predictions", "confirmation_access", "research_question_changed"):
        if packet["access"][key] != 0:
            raise ValueError("Recorded metadata access boundary changed")
    return {"passed": True, "film_entries_replayed": len(films), "retained_metadata_bodies_replayed": 4, "movie_waveform_rating_values_read": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true", help="Replay the saved metadata offline without network access")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(json.loads(args.output.read_bytes())), indent=2))
        return
    if args.output.exists():
        raise FileExistsError("Preserve the existing receipt; use a new output path for another live metadata check")
    packet = build()
    packet["offline_replay"] = verify(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"written": str(args.output), "sha256": sha(args.output.read_bytes()), **packet["offline_replay"]}, indent=2))


if __name__ == "__main__":
    main()
