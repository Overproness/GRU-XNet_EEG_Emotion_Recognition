"""Independently replay public DENS filename/hash metadata; no outcome access."""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = REPO / "results/development/dens_metadata_qualification_2026-10-12"
NEW = REPO / "results/development/dens_author_project_qualification_2026-10-12"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    packet = json.loads((NEW / "metadata_crosswalk.json").read_text())
    annotations = json.loads((OLD / "metadata_annotations.json").read_text())
    catalogue = json.loads((OLD / "material_catalogue.json").read_text())
    bindings = packet["existing_packet_bindings"]
    assert sha(OLD / "metadata_annotations.json") == bindings["metadata_annotations_json_sha256"]
    assert sha(OLD / "material_catalogue.json") == bindings["material_catalogue_json_sha256"]
    old = json.loads((OLD / "completion/verification.json").read_text())
    for rel, expected in old["bindings"].items():
        if sha(REPO / rel) != expected:
            raise ValueError("Existing sealed audit binding changed: " + rel)
    movies = packet["movie_metadata"]
    filenames = {m["filename"] for m in movies}
    keys = set(annotations["summary"]["unique_material_keys"])
    assert len(movies) == len(filenames) == 38 and len(keys) == 20 and keys.issubset(filenames)
    hashes = {m["reported_sha256"] for m in movies}
    afdi = {m["inventory_sha256"] for m in catalogue["films"]}
    assert len(catalogue["films"]) == 69 and len(hashes.intersection(afdi)) == 0
    assert packet["filename_coverage"]["exact_filename_overlap_count"] == 20
    assert packet["afdi_hash_comparison"]["exact_match_count"] == 0
    assert packet["contact_form"]["submitted"] is False
    assert packet["contact_form"]["user_information_entered"] is False
    assert "Mail-Id" in packet["contact_form"]["visibly_required_fields"]
    for movie in movies:
        assert movie["movie_body_retrieved"] is False and movie["hash_verified_against_movie_body"] is False
    limits = packet["limits"]
    for key in ["author_execution_or_export_code_found", "numbered_key_to_afdi_title_crosswalk_qualified",
                "practice_main_phase_qualified", "event_endpoint_semantics_qualified", "movie_body_requests",
                "dataset_archive_body_requests", "eeg_waveform_values_decoded", "individual_rating_values_decoded",
                "event_semantic_values_decoded", "new_fits", "confirmation_outcomes_opened", "external_message_sent"]:
        assert limits[key] in (0, False), key
    print(json.dumps({"passed": True, "old_sealed_bindings_preserved": len(old["bindings"]),
                      "author_movie_metadata_records": 38, "qualified_filenames_covered": 20,
                      "afdi_movie_metadata_records": 69, "exact_reported_full_file_hash_matches": 0,
                      "usable_trial_joins_qualified": False, "inquiry_sent": False,
                      "new_fits": 0, "rating_or_waveform_values_accessed": 0,
                      "metadata_packet_sha256": sha(NEW / "metadata_crosswalk.json")}))


if __name__ == "__main__":
    main()
