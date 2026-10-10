"""Record bounded public-source retrievals; full third-party text stays private."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/development/data_target_review_2026-10-10"
PRIVATE = REPO.parent / "publication_runs/data_target_review_2026-10-10/source_metadata"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def eeg_paths(paths):
    return [s for s in paths if s.startswith("sub-") and "/eeg/" in s and "_task-emotion" in s and s.endswith("_eeg.edf")]


def compute():
    records=[]; logs={}
    for name in ("fetch_log.json", *[f"fetch_log_{i}.json" for i in range(2,6)]):
        path=PRIVATE/name; logs[name]=sha(path)
        for r in json.loads(path.read_text(encoding="utf-8")):
            body=PRIVATE/(r["name"]+".response")
            if "body_sha256" in r and (sha(body)!=r["body_sha256"] or body.stat().st_size!=r["bytes"]):
                raise ValueError("Changed retrieved response")
            records.append(r)
    tree=json.loads((PRIVATE/"emo_mc_tree.response").read_text(encoding="utf-8"))
    if tree.get("truncated"):
        raise ValueError("Cannot audit an incomplete repository tree")
    paths=[r["path"] for r in tree["tree"]]
    waveforms=eeg_paths(paths)
    eeg_subjects=sorted({s.split("/")[0] for s in waveforms})
    beh_subjects={s.split("/")[0] for s in paths if s.startswith("sub-") and s.endswith("_beh.tsv")}
    ec=json.loads((PRIVATE/"emo_mc_commit.response").read_text(encoding="utf-8"))["sha"]
    fc=json.loads((PRIVATE/"faced_commit.response").read_text(encoding="utf-8"))["sha"]
    for branch,fixed in (("emo_mc_readme_pinned_candidate","emo_mc_readme_fixed"),("faced_readme","faced_readme_fixed"),("faced_schema","faced_schema_fixed")):
        if sha(PRIVATE/(branch+".response"))!=sha(PRIVATE/(fixed+".response")):
            raise ValueError("Documentation changed while pinning its revision")
    schema=json.loads((PRIVATE/"faced_schema_fixed.response").read_text(encoding="utf-8"))
    eeg_schema=json.loads((PRIVATE/"emo_mc_eeg_fixed.response").read_text(encoding="utf-8"))
    return {"checked_local_date":"2026-10-10","timezone":"Asia/Karachi","models_fitted":0,
            "new_participant_rating_rows_downloaded":0,"new_eeg_waveforms_downloaded":0,
            "retrievals":records,"private_log_sha256":logs,
            "candidate_revisions":{"EmoEEG-MC":ec,"FACED_NEMAR_conversion":fc},
            "emo_mc_inventory":{"eeg_edf_files":len(waveforms),"participants_with_raw_eeg_edf":len(eeg_subjects),
                                "participants_with_behavior_paths":len(beh_subjects),"behavior_ids_without_raw_eeg_paths":sorted(beh_subjects-set(eeg_subjects)),
                                "channels_tsv_files":sum(s.endswith("_channels.tsv") for s in paths),
                                "events_tsv_files":sum(s.endswith("_events.tsv") for s in paths),
                                "scope":"Repository paths, not successful annex retrieval, decoded waveforms, event validity, rating completeness or biological participant identity."},
            "faced_event_schema_columns":sorted(schema),
            "emo_mc_one_public_eeg_sidecar":{"SamplingFrequency":eeg_schema["SamplingFrequency"],"EEGChannelCount":eeg_schema["EEGChannelCount"],"SoftwareFilters":eeg_schema["SoftwareFilters"]},
            "interpretation":"Metadata access alone does not authenticate local recordings or certify physical calibration. Failed/empty/error-bodied Mendeley routes are not first-party content comparisons. No full third-party text, original rating tables, waveforms, signed URLs or credentials are exported."}


def main():
    parser=ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("build","verify-local"));args=parser.parse_args()
    result=compute();path=OUT/"source_checks.json";proofpath=OUT/"source_verification.json"
    if args.mode=="build":
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        proof={"passed":True,"script_sha256":sha(Path(__file__)),"source_checks_sha256":sha(path),
               "private_response_checks":sum("body_sha256" in r for r in result["retrievals"]),
               "scope":"Retrieval-log and privately retained response byte integrity; narrative scientific interpretation is reviewed separately."}
        proofpath.write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8")
    else:
        proof=json.loads(proofpath.read_text(encoding="utf-8"))
        if proof["script_sha256"]!=sha(Path(__file__)) or proof["source_checks_sha256"]!=sha(path) or result!=json.loads(path.read_text(encoding="utf-8")):
            raise ValueError("Source evidence failed exact reproduction")
    print(json.dumps({"passed":True,"mode":args.mode,"retrievals":len(result["retrievals"]),"raw_eeg_subject_paths":result["emo_mc_inventory"]["participants_with_raw_eeg_edf"]}))


if __name__=="__main__":main()
