"""Cross-check public identity/target metadata against retained source annotations."""
from argparse import ArgumentParser
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
DATA=WORKSPACE/"emotion-recognition-eeg-datasets/Emotion Recognition EEG Datasets"
OUT=REPO/"results/development/data_target_review_2026-10-10/local_label_verification.json"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(encoding="utf-8",newline="") as handle:return list(csv.DictReader(handle))


def verify():
    try:
        import xlrd  # noqa: F401
    except ImportError:
        sys.path.insert(0,str(REPO/".publication_deps"))
    bindings={};counts={}
    def bind(path):
        path=path.resolve()
        if not path.is_relative_to(WORKSPACE):raise ValueError("Annotation path escaped workspace")
        bindings[path.relative_to(WORKSPACE).as_posix()]=sha(path)
        return path
    deap=bind(DATA/"deap/metadata_xls/participant_ratings.xls")
    other=bind(DATA/"deap/Metadata/participant_ratings.xls")
    table=pd.read_excel(deap)
    pd.testing.assert_frame_equal(table,pd.read_excel(other))
    reference={f"DEAP:S{int(r.Participant_id):02d}:T{int(r.Experiment_id):02d}":float(r.Valence) for r in table.itertuples()}
    if len(reference)!=1280:raise ValueError("DEAP annotation join is incomplete")
    path=bind(REPO/"results/development/repeated_material_deap/predictions_transformer_unexposed_group1.csv")
    unique={r["trial_id"]:r for r in csv_rows(path)}
    expected={k:v for k,v in reference.items() if v!=5}
    if set(unique)!=set(expected):raise ValueError("DEAP midpoint/join mismatch")
    max_rating_error=max(abs(float(r["original_label"])-expected[k]) for k,r in unique.items())
    for k,r in unique.items():
        if abs(float(r["original_label"])-expected[k])>1e-12 or int(r["label"])!=int(expected[k]>5):
            raise ValueError("DEAP rating mismatch")
    counts["DEAP"]={"source_annotation_trials":1280,"eligible_metadata_matches":len(unique),"midpoint_exclusions":16,
                    "max_rating_difference":max_rating_error,"rating_comparison_absolute_tolerance":1e-12,"binary_label_differences":0}
    source=bind(DATA/"sead-4/ReadMe.txt")
    text=source.read_text(encoding="utf-8")
    sessions={int(s):ast.literal_eval(v) for s,v in re.findall(r"session([123])_label\s*=\s*(\[[^\]]+\])",text)}
    if set(sessions)!={1,2,3} or any(len(v)!=24 for v in sessions.values()):raise ValueError("SEED label metadata changed")
    path=bind(REPO/"results/development/temporal_native_seediv/predictions_transformer_absolute_native4.csv")
    native={r["trial_id"]:r for r in csv_rows(path)}
    expected={f"SEEDIV:S{s:02d}:R{session}:T{t:02d}":label for s in range(1,16) for session,labels in sessions.items() for t,label in enumerate(labels,1)}
    if set(native)!=set(expected) or any(int(native[k]["original_label"])!=label for k,label in expected.items()):raise ValueError("SEED native label mismatch")
    counts["SEEDIV"]={"source_annotation_trials":1080,"eligible_metadata_matches":1080,"native_class_counts":{str(k):v for k,v in sorted(Counter(expected.values()).items())}}
    source=bind(WORKSPACE/"publication_runs/audit_with_sources/gameemo_sam_ratings.json")
    sam=json.loads(source.read_text(encoding="utf-8"))
    if len(sam)!=112 or len({r["trial_id"] for r in sam})!=112:raise ValueError("GAMEEMO annotation count changed")
    for r in sam:
        if sha(bind(Path(r["pdf"])))!=r["pdf_sha256"]:raise ValueError("SAM source PDF changed")
    eligible={r["trial_id"]:int(r["valence"]>5) for r in sam if r["valence"]!=5}
    path=bind(REPO/"results/development/negative_transfer_neural_gameemo/trial_predictions.csv")
    game={r["trial_id"]:int(r["label"]) for r in csv_rows(path)}
    if game!=eligible:raise ValueError("GAMEEMO SAM/metadata mismatch")
    counts["GAMEEMO"]={"source_annotation_trials":112,"eligible_metadata_matches":93,"midpoint_exclusions":19,"sam_pdf_hashes_checked":112}
    result={"passed":True,"models_fitted":0,"new_inferences":0,"new_labels_extracted":0,"script_sha256":sha(Path(__file__)),
            "datasets":counts,"source_sha256":bindings,"scope":"Local annotation-to-public-metadata joins, duplicate DEAP spreadsheet agreement and retained SAM PDF byte checks. No pickle signal loading, new PDF mark extraction, first-party waveform authentication, physical-unit certification or new cohort outcome access."}
    return result


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__);parser.add_argument("mode",choices=("build","verify-local"));args=parser.parse_args()
    result=verify()
    if args.mode=="build":OUT.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    elif result!=json.loads(OUT.read_text(encoding="utf-8")):raise ValueError("Local target audit failed reproduction")
    print(json.dumps({"passed":True,"mode":args.mode,"annotation_trials":2472,"eligible_metadata_matches":2437,"sam_pdf_hashes_checked":112}))
