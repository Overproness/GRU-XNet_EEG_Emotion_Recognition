"""Independent scalar ANOVA audit of the public scope/precision packet.

Does not import the producing analyzer, private data loaders or training code.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scipy.stats import t

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/development"
PACKET = BASE / "faced_scope_precision_2026-10-11"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit():
    analysis = read(PACKET/"analysis.json")
    roles = read(BASE/"faced_backup_2026-10-10/participant_reservation.json")
    design = read(BASE/"faced_design_proposal_2026-10-11/recommended_proposal.json")
    quality = read(BASE/"faced_inference_quality_2026-10-11/comparison.json")
    catalogue = {x["clip"]:x["source_film_sha256"] for x in read(BASE/"faced_backup_2026-10-10/stimulus_metadata.json")["records"]}
    allowed_people = set(roles["development_source"]+roles["development_validation"])
    allowed_clips = {2,3,4,6,9,11,12,17,18,20,23,25,27,28}
    inputs = {}
    for name in ["faced_inference_quality_2026-10-11","faced_representation_v2_2026-10-11"]:
        with (BASE/name/"predictions.csv").open(newline="",encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["subject"] not in allowed_people or int(row["vid"]) not in allowed_clips:
                raise ValueError("Protected identity in audit input")
            if row["family"] != catalogue[int(row["vid"])]:
                raise ValueError("Family disagreement")
        inputs[name] = rows

    def get(rotation, model):
        collection = inputs["faced_inference_quality_2026-10-11"] if model in {"context_only","context_plus_spectral_eeg"} else inputs["faced_representation_v2_2026-10-11"]
        alpha = quality["rotations"][rotation]["selected"][model]["alpha"] if model in {"context_only","context_plus_spectral_eeg"} else None
        result = {}
        for row in collection:
            if int(row["rotation"]) != rotation or row["arm"] != "validation_unseen" or row["model"] != model:
                continue
            if alpha is not None and float(row["alpha"]) != alpha:
                continue
            key = (row["subject"],row["family"])
            if key in result:
                raise ValueError("Duplicate cell")
            result[key] = (float(row["target"]), float(row["prediction"]))
        return result

    differences = []
    for entry in analysis["contrasts"]:
        name, rotation = entry["contrast"],entry["rotation"]
        ctx_model = "context_kernel" if name == "covariance_joint_vs_kernel_context" else "context_only"
        joint_model = "context_plus_spectral_eeg" if name == "spectral_joint_vs_ridge_context" else "context_covariance_kernel"
        context, joint = get(rotation,ctx_model),get(rotation,joint_model)
        people = sorted(roles["development_validation"])
        films = sorted(x["family"] for x in design["rotations"][rotation]["families"]["development_validation"])
        expected = {(p,f) for p in people for f in films}
        if set(context) != expected or set(joint) != expected:
            raise ValueError("Missing audit cell")
        gains = {}
        for key in expected:
            y,pred = context[key]; other_y,other_pred = joint[key]
            if y != other_y:
                raise ValueError("Target mismatch")
            gains[key] = abs(y-pred)-abs(y-other_pred)
        ni,nj = len(people),len(films)
        total = math.fsum(gains.values())
        correction = total**2/(ni*nj)
        row_sums = [math.fsum(gains[p,f] for f in films) for p in people]
        col_sums = [math.fsum(gains[p,f] for p in people) for f in films]
        ss_total = math.fsum(v*v for v in gains.values())-correction
        ss_p = math.fsum(v*v for v in row_sums)/nj-correction
        ss_f = math.fsum(v*v for v in col_sums)/ni-correction
        ms_e = (ss_total-ss_p-ss_f)/((ni-1)*(nj-1))
        raw = {"participant":(ss_p/(ni-1)-ms_e)/nj,"film":(ss_f/(nj-1)-ms_e)/ni,"residual":ms_e}
        components = {key:max(0,value) for key,value in raw.items()}
        gain = total/(ni*nj)
        source_sd = math.sqrt(math.fsum((v/nj-gain)**2 for v in row_sums)/(ni-1))
        se = math.sqrt(components["participant"]/33+components["film"]/8+components["residual"]/(33*8))
        factor = 1.959963984540054+.8416212335729143
        diffs = [abs(gain-entry["mean_gain"]),abs(source_sd-entry["participant_mean_sample_sd"]),abs(se-entry["planning"]["crossed_se"])]
        diffs += [abs(raw[k]-entry["raw_components"][k]) for k in raw]
        diffs += [abs(factor*se-entry["planning"]["normal_80_percent_detectable_effect"]),abs(t.ppf(.975,32)*source_sd/math.sqrt(33)-entry["planning"]["fixed_roster_t32_95_half_width"])]
        for row in entry["planning"]["margin_sensitivity"]:
            remaining = (row["margin"]/factor)**2-components["film"]/8
            required = None if remaining <= 0 else max(2,math.ceil((components["participant"]+components["residual"]/8)/remaining))
            if required != row["nominal_required_participants_at_eight_films"]:
                raise ValueError("Planning sample-size arithmetic mismatch")
        if max(diffs)>1e-12:
            raise ValueError("Independent variance calculation mismatch")
        differences.append(max(diffs))

    marker_records = read(BASE/"faced_raw_2026-10-11/joins_numeric_projection.json")["records"]
    durations = [s["end_seconds"]-s["start_seconds"] for r in marker_records for s in r["spans"] if s["vid"] not in {13,14,15,16,22}]
    if len(durations) != 2829 or min(durations) != analysis["scope"]["minimum_included_marker_duration_seconds"]:
        raise ValueError("Independent marker geometry disagreement")
    raw_summary = read(BASE/"faced_raw_2026-10-11/summary.json")
    native_floor = 1/min(float(rate) for rate in raw_summary["sampling_frequency_participants"])
    if native_floor != analysis["scope"]["native_integer_endpoint_floor_maximum_lag_bound_seconds"]:
        raise ValueError("Native endpoint-floor bound disagreement")
    if min(durations)-32-native_floor != analysis["scope"]["minimum_32s_input_start_margin_after_endpoint_floor_lower_bound_seconds"]:
        raise ValueError("Conservative marker margin disagreement")
    bindings = read(PACKET/"input_bindings.json")
    for rel,b in bindings["files"].items():
        if hashlib.sha256((REPO/rel).read_bytes()).hexdigest() != b["sha256"]:
            raise ValueError("Frozen input disagreement")
    return {"passed":True,"independent_of_producer_import":True,"contrasts_audited":len(differences),
            "maximum_scalar_algebra_difference":max(differences),"original_marker_spans":3444,"included_marker_spans":len(durations),"native_endpoint_floor_bound_seconds":native_floor,
            "prediction_rows_identity_checked":{name:len(rows) for name,rows in inputs.items()},"input_sha256_bindings_verified":len(bindings["files"]),
            "method":"Scalar corrected sums of squares instead of producer's centered matrix ANOVA",
            "new_fits":0,"protected_outcomes_opened":False,"original_rating_or_waveform_files_opened":False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true");args=parser.parse_args()
    result=audit();path=PACKET/"independent_audit.json"
    if args.check:
        if read(path)!=result:
            raise ValueError("Independent audit changed")
    else:
        path.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__":
    main()
