from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet import grouped_material_controls as control
from gruxnet.data import write_json

if __name__ == "__main__":
    parser = ArgumentParser(description="Repeated participant/material groupings and compatible DEAP control")
    parser.add_argument("action", choices=["prepare-deap", "plan", "feasibility", "freeze", "run", "verify", "analyze", "batch"])
    parser.add_argument("--dataset", choices=["SEEDIV", "DEAP"])
    for name in ("cache", "common-cache", "reve-run", "previous", "output", "plan", "workspace"): parser.add_argument("--"+name, type=Path)
    parser.add_argument("--device", default="cuda"); a = parser.parse_args()
    if a.action == "prepare-deap": result = control.prepare_deap(a.common_cache, a.cache)
    elif a.action == "plan":
        if a.plan.exists(): raise FileExistsError("Existing declaration")
        write_json(a.plan, control.plan()); result = {"saved": str(a.plan)}
    elif a.action == "feasibility":
        _, table, _, _ = control.features(a.dataset, a.cache, a.reve_run)
        result = control.feasibility(table, a.dataset); write_json(a.output, result); result = {"feasible_pairs": len(result), "saved": str(a.output)}
    elif a.action == "freeze":
        runs = a.workspace/"publication_runs"; result = {}
        for dataset, cache_name, output_name in (("SEEDIV", "cache_temporal_native_seediv", "repeated_material_seediv"), ("DEAP", "cache_temporal_deap", "repeated_material_deap")):
            output = runs/output_name
            if output.exists(): raise FileExistsError("Existing freeze/output directory")
            _, original, info, _ = control.features(dataset, runs/cache_name, runs/"reve_frozen_seediv")
            output.mkdir(parents=True)
            write_json(output/"config.json", control.configuration(dataset, info, runs/"reve_frozen_seediv", a.plan, a.device, runs/"within_session_material_seediv"))
            write_json(output/"plan.json", json.loads(a.plan.read_text()))
            feasible = control.feasibility(original, dataset); write_json(output/"feasibility.json", feasible)
            for group in control.GROUPS:
                table = control.annotate(original, dataset, group); fs, _ = control.cells(table, dataset, group)
                write_json(output/f"folds_group{group}.json", fs)
                table[["material_key", "material_rank"]].drop_duplicates().sort_values("material_key").to_csv(output/f"material_assignments_group{group}.csv", index=False)
            result[dataset] = {"feasible_pairs": len(feasible), "saved": str(output)}
    elif a.action == "batch":
        runs = a.workspace/"publication_runs"; result = {}
        for dataset, cache_name, output_name in (("SEEDIV", "cache_temporal_native_seediv", "repeated_material_seediv"), ("DEAP", "cache_temporal_deap", "repeated_material_deap")):
            arguments = (dataset, runs/cache_name, runs/"reve_frozen_seediv", runs/"within_session_material_seediv", runs/output_name)
            result[dataset] = control.run(*arguments, a.plan, a.device)
            result[dataset]["verification"] = control.verify(*arguments, a.device)
            report = control.analyze(dataset, runs/output_name, runs/"within_session_material_seediv")
            replay = control.analyze(dataset, runs/output_name, runs/"within_session_material_seediv", write=False)
            if report != replay: raise ValueError("Analysis not deterministic")
            record = {"passed": True, "comparison_recomputed_exactly": True, "contrasts": len(report["contrasts"])}
            write_json(runs/output_name/"analysis_verification.json", record)
            print(json.dumps({"dataset": dataset, "completed_and_verified": True}), flush=True)
    elif a.action == "run": result = control.run(a.dataset, a.cache, a.reve_run, a.previous, a.output, a.plan, a.device)
    elif a.action == "verify": result = control.verify(a.dataset, a.cache, a.reve_run, a.previous, a.output, a.device)
    else:
        report = control.analyze(a.dataset, a.output, a.previous); result = {"contrasts": len(report["contrasts"])}
    print(json.dumps(result, indent=2))
