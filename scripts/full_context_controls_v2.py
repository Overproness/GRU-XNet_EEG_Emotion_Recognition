"""Prepare, predeclare, run and replay full-width EEG/context controls."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gruxnet.full_context_controls_v2 import plan, prepare, run, verify, analyze
from gruxnet.data import write_json, digest

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'run', 'verify', 'analyze', 'batch'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('SEEDIV', 'DEAP'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); root = args.root.resolve(); plan_path = root/'full_context_v2_plan_2026-10-06.json'
    if args.command == 'plan':
        if plan_path.exists(): raise FileExistsError('Do not overwrite frozen declaration')
        write_json(plan_path, plan()); print(plan_path)
    elif args.command == 'prepare':
        print(json.dumps(prepare(args.dataset, root, root/f'cache_full_context_{args.dataset.lower()}'), indent=2))
    else:
        datasets = ('SEEDIV', 'DEAP') if args.command == 'batch' else (args.dataset,)
        for dataset in datasets:
            cache = root/f'cache_full_context_{dataset.lower()}'; output = root/f'full_context_v2_{dataset.lower()}'
            if args.command in ('run', 'batch'): run(dataset, cache, output, plan_path, args.device)
            if args.command in ('verify', 'batch'): print(json.dumps(verify(dataset, cache, output, args.device), indent=2))
            if args.command in ('analyze', 'batch'):
                result = analyze(dataset, output)
                replay = analyze(dataset, output, write=False)
                if result != replay: raise ValueError('Analysis replay changed')
                write_json(output/'analysis_verification.json', {'passed': True, 'comparison_digest': digest(result), 'independent_recomputation': True})
                print(json.dumps(result['models'], indent=2))
