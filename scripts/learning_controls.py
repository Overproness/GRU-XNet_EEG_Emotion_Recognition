"""Freeze or fit the source-only learning study; no outer-test evaluation."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import torch
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gruxnet.learning_controls import plan, run
from gruxnet.data import write_json, sha256


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--root', type=Path, default=REPO.parent/'publication_runs')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); root = args.root.resolve()
    output = root/'learning_controls_2026-10-06'
    if args.command == 'plan':
        if output.exists(): raise FileExistsError('Do not overwrite a frozen source diagnostic')
        declaration = plan(root)
        if not json.loads((root/'eegnet_author_audit_2026-10-06/port_verification.json').read_text())['passed']:
            raise ValueError('Author-code numerical verification required')
        output.mkdir(); write_json(output/'plan.json', declaration)
        write_json(output/'config.json', {'plan_sha256': sha256(output/'plan.json'), 'torch': torch.__version__,
                                        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                                        'gpu': torch.cuda.get_device_name(), 'device': args.device})
        print(json.dumps(declaration, indent=2))
    else:
        run(root, output, args.device)
