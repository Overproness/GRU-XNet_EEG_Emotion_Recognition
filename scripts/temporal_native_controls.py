"""Run the separately predeclared transformer/native-label development controls."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from gruxnet import temporal_controls as control
from gruxnet.data import write_json

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["plan","prepare","run","verify","analyze"])
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--native-cache",type=Path)
    parser.add_argument("--common-cache",type=Path)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    if args.action=="plan":
        if args.plan.exists():
            raise FileExistsError("Do not overwrite a predeclared plan")
        write_json(args.plan,control.plan())
        result = {"saved":str(args.plan)}
    elif args.action=="prepare":
        result = control.prepare(args.data_root,args.native_cache,args.common_cache,args.cache)
    elif args.action=="run":
        result = control.run(args.cache,args.output,args.plan,args.device)
    elif args.action=="verify":
        result = control.verify(args.cache,args.output,args.device)
    else:
        result = control.analyze(args.output)
    print(json.dumps(result,indent=2))
