from argparse import ArgumentParser
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gruxnet import session_controls as control
from gruxnet.data import write_json

if __name__=="__main__":
    parser=ArgumentParser(description="Exploratory source-session selection and participant-separated stimulus transfer")
    parser.add_argument("action",choices=["plan","run","verify","analyze"])
    parser.add_argument("--plan",type=Path)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    if args.action=="plan":
        if args.plan.exists(): raise FileExistsError("Existing plan")
        write_json(args.plan,control.plan()); result={"saved":str(args.plan)}
    elif args.action=="run": result=control.run(args.cache,args.output,args.plan,args.device)
    elif args.action=="verify": result=control.verify(args.cache,args.output,args.device)
    else: result=control.analyze(args.output)
    print(json.dumps(result,indent=2))
